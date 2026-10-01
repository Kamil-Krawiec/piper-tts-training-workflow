"""Piper training configuration, validation, and process jobs."""

from __future__ import annotations

import json
import csv
import os
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.performance import epoch_cap_for


def training_pythonpath() -> str:
    """Make callback modules importable when Piper runs from a persisted run directory."""
    return str(Path(__file__).resolve().parent.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")


def batch_size_for(device: str) -> int:
    # CPU has no VRAM probe; keep a modest default until throughput is measured.
    return 8 if device == "cuda" else 4


def build_training_command(config: dict[str, Any]) -> list[str]:
    required = ("voice_name", "csv_path", "audio_dir", "sample_rate", "espeak_voice", "cache_dir", "config_path", "device", "batch_size")
    missing = [key for key in required if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"missing training settings: {', '.join(missing)}")
    mode = config.get("training_mode")
    if mode not in {"finetune", "scratch"}:
        raise ValueError("training_mode must be 'finetune' or 'scratch'")
    run_dir = Path(config.get("run_dir", Path(config["config_path"]).parent))
    csv_logger = {
        "class_path": "lightning.pytorch.loggers.CSVLogger",
        "init_args": {"save_dir": str(run_dir), "name": "metrics", "version": 0, "flush_logs_every_n_steps": 1},
    }
    command = [
        sys.executable, str(Path(__file__).with_name("train_cli.py")), "fit",
        "--data.voice_name", str(config["voice_name"]),
        "--data.csv_path", str(config["csv_path"]),
        "--data.audio_dir", str(config["audio_dir"]),
        "--model.sample_rate", str(int(config["sample_rate"])),
        "--data.espeak_voice", str(config["espeak_voice"]),
        "--data.cache_dir", str(config["cache_dir"]),
        "--data.config_path", str(config["config_path"]),
        "--data.batch_size", str(int(config["batch_size"])),
        "--data.num_workers", str(int(config.get("num_workers", 1))),
        "--data.validation_split", str(float(config.get("validation_split", 0.1))),
        "--data.num_test_examples", str(int(config.get("num_test_examples", 5))),
        "--trainer.accelerator", "gpu" if config["device"] == "cuda" else "cpu",
        "--trainer.devices", "1",
        "--trainer.max_epochs", str(1 if config.get("probe") else int(config.get("max_epochs", epoch_cap_for(mode)))),
        "--trainer.default_root_dir", str(run_dir),
        "--trainer.logger", json.dumps(csv_logger),
        "--trainer.log_every_n_steps", "10",
        "--trainer.enable_progress_bar", "false",
        "--seed_everything", str(int(config.get("seed", 42))),
    ]
    if config.get("probe"):
        callbacks = [{"class_path": "app.training_metrics.ProbeMetrics", "init_args": {
            "path": str(run_dir / "probe-result.json")}}]
        command.extend(("--trainer.limit_train_batches", str(int(config.get("probe_batches", 12))),
                        "--trainer.limit_val_batches", "0", "--trainer.num_sanity_val_steps", "0",
                        "--trainer.enable_checkpointing", "false", "--trainer.callbacks", json.dumps(callbacks)))
    else:
        callbacks = [{"class_path": "app.training_metrics.TrainingMetrics", "init_args": {
            "path": str(run_dir / "training-metrics.json"),
            "steps_per_epoch": int(config.get("steps_per_epoch", 0)),
            "max_epochs": int(config.get("max_epochs", epoch_cap_for(mode))),
            "hourly_rate": config.get("gpu_hourly_rate")}}]
        callbacks.append({"class_path": "lightning.pytorch.callbacks.ModelCheckpoint", "init_args": {
            "dirpath": str(run_dir / "checkpoints"), "filename": "epoch-{epoch:04d}",
            "auto_insert_metric_name": False, "every_n_epochs": int(config.get("checkpoint_interval", 250)),
            "save_top_k": -1, "save_last": False}})
        callbacks.append({"class_path": "lightning.pytorch.callbacks.ModelCheckpoint", "init_args": {
            "dirpath": str(run_dir / "checkpoints" / "latest"), "filename": "rolling",
            "every_n_epochs": 25, "save_top_k": 0, "save_last": True}})
        command.extend(("--trainer.callbacks", json.dumps(callbacks)))
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if mode == "finetune":
        checkpoint = config.get("checkpoint")
        if not checkpoint:
            raise ValueError("fine-tuning requires a checkpoint")
        command.extend(("--ckpt_path", str(checkpoint)))
    elif config.get("vocoder_warmstart_checkpoint"):
        command.extend(("--model.vocoder_warmstart_ckpt", str(config["vocoder_warmstart_checkpoint"])))
    return command


def validate_training_config(config: dict[str, Any], cuda_available: bool | None = None) -> list[str]:
    errors: list[str] = []
    dataset = Path(config.get("dataset_dir", "."))
    if not dataset.is_dir():
        errors.append("Selected dataset folder does not exist.")
    if not (dataset / "metadata.csv").is_file():
        errors.append("Dataset is missing metadata.csv.")
    if not Path(config.get("csv_path", dataset / "train_metadata.csv")).is_file():
        errors.append("Dataset is missing its Piper training metadata file.")
    audio_dir = dataset / "audio"
    if not audio_dir.is_dir() or not any(audio_dir.glob("*.wav")):
        errors.append("Dataset audio folder has no WAV files.")
    metadata = Path(config.get("csv_path", ""))
    if metadata.is_file() and audio_dir.is_dir():
        try:
            with metadata.open(encoding="utf-8", newline="") as stream:
                entries = list(csv.reader(stream, delimiter="|"))
            if not entries:
                errors.append("Piper training metadata is empty.")
            for row in entries:
                if len(row) != 2 or not row[1].strip():
                    errors.append("Piper metadata rows must contain an audio filename and non-empty text.")
                    break
                relative = Path(row[0])
                audio_path = (audio_dir / relative).resolve()
                if relative.is_absolute() or not audio_path.is_relative_to(audio_dir.resolve()) or not audio_path.is_file():
                    errors.append(f"Piper metadata references a missing or unsafe audio path: {row[0]}")
                    break
                try:
                    with wave.open(str(audio_path), "rb") as wav:
                        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != int(config.get("sample_rate", 0)):
                            errors.append(f"Piper audio must be mono 16-bit at {config.get('sample_rate')} Hz: {row[0]}")
                            break
                except wave.Error:
                    errors.append(f"Piper metadata references an invalid WAV file: {row[0]}")
                    break
        except (OSError, UnicodeError, csv.Error):
            errors.append("Could not read Piper training metadata.")
    if int(config.get("sample_rate", 0)) != 22050:
        errors.append("Piper medium datasets must be configured at 22050 Hz.")
    if not config.get("espeak_voice"):
        errors.append("Select an eSpeak voice.")
    if config.get("training_mode") not in {"finetune", "scratch"}:
        errors.append("Select a training mode.")
    if config.get("training_mode") == "finetune" and not Path(config.get("checkpoint") or "").is_file():
        errors.append("Fine-tuning requires an existing checkpoint file.")
    if config.get("vocoder_warmstart_checkpoint") and not Path(config["vocoder_warmstart_checkpoint"]).is_file():
        errors.append("The vocoder warm-start checkpoint file does not exist.")
    if config.get("device") == "cuda" and cuda_available is False:
        errors.append("CUDA was selected, but it is unavailable inside this container.")
    try:
        epochs = int(config.get("max_epochs", 1000))
        if epochs < 1 or epochs > 100_000:
            errors.append("Maximum epochs must be between 1 and 100000.")
    except (TypeError, ValueError):
        errors.append("Maximum epochs must be a positive integer.")
    if not shutil.which("espeak-ng"):
        errors.append("eSpeak NG is not installed in the trainer container.")
    else:
        try:
            result = subprocess.run(["espeak-ng", f"--voices={config.get('espeak_voice', '')}"], capture_output=True, text=True, timeout=5)
            if result.returncode or not result.stdout.strip():
                errors.append(f"eSpeak voice '{config.get('espeak_voice', '')}' is not available.")
        except (OSError, subprocess.TimeoutExpired):
            errors.append("Could not check the selected eSpeak voice.")
    try:
        build_training_command(config)
    except (TypeError, ValueError) as error:
        if str(error) not in errors:
            errors.append(str(error))
    return errors


class TrainingJobs:
    """One persisted subprocess job per project, with restart-aware status."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen[Any]] = {}

    def _run_dir(self, project_id: str, run_id: str) -> Path:
        if not project_id.replace("-", "").isalnum() or not run_id.replace("-", "").isalnum():
            raise ValueError("invalid project or run id")
        return self.data_dir / "projects" / project_id / "runs" / run_id

    def start(self, project_id: str, config: dict[str, Any], run_id: str | None = None) -> str:
        with self._lock:
            current = self.status(project_id)
            if current and current["status"] in {"preparing", "training", "queued"}:
                raise RuntimeError("a training job is already active for this project")
            run_id = run_id or str(uuid.uuid4())
            run_dir = self._run_dir(project_id, run_id)
            run_dir.mkdir(parents=True, exist_ok=False)
            command = build_training_command({**config, "run_dir": run_dir})
            actual = {**config, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                      "resolution_status": "pending", "command": command}
            (run_dir / "run-config.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"status": "preparing", "updated_at": time.time()}), encoding="utf-8")
            with (run_dir / "train.log").open("ab") as log_file:
                launch = [sys.executable, "-m", "app.train_run", str(run_dir / "run-config.json")]
                environment = {**os.environ, "PYTHONPATH": training_pythonpath()}
                process = subprocess.Popen(launch, cwd=run_dir, env=environment, stdout=log_file,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            self._processes[run_id] = process
            (run_dir / "pid").write_text(str(process.pid), encoding="ascii")
            status = {"status": "training", "pid": process.pid, "updated_at": time.time(), "command": command}
            (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            return run_id

    def status(self, project_id: str) -> dict[str, Any] | None:
        root = self.data_dir / "projects" / project_id / "runs"
        if not root.exists():
            return None
        for run_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True):
            status_file = run_dir / "status.json"
            if status_file.exists():
                state = json.loads(status_file.read_text(encoding="utf-8"))
                pid = state.get("pid")
                process = self._processes.get(run_dir.name)
                return_code = process.poll() if process is not None else None
                process_ended = (process is not None and return_code is not None) or (process is None and pid and not _process_exists(pid))
                if state.get("status") == "training" and process_ended:
                    has_checkpoint = any(run_dir.rglob("*.ckpt"))
                    state["status"] = "completed" if (return_code == 0 or process is None) and has_checkpoint else "failed"
                    state["message"] = "Training process ended. Inspect saved logs and checkpoints."
                    if return_code is not None:
                        state["exit_code"] = return_code
                    status_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
                state["run_id"] = run_dir.name
                state["log_tail"] = _tail(run_dir / "train.log")
                state["config"] = json.loads((run_dir / "run-config.json").read_text(encoding="utf-8"))
                state["progress"] = _training_progress(run_dir, state["config"])
                metrics = run_dir / "training-metrics.json"
                if metrics.is_file():
                    try:
                        state["performance"] = json.loads(metrics.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        pass
                hardware = run_dir / "hardware-metrics.jsonl"
                if hardware.is_file():
                    try:
                        recent = [json.loads(line) for line in _tail(hardware, 10).splitlines() if line.strip()]
                        state["hardware"] = recent[-1] if recent else None
                        values = [item["gpu"]["compute_percent"] for item in recent if item.get("gpu") and item["gpu"].get("compute_percent") is not None]
                        state["gpu_compute_average"] = round(sum(values) / len(values), 1) if values else None
                    except (OSError, ValueError):
                        pass
                return state
        return None

    def cancel(self, project_id: str) -> bool:
        state = self.status(project_id)
        if not state or state.get("status") != "training" or not state.get("pid"):
            return False
        try:
            os.killpg(int(state["pid"]), signal.SIGTERM)
        except ProcessLookupError:
            return False
        run_dir = self._run_dir(project_id, state["run_id"])
        (run_dir / "status.json").write_text(json.dumps({**state, "status": "cancelled", "updated_at": time.time()}), encoding="utf-8")
        return True


def _process_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, ValueError):
        return False


def _tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 128 * 1024))
        content = stream.read().decode("utf-8", "replace")
    return "\n".join(content.splitlines()[-lines:])


def _training_progress(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read the trainer's durable CSV metrics; an incomplete last row is ignored."""
    maximum = int(config.get("max_epochs", 0))
    result: dict[str, Any] = {"max_epochs": maximum, "current_epoch": 0, "completed_epochs": 0,
                              "global_step": 0, "steps_per_epoch": int(config.get("steps_per_epoch", 0)),
                              "total_optimizer_steps": int(config.get("total_optimizer_steps", 0)), "losses": []}
    path = run_dir / "metrics" / "version_0" / "metrics.csv"
    if not path.is_file():
        return result
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("epoch"):
                    continue
                epoch = int(float(row["epoch"])) + 1
                result["current_epoch"] = max(result["current_epoch"], epoch)
                if row.get("step"):
                    result["global_step"] = max(result["global_step"], int(float(row["step"])))
                for key, label in (("loss_g", "Training"), ("val_loss", "Validation")):
                    if row.get(key):
                        if key == "val_loss":
                            result["completed_epochs"] = max(result["completed_epochs"], epoch)
                        result["losses"].append({"epoch": epoch, "loss": float(row[key]), "series": label})
    except (OSError, UnicodeError, ValueError, csv.Error):
        return result
    result["losses"] = result["losses"][-300:]
    return result
