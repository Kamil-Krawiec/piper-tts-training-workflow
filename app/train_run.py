"""Resolve automatic settings, probe CUDA safely, and supervise one Piper run."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

from app.performance import (GPU_BATCH_CANDIDATES, GPU_MEMORY_MARGIN, available_ram_bytes, cpu_batch_for, cpu_snapshot,
                             cpu_threads_for, effective_cpu_count, gpu_snapshot,
                             physical_cpu_count, select_batch_size, workers_for)
from app.training import build_training_command, training_pythonpath
from app.datasets import saved_split_indices


def _save_config(path: Path, config: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _environment(config: dict[str, Any]) -> dict[str, str]:
    threads = str(config["torch_threads"])
    return {**os.environ, "PYTHONPATH": training_pythonpath(), "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
            "PIPER_TORCH_THREADS": threads, "PIPER_TORCH_INTEROP_THREADS": str(config["torch_interop_threads"]),
            "PIPER_RUN_CONFIG_PATH": str(Path(config["run_dir"]) / "run-config.json")}


def _probe(config: dict[str, Any], environment: dict[str, str], train_count: int) -> tuple[int, list[dict[str, Any]]]:
    run_dir = Path(config["run_dir"])
    gpu = gpu_snapshot() or {}
    total_vram = gpu.get("memory_total_bytes")
    if not total_vram:
        import torch
        total_vram = torch.cuda.get_device_properties(0).total_memory
    config["gpu"] = gpu.get("name")
    config["gpu_vram_bytes"] = total_vram
    outcomes: list[dict[str, Any]] = []
    for size in config.get("gpu_batch_candidates", GPU_BATCH_CANDIDATES):
        size = int(size)
        if size > train_count:
            outcomes.append({"size": size, "outcome": "skipped: exceeds training sample count"})
            continue
        trial_dir = run_dir / "probe" / str(size)
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial = {**config, "run_dir": str(trial_dir), "batch_size": size, "probe": True,
                 "probe_batches": 4, "num_workers": min(2, config["num_workers"])}
        command = build_training_command(trial)
        print(f"Automatic batch-size probe: {size}", flush=True)
        trial_environment = {key: value for key, value in environment.items() if key != "PIPER_RUN_CONFIG_PATH"}
        with (trial_dir / "train.log").open("wb") as log:
            result = subprocess.run(command, cwd=trial_dir, env=trial_environment, stdout=log, stderr=subprocess.STDOUT)
        log_text = (trial_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        probe_result = trial_dir / "probe-result.json"
        metrics = json.loads(probe_result.read_text()) if probe_result.is_file() else {}
        peak = metrics.get("peak_device_used_bytes", metrics.get("peak_vram_bytes"))
        if result.returncode == 0 and peak is not None:
            outcome = "pass"
        elif "out of memory" in log_text.lower() or "OutOfMemoryError" in log_text:
            outcome = "oom"
        else:
            raise RuntimeError(f"Batch probe {size} failed for a reason other than CUDA memory; see {trial_dir / 'train.log'}")
        outcomes.append({"size": size, "outcome": outcome, "peak_vram_bytes": peak})
        print(f"  {size}: {outcome}, peak VRAM {peak} bytes", flush=True)
        if outcome == "oom":
            break
    selected = select_batch_size([(item["size"], item["outcome"], item.get("peak_vram_bytes"))
                                  for item in outcomes], total_vram, config.get("gpu_memory_margin", GPU_MEMORY_MARGIN))
    print(f"Selected batch size: {selected}", flush=True)
    return selected, outcomes


def _sample_hardware(pid: int, path: Path, stop: threading.Event) -> None:
    previous = None
    while not stop.is_set():
        cpu, previous = cpu_snapshot(pid, previous)
        item = {"timestamp": time.time(), "cpu": cpu, "gpu": gpu_snapshot()}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
        stop.wait(5)


def run(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["run_dir"])
    logical = effective_cpu_count()
    physical = physical_cpu_count()
    with Path(config["csv_path"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="|"))
    samples = len(rows)
    if config.get("split_mode") == "saved":
        indices = saved_split_indices(Path(config["dataset_dir"]))
        config["split_counts"] = {name: len(values) for name, values in indices.items()}
        train_count = config["split_counts"]["train"]
    else:
        train_count = samples - int(samples * config.get("validation_split", 0.1)) - int(config.get("num_test_examples", 5))
    if train_count <= 0:
        raise ValueError("Dataset has no training samples after Piper's validation and test split")
    config["training_samples"] = train_count
    auto_workers = config.get("num_workers_mode", "auto") == "auto"
    config["num_workers"] = workers_for(config["device"], physical, logical, train_count) if auto_workers else int(config["num_workers"])
    auto_threads = config.get("torch_threads_mode", "auto") == "auto"
    config["torch_threads"] = cpu_threads_for(config["device"], physical, logical, config["num_workers"]) if auto_threads else int(config["torch_threads"])
    config["torch_interop_threads"] = 1
    config["logical_cores"] = logical
    config["physical_cores"] = physical
    if config.get("batch_size_mode", "manual") == "auto":
        if config["device"] == "cuda":
            config["batch_size"], config["batch_probe_results"] = _probe(config, _environment(config), train_count)
        else:
            longest = 0.0
            try:
                for row in rows:
                    with wave.open(str(Path(config.get("audio_dir", "")) / row[0]), "rb") as audio:
                        longest = max(longest, audio.getnframes() / audio.getframerate())
            except (OSError, wave.Error, IndexError, ZeroDivisionError):
                longest = float("inf")
            config["longest_utterance_seconds"] = longest if math.isfinite(longest) else None
            config["available_ram_bytes"] = available_ram_bytes()
            config["batch_size"] = cpu_batch_for(train_count, config["available_ram_bytes"], longest)
            config["batch_probe_results"] = []
    config["steps_per_epoch"] = math.ceil(train_count / config["batch_size"])
    config["total_optimizer_steps"] = config["steps_per_epoch"] * config["max_epochs"] * 2
    config["command"] = build_training_command(config)
    config["resolution_status"] = "complete"
    _save_config(config_path, config)
    print(f"Training {config['training_mode']} for up to {config['max_epochs']} epochs on {config['device']}; "
          f"batch size {config['batch_size']}, workers {config['num_workers']}, PyTorch threads {config['torch_threads']}", flush=True)
    with (run_dir / "train.log").open("ab") as log:
        child = subprocess.Popen(config["command"], cwd=run_dir, env=_environment(config), stdout=log,
                                 stderr=subprocess.STDOUT)
        stop = threading.Event()
        sampler = threading.Thread(target=_sample_hardware, args=(child.pid, run_dir / "hardware-metrics.jsonl", stop), daemon=True)
        sampler.start()
        try:
            return child.wait()
        finally:
            stop.set()
            sampler.join(timeout=6)


if __name__ == "__main__":
    try:
        sys.exit(run(Path(sys.argv[1])))
    except Exception as error:
        print(f"Training setup failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
