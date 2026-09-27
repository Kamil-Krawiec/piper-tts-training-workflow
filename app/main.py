"""Gradio UI for the local Piper voice workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr

from app.audio import inspect_wav, normalize_audio
from app.bundles import export_bundle, import_bundle
from app.checkpoints import curated_checkpoints, download_checkpoint
from app.datasets import create_dataset
from app.export import export_onnx, package_model, publish_model, validate_voice_name
from app.inference import api_health, api_synthesize, synthesize
from app.projects import ProjectStore
from app.text import estimate_text, parse_prompts, prompt_recommendation
from app.training import TrainingJobs, batch_size_for, validate_training_config


DATA_DIR = Path(os.environ.get("PIPER_DATA_DIR", "/data"))
STORE = ProjectStore(DATA_DIR)
JOBS = TrainingJobs(DATA_DIR)
BUILTIN_PROMPTS = Path("/app/prompts/pl_PL_demo.txt")
PROMPT_DIR = Path("/app/prompts") if Path("/app/prompts").exists() else Path(__file__).resolve().parent.parent / "prompts"


def _projects() -> list[dict[str, Any]]:
    return STORE.list_projects()


def _project_choices() -> list[tuple[str, str]]:
    return [(f"{item['name']} ({item['language']})", item["id"]) for item in _projects()]


def _dataset_dirs(project_id: str) -> list[Path]:
    root = STORE.project_dir(project_id) / "datasets"
    return sorted([path for path in root.iterdir() if path.is_dir() and (path / "metadata.csv").exists()], key=lambda path: path.name) if root.exists() else []


def _dataset_choices(project_id: str) -> list[tuple[str, str]]:
    choices = []
    for path in _dataset_dirs(project_id):
        manifest_path = path / "dataset.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            label = f"{path.name} — {manifest.get('sample_count', '?')} samples / {manifest.get('total_seconds', 0) / 60:.1f} min"
        else:
            label = path.name
        choices.append((label, path.name))
    return choices


def _selected_dataset(project_id: str, dataset_name: str) -> Path:
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("Select a dataset")
    path = STORE.project_dir(project_id) / "datasets" / dataset_name
    if not path.is_dir():
        raise ValueError("Selected dataset was not found")
    return path


def evaluation_choices(project_id: str, dataset_name: str):
    if not project_id or not dataset_name:
        return []
    dataset = _selected_dataset(project_id, dataset_name)
    index_path = dataset / "sample-index.json"
    if not index_path.is_file():
        return []
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    return [(f"{row['sample_id']}: {row['text'][:90]}", row["sample_id"]) for row in rows if row.get("split") == "test"]


def load_evaluation_prompt(project_id: str, dataset_name: str, sample_id: str):
    try:
        dataset = _selected_dataset(project_id, dataset_name)
        rows = json.loads((dataset / "sample-index.json").read_text(encoding="utf-8"))
        prompt = next(row["text"] for row in rows if row["sample_id"] == sample_id and row["split"] == "test")
        audio = dataset / "audio" / next(row["filename"] for row in rows if row["sample_id"] == sample_id)
        return prompt, str(audio), "Loaded fixed held-out test prompt and reference recording. This sample is excluded from Piper's training metadata."
    except Exception as error:
        return "", None, f"Could not load test prompt: {error}"


def _sample_rows(project_id: str, status_filter: str = "all", query: str = "") -> list[list[Any]]:
    rows = []
    needle = query.casefold().strip()
    for sample in STORE.list_samples(project_id):
        if status_filter != "all" and sample["status"] != status_filter:
            continue
        if needle and needle not in sample["text"].casefold():
            continue
        quality = sample["quality"]
        rows.append([
            sample["id"], sample["text"], sample["status"], round(sample["duration_seconds"], 2),
            quality.get("peak_dbfs", ""), "; ".join(quality.get("warnings", [])), sample.get("audio_file") or sample["raw_file"],
        ])
    return rows


def _project_summary(project_id: str | None) -> str:
    if not project_id:
        return "Create or select a project to begin."
    project = STORE.get_project(project_id)
    counts = project["sample_counts"]
    accepted = counts.get("accepted", {}).get("seconds", 0)
    prompt_total = len(project["prompts"])
    samples = [sample for sample in STORE.list_samples(project_id) if sample["status"] != "superseded"]
    recorded = len({sample["prompt_id"] for sample in samples})
    accepted_count = counts.get("accepted", {}).get("count", 0)
    rejected_count = counts.get("rejected", {}).get("count", 0)
    review_count = counts.get("review", {}).get("count", 0)
    coverage = min(100, int(accepted / 3600 * 100))
    return (
        f"### {project['name']}\n"
        f"Language: `{project['language']}` · eSpeak: `{project['espeak_voice']}` · Sample rate: `22050 Hz`\n\n"
        f"Prompts: **{prompt_total}** · Recorded: **{recorded} / {prompt_total}** · "
        f"Accepted: **{accepted_count}** · Review: **{review_count}** · Rejected: **{rejected_count}**\n\n"
        f"Accepted audio: **{_clock(accepted)}** · 30 min target: **{'reached' if accepted >= 1800 else f'{accepted / 1800 * 100:.0f}%'}** · "
        f"60 min target: **{coverage}%**"
    )


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02}:{(total % 3600) // 60:02}:{total % 60:02}"


def _queue_rows(project_id: str, search: str = "") -> list[list[Any]]:
    rows = []
    needle = search.casefold().strip()
    seen: set[str] = set()
    latest_by_prompt = {}
    for sample in STORE.list_samples(project_id):
        if sample["status"] != "superseded":
            latest_by_prompt[sample["prompt_id"]] = sample
    for prompt in STORE.get_project(project_id)["prompts"]:
        if needle and needle not in prompt["text"].casefold():
            continue
        recommendation = prompt_recommendation(prompt["text"], seen)
        seen.add(prompt["text"].casefold())
        sample = latest_by_prompt.get(prompt["id"])
        status = sample["status"].capitalize() if sample else recommendation
        if sample and recommendation != "Ready":
            status += f" · {recommendation}"
        rows.append([prompt["id"], prompt["text"], len(prompt["text"].split()), status])
    return rows


def _source_content(upload: str | None, pasted: str) -> tuple[str, str]:
    if upload:
        path = Path(upload)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Text source exceeds the 2 MiB upload limit")
        return path.read_text(encoding="utf-8-sig"), path.name
    return pasted, "pasted.txt"


def create_project(name: str, language: str, espeak: str):
    try:
        project = STORE.create_project(name, language.strip() or "pl_PL", espeak.strip() or "pl")
        return gr.update(choices=_project_choices(), value=project["id"]), _project_summary(project["id"]), "Project created."
    except Exception as error:
        return gr.update(), "", f"Could not create project: {error}"


def select_project(project_id: str | None):
    if not project_id:
        return "Create or select a project.", [], gr.update(choices=[], value=None), [], "", [], 0, "", "0 / 0", None
    project = STORE.get_project(project_id)
    datasets = _dataset_choices(project_id)
    active_id = STORE.get_active_prompt(project_id)
    prompts = project["prompts"]
    active_index = next((index for index, item in enumerate(prompts) if item["id"] == active_id), 0)
    cursor = current_prompt(project_id, active_index)
    return (
        _project_summary(project_id), _queue_rows(project_id),
        gr.update(choices=datasets, value=datasets[0][1] if datasets else None),
        _sample_rows(project_id), "", _queue_rows(project_id), *cursor,
    )


def load_app_state(current_project_id: str | None):
    choices = _project_choices()
    available = {value for _, value in choices}
    selected = current_project_id if current_project_id in available else (choices[0][1] if choices else None)
    return (gr.update(choices=choices, value=selected), *select_project(selected))


def preview_source(project_id: str, upload: str | None, pasted: str, mode: str, rate: int):
    if not project_id:
        return [], "Select a project first.", "", ""
    try:
        text, filename = _source_content(upload, pasted)
        prompts = parse_prompts(text, mode)
        if len(prompts) > 20_000:
            raise ValueError("Source creates more than 20,000 prompts; split it into smaller imports.")
        if upload and BUILTIN_PROMPTS.exists() and filename == BUILTIN_PROMPTS.name:
            filename = f"built-in:{filename}"
        ids = STORE.import_prompts(project_id, filename, text, mode)
        result = estimate_text(prompts, int(rate))
        mins = result["estimated_seconds"] / 60
        summary = (
            f"**Source:** {filename} · **Mode:** {mode}\n\n"
            f"Prompts: **{result['prompts']:,}** · Words: **{result['words']:,}** · Characters: **{result['characters']:,}**\n\n"
            f"Estimated spoken time: **~{mins:.1f} min** at {rate} words/min. "
            f"Estimated usable audio: **~{result['usable_min_seconds']/60:.1f}–{result['usable_max_seconds']/60:.1f} min**. "
            "This is an estimate, not a guaranteed recording duration."
        )
        rows = []
        seen: set[str] = set()
        for prompt_id, prompt in zip(ids, prompts):
            status = prompt_recommendation(prompt, seen)
            seen.add(prompt.casefold())
            rows.append([prompt_id, prompt, len(prompt.split()), status])
        return rows, summary, "Queue saved. Review or edit prompts before recording.", _project_summary(project_id)
    except Exception as error:
        return [], "", f"Could not parse source: {error}", _project_summary(project_id)


def preview_builtin():
    if not BUILTIN_PROMPTS.exists():
        return None, "Built-in Polish demo prompt pack was not found."
    return str(BUILTIN_PROMPTS), "Built-in Polish demo pack selected. Click Preview and save queue."


def select_prompt_pack(name: str | None):
    if not name or Path(name).name != name:
        return None, "Choose a prompt pack."
    path = PROMPT_DIR / name
    if path.suffix != ".txt" or not path.is_file():
        return None, "Prompt pack was not found."
    return str(path), f"Selected prompt pack: {name}. Click Preview and save queue."


def restore_source_queue(project_id: str):
    try:
        project = STORE.get_project(project_id)
        if not project["source"]:
            raise ValueError("This project has no imported source to restore.")
        source_file = STORE.project_dir(project_id) / "source" / "original.txt"
        text = source_file.read_text(encoding="utf-8")
        ids = STORE.import_prompts(project_id, project["source"]["filename"], text, project["source"]["mode"])
        prompts = parse_prompts(text, project["source"]["mode"])
        rows = [[prompt_id, prompt, len(prompt.split()), "Ready"] for prompt_id, prompt in zip(ids, prompts)]
        return rows, "Queue restored from the original imported source."
    except Exception as error:
        return _queue_rows(project_id), f"Could not restore source queue: {error}"


def save_queue(project_id: str, rows: list[list[Any]] | None):
    try:
        values = [{"id": str(row[0]), "text": str(row[1])} for row in (rows or []) if len(row) >= 2]
        STORE.save_prompt_queue(project_id, values)
        return _queue_rows(project_id), _project_summary(project_id), "Prompt queue saved."
    except Exception as error:
        return rows, _project_summary(project_id), f"Could not save prompt queue: {error}"


def prompt_action(project_id: str, prompt_index: int, action: str, left: str, right: str):
    try:
        if action == "up":
            STORE.move_prompt(project_id, prompt_index, -1)
        elif action == "down":
            STORE.move_prompt(project_id, prompt_index, 1)
        elif action == "delete":
            queue = STORE.get_project(project_id)["prompts"]
            STORE.delete_prompt(project_id, queue[prompt_index]["id"])
        elif action == "split":
            STORE.split_prompt(project_id, prompt_index, left, right)
        elif action == "merge":
            STORE.merge_prompts(project_id, prompt_index)
        return _queue_rows(project_id), _project_summary(project_id), "Queue updated."
    except Exception as error:
        return _queue_rows(project_id), _project_summary(project_id), f"Queue action failed: {error}"


def current_prompt(project_id: str, index: int):
    prompts = STORE.get_project(project_id)["prompts"]
    if not prompts:
        return 0, "Import or paste text to create a recording queue.", "0 / 0", None
    index = max(0, min(int(index), len(prompts) - 1))
    STORE.set_active_prompt(project_id, prompts[index]["id"])
    return index, prompts[index]["text"], f"Prompt {index + 1} of {len(prompts)}", prompts[index]["id"]


def resume_prompt(project_id: str):
    prompts = STORE.get_project(project_id)["prompts"]
    active_id = STORE.get_active_prompt(project_id)
    index = next((i for i, prompt in enumerate(prompts) if prompt["id"] == active_id), 0)
    return current_prompt(project_id, index)


def record_sample(project_id: str, index: int, prompt_id: str | None, recording_path: str | None, decision: str):
    if not project_id or not recording_path:
        return "Record audio with the microphone first.", None, _project_summary(project_id), _sample_rows(project_id) if project_id else [], index, _queue_rows(project_id) if project_id else []
    try:
        prompts = STORE.get_project(project_id)["prompts"]
        if not prompts:
            raise ValueError("No prompts are available")
        prompt_matches = [item for item in prompts if item["id"] == prompt_id]
        if not prompt_matches:
            raise ValueError("The displayed prompt was removed or changed. Select a prompt again before saving.")
        prompt = prompt_matches[0]
        raw_source = Path(recording_path)
        project_root = STORE.project_dir(project_id)
        suffix = raw_source.suffix.lower() or ".audio"
        sample_token = uuid.uuid4().hex
        raw_target = project_root / "recordings" / "raw" / f"{prompt['id']}-{sample_token}{suffix}"
        shutil.copy2(raw_source, raw_target)
        wav_target = project_root / "recordings" / "normalized" / f"{prompt['id']}-{sample_token}.wav"
        normalize_audio(raw_target, wav_target)
        quality = inspect_wav(wav_target)
        sample_status = "rejected" if decision == "reject" else "accepted" if decision == "accept" and quality["status"] == "ok" else "review"
        STORE.add_sample(project_id, prompt["id"], str(raw_target.relative_to(project_root)), quality["duration_seconds"], sample_status, quality, str(wav_target))
        current_index = next(position for position, item in enumerate(prompts) if item["id"] == prompt_id)
        next_index = min(current_index + (1 if decision == "accept" else 0), len(prompts) - 1)
        prompt_text = prompts[next_index]["text"] if prompts else ""
        state = f"{sample_status.upper()} · {_clock(quality['duration_seconds'])}\n" + ("\n".join(quality["warnings"]) if quality["warnings"] else "Quality checks look good.")
        return state, str(wav_target), _project_summary(project_id), _sample_rows(project_id), next_index, _queue_rows(project_id)
    except Exception as error:
        return f"Could not save recording: {error}", None, _project_summary(project_id), _sample_rows(project_id), index, _queue_rows(project_id)


def load_sample_audio(project_id: str, sample_id: str):
    try:
        for sample in STORE.list_samples(project_id):
            if sample["id"] == sample_id:
                path = Path(sample.get("audio_file") or "")
                if not path.is_file():
                    project_path = STORE.project_dir(project_id) / sample["raw_file"]
                    path = project_path if project_path.is_file() else path
                return str(path) if path.is_file() else None, f"{sample['text']} · {sample['duration_seconds']:.2f}s · {sample['status']}"
        raise KeyError("sample not found")
    except Exception as error:
        return None, f"Could not load sample: {error}"


def review_sample(project_id: str, sample_id: str, status: str):
    try:
        STORE.set_sample_status(project_id, sample_id, status)
        return _sample_rows(project_id), _project_summary(project_id), f"Sample marked {status}.", _queue_rows(project_id)
    except Exception as error:
        return _sample_rows(project_id), _project_summary(project_id), str(error), _queue_rows(project_id)


def make_dataset(project_id: str, target: str, custom_minutes: int = 30):
    try:
        sample_rows = STORE.list_samples(project_id)
        target_seconds = {"15 minutes": 15 * 60, "30 minutes": 30 * 60, "60 minutes": 60 * 60, "All accepted": None}.get(target)
        if target == "Custom duration":
            target_seconds = int(custom_minutes) * 60
        if not any(item["status"] == "accepted" and item.get("audio_file") for item in sample_rows):
            raise ValueError("Accept at least one normalized recording before creating a dataset.")
        dataset_id = f"dataset-{target.replace(' ', '').replace('minutes','m')}-{uuid.uuid4().hex[:8]}"
        output = STORE.project_dir(project_id) / "datasets" / dataset_id
        manifest = create_dataset(sample_rows, output, target_seconds, seed=42)
        project = STORE.get_project(project_id)
        info = {key: project[key] for key in ("name", "language", "espeak_voice")}
        info.update({"sample_rate": 22050, "dataset_id": dataset_id})
        (output / "project-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        minutes = manifest["total_seconds"] / 60
        choices = _dataset_choices(project_id)
        return gr.update(choices=choices, value=dataset_id), gr.update(choices=choices, value=dataset_id), f"Created {manifest['sample_count']} samples ({minutes:.1f} min), seed 42.", _project_summary(project_id)
    except Exception as error:
        return gr.update(), gr.update(), f"Dataset creation failed: {error}", _project_summary(project_id)


def export_dataset_ui(project_id: str, dataset_name: str):
    try:
        dataset = _selected_dataset(project_id, dataset_name)
        info_file = dataset / "project-info.json"
        info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
        archive = DATA_DIR / "exports" / f"{STORE.get_project(project_id)['name'].lower().replace(' ', '-')}-{dataset.name}.piper-dataset.zip"
        return str(export_bundle(dataset, archive, info)), "Dataset bundle is ready to download."
    except Exception as error:
        return None, f"Dataset export failed: {error}"


def import_dataset_ui(project_id: str, archive_path: str | None):
    try:
        if not archive_path:
            raise ValueError("Select a dataset ZIP")
        target = STORE.project_dir(project_id) / "datasets" / f"imported-{uuid.uuid4().hex[:8]}"
        import_bundle(archive_path, target)
        choices = _dataset_choices(project_id)
        update = gr.update(choices=choices, value=target.name)
        return update, update, "Dataset bundle validated and imported."
    except Exception as error:
        return gr.update(), gr.update(), f"Dataset import failed: {error}"


def download_base_checkpoint(checkpoint_id: str):
    try:
        checkpoint = download_checkpoint(checkpoint_id, DATA_DIR / "checkpoints")
        return str(checkpoint), f"Cached checkpoint: {checkpoint}"
    except Exception as error:
        return "", f"Checkpoint download failed: {error}"


def save_uploaded_checkpoint(path: str | None):
    try:
        if not path:
            raise ValueError("Choose a .ckpt file to upload")
        source = Path(path)
        if source.suffix != ".ckpt":
            raise ValueError("Checkpoint upload must be a .ckpt file")
        target = DATA_DIR / "checkpoints" / "uploaded" / f"{uuid.uuid4().hex}-{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return str(target), "Checkpoint copied into persistent local storage."
    except Exception as error:
        return "", f"Checkpoint upload failed: {error}"


def start_training(project_id: str, dataset_name: str, mode: str, checkpoint_path: str, warmstart_path: str, device: str, batch_size: int, seed: int, max_epochs: int):
    try:
        dataset = _selected_dataset(project_id, dataset_name)
        selected_device = device
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if selected_device == "auto":
            selected_device = "cuda" if has_cuda else "cpu"
        dataset_info_path = dataset / "project-info.json"
        project_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
        project = STORE.get_project(project_id)
        run_id = str(uuid.uuid4())
        run_dir = STORE.project_dir(project_id) / "runs" / run_id
        config = {
            "dataset_id": dataset.name,
            "dataset_manifest_hash": _hash_if_exists(dataset / "dataset.json"),
            "dataset_dir": str(dataset), "voice_name": validate_voice_name(project["name"].replace(" ", "-")),
            "sample_rate": int(project_info.get("sample_rate", 22050)),
            "espeak_voice": project_info.get("espeak_voice", project["espeak_voice"]),
            "csv_path": str(dataset / ("train_metadata.csv" if (dataset / "train_metadata.csv").is_file() else "metadata.csv")), "audio_dir": str(dataset / "audio"),
            "cache_dir": str(run_dir / "cache"), "config_path": str(run_dir / "voice.onnx.json"),
            "run_dir": str(run_dir), "training_mode": mode,
            "checkpoint": (checkpoint_path or None) if mode == "finetune" else None,
            "checkpoint_id": ("pl_PL-darkman-medium" if checkpoint_path and "pl_PL-darkman-medium" in checkpoint_path else "local-or-uploaded") if mode == "finetune" and checkpoint_path else None,
            "vocoder_warmstart_checkpoint": warmstart_path or None, "device": selected_device,
            "batch_size": int(batch_size) if int(batch_size) > 0 else batch_size_for(selected_device), "seed": int(seed), "max_epochs": int(max_epochs),
            "validation_split": 0.1, "num_test_examples": 5,
            "piper_revision": "v1.3.0", "base_checkpoint_hash": _hash_if_exists(Path(checkpoint_path)) if mode == "finetune" and checkpoint_path else None,
        }
        errors = validate_training_config(config, cuda_available=has_cuda)
        if errors:
            raise ValueError("\n".join(errors))
        actual_run_id = JOBS.start(project_id, config)
        return f"Started {mode} run {actual_run_id} on {selected_device}.", _run_status_text(project_id)
    except Exception as error:
        return f"Training not started: {error}", _run_status_text(project_id) if project_id else ""


def _hash_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_status_text(project_id: str):
    status = JOBS.status(project_id)
    if not status:
        return "No training runs yet."
    return f"Run: {status['run_id']} · Status: {status['status']}\n{status.get('log_tail','')}"


def cancel_training(project_id: str):
    cancelled = JOBS.cancel(project_id)
    return "Training process stopped." if cancelled else "No active training process to stop.", _run_status_text(project_id)


def run_choices(project_id: str):
    root = STORE.project_dir(project_id) / "runs"
    return [(path.name, path.name) for path in sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True) if path.is_dir()] if root.exists() else []


def inspect_run_config(project_id: str, run_id: str):
    try:
        path = STORE.project_dir(project_id) / "runs" / Path(run_id).name / "run-config.json"
        if not path.is_file():
            raise ValueError("Run configuration is unavailable")
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    except Exception as error:
        return f"Could not read run configuration: {error}"


def compare_run_configs(project_id: str, run_a: str, run_b: str):
    try:
        config_a = json.loads(inspect_run_config(project_id, run_a))
        config_b = json.loads(inspect_run_config(project_id, run_b))
        fields = (
            ("Dataset", "dataset_id"), ("Dataset manifest hash", "dataset_manifest_hash"),
            ("Training mode", "training_mode"), ("Base checkpoint SHA-256", "base_checkpoint_hash"),
            ("Random seed", "seed"), ("Piper model name", "voice_name"),
            ("Sample rate", "sample_rate"), ("eSpeak voice", "espeak_voice"),
            ("Internal validation split", "validation_split"), ("Test examples", "num_test_examples"),
            ("Piper revision", "piper_revision"),
        )
        rows = [f"| Setting | Run A | Run B | Same |", "|---|---|---|---|"]
        for label, key in fields:
            left, right = config_a.get(key, "—"), config_b.get(key, "—")
            rows.append(f"| {label} | `{left}` | `{right}` | {'yes' if left == right else 'no'} |")
        return "\n".join(rows)
    except Exception as error:
        return f"Could not compare runs: {error}"


def export_run(project_id: str, run_id: str, voice_name: str):
    try:
        run_dir = STORE.project_dir(project_id) / "runs" / Path(run_id).name
        if not run_dir.is_dir():
            raise ValueError("Select a training run")
        checkpoints = list(run_dir.rglob("*.ckpt"))
        checkpoints = [path for path in checkpoints if "last" not in path.name.lower()]
        if not checkpoints:
            checkpoints = list(run_dir.rglob("*.ckpt"))
        if not checkpoints:
            raise ValueError("No training checkpoint is available in this run yet")
        config = run_dir / "voice.onnx.json"
        output = STORE.project_dir(project_id) / "models" / validate_voice_name(voice_name)
        model, config_file = export_onnx(max(checkpoints, key=lambda path: path.stat().st_mtime), config, output, voice_name)
        archive = DATA_DIR / "exports" / f"{voice_name}.piper-model.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        package_model(model, config_file, archive)
        choices = model_choices(project_id)
        return str(archive), f"Exported model pair: {model.name} ({model.stat().st_size:,} bytes) and {config_file.name} ({config_file.stat().st_size:,} bytes).", gr.update(choices=choices, value=str(model))
    except Exception as error:
        return None, f"ONNX export failed: {error}", gr.update()


def model_choices(project_id: str):
    root = STORE.project_dir(project_id) / "models"
    pairs = []
    if root.exists():
        for config in root.rglob("*.onnx.json"):
            model = config.with_name(config.name[:-5])
            if model.exists():
                pairs.append((model.stem, str(model)))
    return pairs


def synthesize_model(model_path: str, text: str):
    try:
        model = Path(model_path)
        output = DATA_DIR / "exports" / f"test-{uuid.uuid4().hex}.wav"
        config = model.with_name(model.name + ".json")
        return str(synthesize(model, config, text, output)), "Speech generated locally."
    except Exception as error:
        return None, f"Synthesis failed: {error}"


def compare_models(model_a: str, model_b: str, text: str):
    try:
        if not model_a or not model_b:
            raise ValueError("Select two exported models")
        output_a = DATA_DIR / "exports" / f"compare-a-{uuid.uuid4().hex}.wav"
        output_b = DATA_DIR / "exports" / f"compare-b-{uuid.uuid4().hex}.wav"
        synthesize(Path(model_a), Path(model_a).with_name(Path(model_a).name + ".json"), text, output_a)
        synthesize(Path(model_b), Path(model_b).with_name(Path(model_b).name + ".json"), text, output_b)
        return str(output_a), str(output_b), "Both models generated the same text locally."
    except Exception as error:
        return None, None, f"Model comparison failed: {error}"


def publish_selected(model_path: str):
    try:
        model = Path(model_path)
        config = model.with_name(model.name + ".json")
        paths = publish_model(model, config, DATA_DIR / "piper-voices")
        return f"Published {paths[0].name} and {paths[1].name} to the shared Piper voice directory."
    except Exception as error:
        return f"Publish failed: {error}"


def diagnostics():
    try:
        import torch
        cuda = f"available — {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "unavailable"
    except Exception:
        cuda = "unavailable"
    ffmpeg = "OK" if shutil.which("ffmpeg") else "missing"
    espeak = "OK" if shutil.which("espeak-ng") else "missing"
    piper_training = subprocess.run([os.sys.executable, "-m", "piper.train", "fit", "--help"], capture_output=True, text=True, timeout=20).returncode == 0
    api = "connected" if api_health() else "optional / disconnected"
    return f"Application: OK\nffmpeg: {ffmpeg}\neSpeak NG: {espeak}\nPiper training: {'OK' if piper_training else 'missing'}\nCUDA: {cuda}\nPiper API: {api}\nData directory: {DATA_DIR}"


def prepare_training_summary(project_id: str, dataset_name: str, mode: str, checkpoint_path: str, warmstart_path: str, device: str, batch_size: int, seed: int, max_epochs: int):
    try:
        dataset = _selected_dataset(project_id, dataset_name)
        manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8")) if (dataset / "dataset.json").exists() else {}
        split_sizes = {}
        for name in ("train", "validation", "test"):
            path = dataset / "splits" / f"{name}.json"
            split_sizes[name] = len(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else "not supplied"
        selected_device = device
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if selected_device == "auto":
            selected_device = "cuda" if has_cuda else "cpu"
        project = STORE.get_project(project_id)
        actual_batch = int(batch_size) if int(batch_size) > 0 else batch_size_for(selected_device)
        return (
            f"### Review this run before starting\n\n"
            f"Dataset: **{dataset.name}** · Samples: **{manifest.get('sample_count', 'imported')}** · "
            f"Training samples: **{manifest.get('training_sample_count', 'see metadata')}** · "
            f"Total audio: **{_clock(manifest.get('total_seconds', 0))}**\n\n"
            f"Fixed train / validation / test audio: **{_clock(manifest.get('split_durations_seconds', {}).get('train', 0))} / "
            f"{_clock(manifest.get('split_durations_seconds', {}).get('validation', 0))} / "
            f"{_clock(manifest.get('split_durations_seconds', {}).get('test', 0))}** · "
            f"Rows: **{split_sizes['train']} / {split_sizes['validation']} / {split_sizes['test']}**\n\n"
            f"Mode: **{'Fine-tune' if mode == 'finetune' else 'Full training from scratch'}** · "
            f"Checkpoint: **{Path(checkpoint_path).name if mode == 'finetune' and checkpoint_path else 'none'}** · "
            f"Vocoder warm-start: **{Path(warmstart_path).name if mode == 'scratch' and warmstart_path else 'none'}**\n\n"
                f"Sample rate: **22050 Hz** · eSpeak: **{project['espeak_voice']}** · Device: **{selected_device.upper()}** · Batch size: **{actual_batch}**\n\n"
            f"Maximum epochs: **{int(max_epochs)}** · Seed: **{int(seed)}** · Piper internal validation: **10% / 5 test examples**",
            "Review the summary. Click START TRAINING to launch the persisted job.",
        )
    except Exception as error:
        return "", f"Could not prepare training summary: {error}"


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    with gr.Blocks(title="Piper Voice Trainer", theme=theme, analytics_enabled=False) as demo:
        gr.Markdown("# Piper Voice Trainer\nRecord known prompts, prepare a reproducible dataset, and train a Piper voice locally. Recordings and models stay in the mounted data directory.")
        with gr.Row():
            project_select = gr.Dropdown(label="Current project", choices=_project_choices(), value=(_projects()[0]["id"] if _projects() else None), allow_custom_value=True, scale=3)
            project_status = gr.Markdown(_project_summary(project_select.value))
        with gr.Tabs():
            with gr.Tab("Project"):
                with gr.Row():
                    project_name = gr.Textbox(label="Project / voice name", value="Kamil PL")
                    language = gr.Textbox(label="Language", value="pl_PL")
                    espeak = gr.Textbox(label="eSpeak voice", value="pl")
                    create_button = gr.Button("Create project", variant="primary")
                create_status = gr.Markdown()
                gr.Markdown("#### Recording source\nPaste prose or upload a `.txt` file. Preview creates a saved queue but never starts recording.")
                with gr.Row():
                    prompt_upload = gr.File(label="Source text file (.txt)", file_types=[".txt"], type="filepath")
                    prompt_pack = gr.Dropdown(label="Built-in prompt pack", choices=[path.name for path in PROMPT_DIR.glob("*.txt")], value=None, allow_custom_value=True)
                    builtin_button = gr.Button("Use selected prompt pack")
                prompt_text = gr.Textbox(label="Paste source text", lines=6, placeholder="Paste Polish prose or one prompt per line")
                with gr.Row():
                    parse_mode = gr.Radio([("Prose / auto split", "prose"), ("One prompt per line", "lines")], value="prose", label="Parsing mode")
                    speaking_rate = gr.Slider(60, 240, value=140, step=5, label="Assumed speaking rate (words/minute)")
                    preview_button = gr.Button("Preview and save queue", variant="primary")
                estimate = gr.Markdown()
                queue_status = gr.Markdown()
                queue_search = gr.Textbox(label="Search prompts")
                queue_table = gr.Dataframe(headers=["Prompt ID", "Text", "Words", "Status"], datatype=["str", "str", "number", "str"], type="array", interactive=True, row_count=(0, "dynamic"), label="Editable prompt queue")
                with gr.Row():
                    save_queue_button = gr.Button("Save queue edits")
                    restore_queue_button = gr.Button("Restore queue from original source")
                with gr.Row():
                    prompt_index = gr.Number(value=0, precision=0, label="Prompt row (0-based)", scale=1)
                    move_up = gr.Button("Move up")
                    move_down = gr.Button("Move down")
                    split_left = gr.Textbox(label="Split: first part")
                    split_right = gr.Textbox(label="Split: second part")
                    split_button = gr.Button("Split row")
                    merge_button = gr.Button("Merge with next")
                    delete_prompt_button = gr.Button("Delete row", variant="stop")
                queue_action_status = gr.Markdown()
            with gr.Tab("Record"):
                gr.Markdown("Read the displayed prompt once. Accept stores a normalized WAV; a warning sends it to review. Re-recording the prompt marks the previous sample as superseded while keeping its files.")
                with gr.Row():
                    prompt_position = gr.Number(value=0, precision=0, label="Queue position", scale=1)
                    active_prompt_id = gr.State(value=None)
                    prev_button = gr.Button("Previous")
                    next_button = gr.Button("Next")
                    prompt_progress = gr.Markdown("0 / 0")
                current_text = gr.Markdown("Create or import a prompt queue first.")
                recording = gr.Audio(label="Record from microphone", sources=["microphone"], type="filepath")
                with gr.Row():
                    accept_button = gr.Button("Accept & next", variant="primary")
                    review_button = gr.Button("Save for review")
                    reject_button = gr.Button("Reject recording", variant="stop")
                    rerecord_button = gr.Button("Record again")
                record_result = gr.Markdown()
                sample_player = gr.Audio(label="Saved recording", interactive=False)
            with gr.Tab("Samples"):
                sample_filter = gr.Dropdown(["all", "accepted", "review", "rejected", "superseded"], value="all", label="Filter")
                sample_search = gr.Textbox(label="Search prompt text")
                sample_table = gr.Dataframe(headers=["Sample ID", "Prompt", "Status", "Seconds", "Peak dBFS", "Warnings", "Audio path"], datatype=["str", "str", "str", "number", "str", "str", "str"], type="array", interactive=False)
                with gr.Row():
                    sample_id = gr.Textbox(label="Sample ID to review")
                    load_sample_button = gr.Button("Play selected sample")
                    accept_sample_button = gr.Button("Mark accepted")
                    flag_sample_button = gr.Button("Mark for review")
                    reject_sample_button = gr.Button("Reject sample")
                sample_review_status = gr.Markdown()
                sample_audio_player = gr.Audio(label="Selected sample", interactive=False)
            with gr.Tab("Dataset"):
                gr.Markdown("Datasets use accepted normalized samples only. Duration targets are estimates based on actual recorded sample lengths; 30 and 60 minute subsets use the same deterministic seed.")
                with gr.Row():
                    duration_target = gr.Dropdown(["15 minutes", "30 minutes", "60 minutes", "All accepted", "Custom duration"], value="30 minutes", label="Dataset size")
                    custom_minutes_input = gr.Number(value=45, precision=0, minimum=1, maximum=10000, label="Custom target minutes", visible=False)
                    create_dataset_button = gr.Button("Create dataset", variant="primary")
                dataset_status = gr.Markdown()
                dataset_select = gr.Dropdown(label="Dataset", value=None, allow_custom_value=True)
                with gr.Row():
                    export_dataset_button = gr.Button("Export dataset ZIP")
                    dataset_archive = gr.File(label="Download portable dataset")
                    dataset_import_file = gr.File(label="Import dataset ZIP", file_types=[".zip"], type="filepath")
                    import_dataset_button = gr.Button("Import bundle")
                dataset_transfer_status = gr.Markdown()
            with gr.Tab("Train"):
                gr.Markdown("Fine-tuning is the recommended default. Scratch training is a separate workflow and does not use the base checkpoint. Vocoder warm-start initializes compatible vocoder weights without becoming a fine-tuning run.")
                with gr.Row():
                    train_dataset = gr.Dropdown(label="Dataset", value=None, allow_custom_value=True)
                    training_mode = gr.Radio([("Fine-tune existing Piper checkpoint", "finetune"), ("Full training from scratch", "scratch")], value="finetune", label="Training mode")
                base_select = gr.Dropdown(label="Base checkpoint selector", choices=[item["id"] for item in curated_checkpoints()] + ["Upload or use a local checkpoint"], value="pl_PL-darkman-medium")
                checkpoint_info = gr.Markdown("**pl_PL-darkman-medium** · Polish · one speaker · medium · 22,050 Hz · cache status shown by the path below · upstream source revision pinned. Curated presets are medium quality; uploaded/local files are your responsibility to validate.")
                with gr.Row():
                    base_checkpoint_path = gr.Textbox(label="Checkpoint path (downloaded, uploaded, or local)")
                    download_checkpoint_button = gr.Button("Download curated checkpoint")
                checkpoint_upload = gr.File(label="Upload .ckpt", file_types=[".ckpt"], type="filepath")
                checkpoint_upload_button = gr.Button("Copy uploaded checkpoint to persistent storage")
                scratch_warning = gr.Markdown("**Scratch training selected.** Small datasets may produce poor results compared with fine-tuning; training is still allowed.", visible=False)
                warmstart_checkbox = gr.Checkbox(label="Warm-start vocoder from a checkpoint", value=False, visible=False)
                warmstart_path = gr.Textbox(label="Vocoder warm-start checkpoint path", visible=False)
                with gr.Accordion("Advanced settings", open=False):
                    max_epochs = gr.Number(value=1000, precision=0, minimum=1, maximum=100000, label="Maximum epochs")
                    training_seed = gr.Number(value=42, precision=0, minimum=0, label="Random seed")
                with gr.Row():
                    device = gr.Radio([("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda")], value="auto", label="Device")
                    batch_size = gr.Number(value=0, precision=0, label="Batch size (0 = automatic)")
                    prepare_button = gr.Button("Prepare run summary")
                    cancel_button = gr.Button("Stop training", variant="stop")
                run_summary = gr.Markdown()
                train_button = gr.Button("START TRAINING", variant="primary")
                training_status = gr.Markdown()
                run_status = gr.Code(label="Persisted status and recent logs", language="shell")
                refresh_run_button = gr.Button("Refresh training status")
            with gr.Tab("Models"):
                run_select = gr.Dropdown(label="Training run", value=None, allow_custom_value=True)
                run_compare_select = gr.Dropdown(label="Second training run for comparison", value=None, allow_custom_value=True)
                inspect_run_button = gr.Button("Inspect saved run configuration")
                run_config_view = gr.Code(label="Saved run configuration and exact command", language="json")
                compare_runs_button = gr.Button("Compare run settings")
                run_comparison = gr.Markdown()
                with gr.Row():
                    model_name = gr.Textbox(label="Safe Piper voice filename", value="pl_PL-kamil-medium")
                    export_model_button = gr.Button("Export ONNX pair")
                    model_archive = gr.File(label="Download model ZIP")
                model_status = gr.Markdown()
                model_select = gr.Dropdown(label="Exported model", choices=[], value=None, allow_custom_value=True)
                fixed_test_prompt = gr.Dropdown(label="Fixed held-out test prompt", choices=[], value=None, allow_custom_value=True)
                load_fixed_prompt_button = gr.Button("Use selected held-out prompt")
                reference_audio = gr.Audio(label="Original held-out speaker recording", interactive=False)
                compare_model_select = gr.Dropdown(label="Second model for A/B comparison", choices=[], value=None, allow_custom_value=True)
                local_test_text = gr.Textbox(label="Text to synthesize locally", value="Dzisiaj sprawdzam własny model głosu.")
                with gr.Row():
                    synth_button = gr.Button("Generate local test speech")
                    compare_button = gr.Button("Compare both models")
                    publish_button = gr.Button("Publish to Piper API shared directory")
                synth_audio = gr.Audio(label="Generated speech", interactive=False)
                with gr.Row():
                    compare_audio_a = gr.Audio(label="Model A", interactive=False)
                    compare_audio_b = gr.Audio(label="Model B", interactive=False)
                publish_status = gr.Markdown()
            with gr.Tab("Test / Deploy"):
                gr.Markdown("Optional integration with the existing OpenAI-compatible Piper API at `http://piper-api:5000`. Compose publishes exported pairs from `data/piper-voices` into the server's `/data` volume.")
                api_status = gr.Markdown("Click diagnostics to check the optional Piper API service.")
                api_text = gr.Textbox(label="Text", value="To jest test mojego własnego modelu głosu.")
                api_voice = gr.Textbox(label="Voice name", value="pl_PL-kamil-medium")
                api_test_button = gr.Button("Test through OpenAI API")
                api_audio = gr.Audio(label="API-generated speech", interactive=False)
                diagnostics_button = gr.Button("Run diagnostics")
                diagnostics_text = gr.Code(label="Diagnostics", language="shell")

        create_button.click(create_project, [project_name, language, espeak], [project_select, project_status, create_status])
        project_select.change(select_project, project_select, [project_status, queue_table, dataset_select, sample_table, queue_action_status, train_dataset, prompt_position, current_text, prompt_progress, active_prompt_id])
        builtin_button.click(select_prompt_pack, prompt_pack, [prompt_upload, create_status])
        preview_button.click(preview_source, [project_select, prompt_upload, prompt_text, parse_mode, speaking_rate], [queue_table, estimate, queue_status, project_status]).then(lambda pid: current_prompt(pid, 0), project_select, [prompt_position, current_text, prompt_progress, active_prompt_id])
        queue_search.change(lambda pid, search: _queue_rows(pid, search) if pid else [], [project_select, queue_search], queue_table)
        save_queue_button.click(save_queue, [project_select, queue_table], [queue_table, project_status, queue_action_status]).then(resume_prompt, project_select, [prompt_position, current_text, prompt_progress, active_prompt_id])
        restore_queue_button.click(restore_source_queue, project_select, [queue_table, queue_action_status]).then(lambda pid: current_prompt(pid, 0), project_select, [prompt_position, current_text, prompt_progress, active_prompt_id])
        for button, action in ((move_up, "up"), (move_down, "down"), (split_button, "split"), (merge_button, "merge"), (delete_prompt_button, "delete")):
            button.click(lambda pid, index, left, right, act=action: prompt_action(pid, int(index), act, left, right), [project_select, prompt_index, split_left, split_right], [queue_table, project_status, queue_action_status]).then(resume_prompt, project_select, [prompt_position, current_text, prompt_progress, active_prompt_id])
        prev_button.click(lambda pid, idx: current_prompt(pid, max(0, int(idx)-1)), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        next_button.click(lambda pid, idx: current_prompt(pid, int(idx)+1), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        for button, decision in ((accept_button, "accept"), (review_button, "review"), (reject_button, "reject")):
            button.click(lambda pid, idx, prompt_id, path, choice=decision: record_sample(pid, int(idx), prompt_id, path, choice), [project_select, prompt_position, active_prompt_id, recording], [record_result, sample_player, project_status, sample_table, prompt_position, queue_table])
        rerecord_button.click(lambda: None, outputs=recording)
        prompt_position.change(lambda pid, idx: current_prompt(pid, int(idx)) if pid else (0, "", "0 / 0", None), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        sample_filter.change(lambda pid, filt, query: _sample_rows(pid, filt, query) if pid else [], [project_select, sample_filter, sample_search], sample_table)
        sample_search.change(lambda pid, filt, query: _sample_rows(pid, filt, query) if pid else [], [project_select, sample_filter, sample_search], sample_table)
        load_sample_button.click(load_sample_audio, [project_select, sample_id], [sample_audio_player, sample_review_status])
        for button, status in ((accept_sample_button, "accepted"), (flag_sample_button, "review"), (reject_sample_button, "rejected")):
            button.click(lambda pid, sid, value=status: review_sample(pid, sid, value), [project_select, sample_id], [sample_table, project_status, sample_review_status, queue_table])
        duration_target.change(lambda target: gr.update(visible=(target == "Custom duration")), duration_target, custom_minutes_input)
        create_dataset_button.click(make_dataset, [project_select, duration_target, custom_minutes_input], [dataset_select, train_dataset, dataset_status, project_status])
        export_dataset_button.click(export_dataset_ui, [project_select, dataset_select], [dataset_archive, dataset_transfer_status])
        import_dataset_button.click(import_dataset_ui, [project_select, dataset_import_file], [dataset_select, train_dataset, dataset_transfer_status])
        download_checkpoint_button.click(download_base_checkpoint, base_select, [base_checkpoint_path, training_status])
        checkpoint_upload_button.click(save_uploaded_checkpoint, checkpoint_upload, [base_checkpoint_path, training_status])
        training_mode.change(lambda mode: (gr.update(visible=(mode == "scratch")), gr.update(visible=(mode == "scratch")), gr.update(visible=(mode == "scratch"))), training_mode, [scratch_warning, warmstart_checkbox, warmstart_path])
        warmstart_checkbox.change(lambda enabled: gr.update(visible=enabled), warmstart_checkbox, warmstart_path)
        prepare_button.click(prepare_training_summary, [project_select, train_dataset, training_mode, base_checkpoint_path, warmstart_path, device, batch_size, training_seed, max_epochs], [run_summary, training_status])
        train_button.click(start_training, [project_select, train_dataset, training_mode, base_checkpoint_path, warmstart_path, device, batch_size, training_seed, max_epochs], [training_status, run_status])
        cancel_button.click(cancel_training, project_select, [training_status, run_status])
        refresh_run_button.click(lambda pid: (_run_status_text(pid), gr.update(choices=run_choices(pid))), project_select, [run_status, run_select])
        refresh_run_button.click(lambda pid: gr.update(choices=run_choices(pid)), project_select, run_compare_select)
        project_select.change(lambda pid: (gr.update(choices=run_choices(pid), value=None), gr.update(choices=run_choices(pid), value=None)), project_select, [run_select, run_compare_select])
        inspect_run_button.click(inspect_run_config, [project_select, run_select], run_config_view)
        compare_runs_button.click(compare_run_configs, [project_select, run_select, run_compare_select], run_comparison)
        export_model_button.click(export_run, [project_select, run_select, model_name], [model_archive, model_status, model_select])
        model_archive.change(lambda pid: gr.update(choices=model_choices(pid)), project_select, model_select)
        synth_button.click(synthesize_model, [model_select, local_test_text], [synth_audio, model_status])
        compare_button.click(compare_models, [model_select, compare_model_select, local_test_text], [compare_audio_a, compare_audio_b, model_status])
        publish_button.click(publish_selected, model_select, publish_status)
        project_select.change(lambda pid: gr.update(choices=model_choices(pid)), project_select, compare_model_select)
        dataset_select.change(lambda pid, dataset: gr.update(choices=evaluation_choices(pid, dataset)), [project_select, dataset_select], fixed_test_prompt)
        load_fixed_prompt_button.click(load_evaluation_prompt, [project_select, dataset_select, fixed_test_prompt], [local_test_text, reference_audio, model_status])
        diagnostics_button.click(diagnostics, outputs=diagnostics_text)
        api_test_button.click(lambda text, voice: (str(api_synthesize(text, voice, DATA_DIR / "exports" / f"api-{uuid.uuid4().hex}.wav")), "API request completed."), [api_text, api_voice], [api_audio, api_status])
        demo.load(load_app_state, project_select, [project_select, project_status, queue_table, dataset_select, sample_table, queue_action_status, train_dataset, prompt_position, current_text, prompt_progress, active_prompt_id])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), show_error=True, max_file_size="2gb")
