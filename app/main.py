"""Gradio UI for the local Piper voice workflow."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from app.audio import inspect_wav, normalize_audio, trim_edge_silence
from app.bundles import export_bundle, import_bundle
from app.checkpoints import download_checkpoint
from app.datasets import create_dataset
from app.export import export_onnx, package_model, publish_model, validate_voice_name
from app.inference import synthesize
from app.projects import ProjectStore
from app.text import estimate_text, parse_prompts, prompt_recommendation
from app.training import TrainingJobs, batch_size_for, validate_training_config
from app.workflow import WorkflowProgress
from app.ui import APP_CSS, step_heading


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
    if not STORE.has_project(project_id):
        return []
    root = STORE.project_dir(project_id) / "datasets"
    return sorted([path for path in root.iterdir() if path.is_dir() and (path / "metadata.csv").exists()], key=lambda path: path.name) if root.exists() else []


def _dataset_choices(project_id: str) -> list[tuple[str, str]]:
    choices = []
    for path in _dataset_dirs(project_id):
        manifest_path = path / "dataset.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            count = manifest.get("sample_count", "?")
            target = manifest.get("target_seconds")
            target_label = "All accepted" if target is None else f"{round(target / 60)} min target"
            if path.name.startswith("imported-"):
                target_label = f"Imported · {target_label}"
            saved = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %H:%M")
            label = f"{target_label} · {count} sample{'s' if count != 1 else ''} / {_duration_label(manifest.get('total_seconds', 0))} · {saved}"
        else:
            label = path.name
        choices.append((label, path.name))
    return choices


def _selected_dataset(project_id: str, dataset_name: str) -> Path:
    if not STORE.has_project(project_id):
        raise ValueError("Create or select a saved project first")
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("Select a dataset")
    path = STORE.project_dir(project_id) / "datasets" / dataset_name
    if not path.is_dir():
        raise ValueError("Selected dataset was not found")
    return path


def evaluation_choices(project_id: str, dataset_name: str):
    if not STORE.has_project(project_id) or not dataset_name:
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


def _sample_choices(project_id: str) -> list[tuple[str, str]]:
    if not STORE.has_project(project_id):
        return []
    choices = []
    for sample in STORE.list_samples(project_id):
        warning = " · check quality" if sample["quality"].get("warnings") else ""
        label = f"{sample['status'].capitalize()} · {sample['duration_seconds']:.1f}s · {sample['text'][:80]}{warning}"
        choices.append((label, sample["id"]))
    return choices


def _project_summary(project_id: str | None) -> str:
    if not STORE.has_project(project_id):
        return "**Your workspace is ready.** Start with Step 1: choose or create a project."
    project = STORE.get_project(project_id)
    counts = project["sample_counts"]
    accepted = counts.get("accepted", {}).get("seconds", 0)
    prompt_total = len(project["prompts"])
    samples = [sample for sample in STORE.list_samples(project_id) if sample["status"] != "superseded"]
    recorded = len({sample["prompt_id"] for sample in samples})
    accepted_count = counts.get("accepted", {}).get("count", 0)
    rejected_count = counts.get("rejected", {}).get("count", 0)
    review_count = counts.get("review", {}).get("count", 0)
    needs_attention = []
    if review_count:
        needs_attention.append(f"**{review_count}** to review")
    if rejected_count:
        needs_attention.append(f"**{rejected_count}** rejected")
    details = " · ".join(needs_attention)
    return (
        f"**Project: {project['name']}** · {project['language']}  \n"
        f"**{recorded}/{prompt_total}** prompts recorded · **{accepted_count}** accepted · "
        f"**{_duration_label(accepted)}** accepted audio"
        + (f" · {details}" if details else "")
    )


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02}:{(total % 3600) // 60:02}:{total % 60:02}"


def _duration_label(seconds: float) -> str:
    return f"{seconds:.0f} sec" if seconds < 60 else f"{seconds / 60:.1f} min"


def _queue_rows(project_id: str, search: str = "") -> list[list[Any]]:
    if not STORE.has_project(project_id):
        return []
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
        return (
            gr.update(choices=_project_choices(), value=project["id"]),
            _project_summary(project["id"]), "Project created.", gr.update(open=False),
        )
    except Exception as error:
        return gr.update(), "", f"Could not create project: {error}", gr.update(open=True)


def select_project(project_id: str | None):
    if not STORE.has_project(project_id):
        return (
            _project_summary(None), [], gr.update(choices=[], value=None), gr.update(choices=[], value=None),
            "", gr.update(choices=[], value=None), 0,
            "Create a project above to begin.", "0 / 0", None,
        )
    project = STORE.get_project(project_id)
    datasets = _dataset_choices(project_id)
    active_id = STORE.get_active_prompt(project_id)
    prompts = project["prompts"]
    active_index = next((index for index, item in enumerate(prompts) if item["id"] == active_id), 0)
    cursor = current_prompt(project_id, active_index)
    return (
        _project_summary(project_id), _queue_rows(project_id),
        gr.update(choices=datasets, value=datasets[0][1] if datasets else None),
        gr.update(choices=_sample_choices(project_id), value=None), "",
        gr.update(choices=datasets, value=datasets[0][1] if datasets else None), *cursor,
    )


def select_project_state(project_id: str | None):
    selected = project_id if STORE.has_project(project_id) else None
    run_options = run_choices(selected)
    model_options = model_choices(selected)
    return (
        *select_project(selected),
        gr.update(choices=run_options, value=run_options[0][1] if run_options else None),
        gr.update(choices=model_options, value=model_options[0][1] if model_options else None),
        gr.update(choices=model_options, value=None),
    )


def load_app_state(current_project_id: str | None):
    choices = _project_choices()
    available = {value for _, value in choices}
    selected = current_project_id if current_project_id in available else (choices[0][1] if choices else None)
    return (gr.update(choices=choices, value=selected), *select_project_state(selected))


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


def save_queue(project_id: str, rows: list[list[Any]] | None, search: str = ""):
    try:
        if search.strip():
            raise ValueError("Clear the prompt search before saving edits so the full queue is shown.")
        values = [{"id": str(row[0]), "text": str(row[1])} for row in (rows or []) if len(row) >= 2]
        STORE.save_prompt_queue(project_id, values)
        return _queue_rows(project_id), _project_summary(project_id), "Prompt queue saved."
    except Exception as error:
        return rows, _project_summary(project_id), f"Could not save prompt queue: {error}"


def current_prompt(project_id: str, index: int):
    if not STORE.has_project(project_id):
        return 0, "Choose or create a project in Step 1 first.", "0 / 0", None
    prompts = STORE.get_project(project_id)["prompts"]
    if not prompts:
        return 0, "Import or paste text to create a recording queue.", "0 / 0", None
    index = max(0, min(int(index), len(prompts) - 1))
    STORE.set_active_prompt(project_id, prompts[index]["id"])
    return index, prompts[index]["text"], f"Prompt {index + 1} of {len(prompts)}", prompts[index]["id"]


def resume_prompt(project_id: str):
    if not STORE.has_project(project_id):
        return current_prompt(project_id, 0)
    prompts = STORE.get_project(project_id)["prompts"]
    active_id = STORE.get_active_prompt(project_id)
    index = next((i for i, prompt in enumerate(prompts) if prompt["id"] == active_id), 0)
    return current_prompt(project_id, index)


def record_sample(project_id: str, index: int, prompt_id: str | None, recording_path: str | None, decision: str):
    if not project_id or not recording_path:
        return "Record audio with the microphone first.", None, _project_summary(project_id), gr.update(choices=_sample_choices(project_id)), index, _queue_rows(project_id), gr.update()
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
        state = f"{sample_status.upper()} · {_clock(quality['duration_seconds'])}\n" + ("\n".join(quality["warnings"]) if quality["warnings"] else "Quality checks look good.")
        return state, str(wav_target), _project_summary(project_id), gr.update(choices=_sample_choices(project_id), value=None), next_index, _queue_rows(project_id), None
    except Exception as error:
        return f"Could not save recording: {error}", None, _project_summary(project_id), gr.update(choices=_sample_choices(project_id)), index, _queue_rows(project_id), gr.update()


def load_sample_audio(project_id: str, sample_id: str):
    if not sample_id:
        return None, "Select a sample row to listen to its recording."
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
        return gr.update(choices=_sample_choices(project_id), value=sample_id), _project_summary(project_id), f"Sample marked {status}.", _queue_rows(project_id)
    except Exception as error:
        return gr.update(choices=_sample_choices(project_id), value=sample_id), _project_summary(project_id), str(error), _queue_rows(project_id)


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
        staging_root = DATA_DIR / "tmp"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dataset-audio-", dir=staging_root) as staging:
            prepared_samples = []
            for sample in sample_rows:
                if sample["status"] != "accepted" or not sample.get("audio_file"):
                    prepared_samples.append(sample)
                    continue
                source = Path(sample["audio_file"])
                trimmed = Path(staging) / f"{sample['id']}.wav"
                trim_edge_silence(source, trimmed)
                prepared_samples.append({
                    **sample,
                    "audio_file": str(trimmed),
                    "duration_seconds": inspect_wav(trimmed)["duration_seconds"],
                })
            manifest = create_dataset(prepared_samples, output, target_seconds, seed=42)
        project = STORE.get_project(project_id)
        info = {key: project[key] for key in ("name", "language", "espeak_voice")}
        info.update({"sample_rate": 22050, "dataset_id": dataset_id})
        (output / "project-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        choices = _dataset_choices(project_id)
        count = manifest["sample_count"]
        status = f"Created {count} sample{'s' if count != 1 else ''} ({_duration_label(manifest['total_seconds'])}), seed 42. Leading and trailing silence were trimmed in the training copies; original recordings are unchanged."
        return gr.update(choices=choices, value=dataset_id), gr.update(choices=choices, value=dataset_id), status, _project_summary(project_id)
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
        actual_run_id = JOBS.start(project_id, config, run_id=run_id)
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
    if not STORE.has_project(project_id):
        return "Choose or create a project in Step 1 first."
    status = JOBS.status(project_id)
    if not status:
        return "No training runs yet."
    return f"Run: {status['run_id']} · Status: {status['status']}\n{status.get('log_tail','')}"


def training_view(project_id: str | None):
    """Return a concise saved summary, optional loss history, and diagnostic logs."""
    if not STORE.has_project(project_id):
        return "Select a project to see training progress.", gr.update(value=None, visible=False), "No training runs yet."
    state = JOBS.status(project_id)
    if not state:
        return "No training runs yet.", gr.update(value=None, visible=False), "No training runs yet."
    from html import escape
    progress = state["progress"]
    maximum = progress["max_epochs"]
    completed = min(progress["completed_epochs"], maximum) if maximum else 0
    percentage = round(100 * completed / maximum) if maximum else 0
    current = progress["current_epoch"]
    phase = escape(state["status"].replace("_", " ").title())
    epoch_label = f"Epoch {current} of {maximum}" if current else "Waiting for first epoch"
    run_label = escape(state["run_id"][:8])
    summary = (
        f'<div class="train-summary"><div class="train-summary-head"><strong>{phase}</strong>'
        f'<span>Run {run_label}</span></div><p>{epoch_label} · {percentage}% of epochs completed</p>'
        f'<div class="train-progress" role="progressbar" aria-label="Completed training epochs" '
        f'aria-valuenow="{completed}" aria-valuemin="0" aria-valuemax="{maximum or 1}">'
        f'<span style="width:{percentage}%"></span></div></div>'
    )
    losses = progress["losses"]
    chart = gr.update(value=pd.DataFrame(losses) if losses else None, visible=bool(losses))
    logs = f"Run: {state['run_id']} · Status: {state['status']}\n{state.get('log_tail', '')}"
    return summary, chart, logs


def cancel_training(project_id: str):
    if not STORE.has_project(project_id):
        return "Choose or create a project in Step 1 first.", "No training runs yet."
    cancelled = JOBS.cancel(project_id)
    return "Training process stopped." if cancelled else "No active training process to stop.", _run_status_text(project_id)


def run_choices(project_id: str):
    if not STORE.has_project(project_id):
        return []
    root = STORE.project_dir(project_id) / "runs"
    choices = []
    for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True) if root.exists() else []:
        if not path.is_dir():
            continue
        config_path = path / "run-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        mode = "Fine-tune" if config.get("training_mode") == "finetune" else "Scratch" if config.get("training_mode") == "scratch" else "Run"
        saved = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %H:%M")
        choices.append((f"{mode} · {saved} · {path.name[:8]}", path.name))
    return choices


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
    if not STORE.has_project(project_id):
        return []
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


def workflow_progress(project_id: str | None) -> WorkflowProgress:
    if not STORE.has_project(project_id):
        return WorkflowProgress()
    project = STORE.get_project(project_id)
    run = JOBS.status(project_id)
    return WorkflowProgress(
        project=True, prompts=len(project["prompts"]),
        accepted=project["sample_counts"].get("accepted", {}).get("count", 0),
        datasets=len(_dataset_dirs(project_id)), runs=len(run_choices(project_id)),
        models=len(model_choices(project_id)),
        training=bool(run and run["status"] in {"preparing", "queued", "training"}),
    )


def step_availability(project_id: str | None):
    progress = workflow_progress(project_id)
    return tuple(gr.update(interactive=progress.blocked_reason(step) is None) for step in range(1, 7))


def navigate_step(project_id: str | None, current: int, target: int):
    reason = workflow_progress(project_id).blocked_reason(target)
    if reason:
        return gr.update(selected=current), f"**Before continuing:** {reason}"
    return gr.update(selected=target), ""


def reset_project_view(project_id: str | None):
    text, mode = "", "prose"
    if STORE.has_project(project_id):
        project = STORE.get_project(project_id)
        if project["source"]:
            source = STORE.project_dir(project_id) / "source" / "original.txt"
            text = source.read_text(encoding="utf-8") if source.is_file() else ""
            mode = project["source"]["mode"]
    return text, None, mode, None, _run_status_text(project_id)


def refresh_runs(project_id: str | None):
    choices = run_choices(project_id)
    summary, chart, logs = training_view(project_id)
    return (
        logs,
        gr.update(choices=choices, value=choices[0][1] if choices else None),
        summary,
        chart,
    )


def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate").set(
        block_label_background_fill="transparent", block_label_text_color="#334953",
        body_text_color="#243743", body_text_color_subdued="#52616f",
    )
    projects = _projects()
    initial_project = projects[0]["id"] if projects else None
    progress = workflow_progress(initial_project)
    forward_buttons = []
    back_buttons = []
    with gr.Blocks(title="Piper Voice Studio", theme=theme, css=APP_CSS, analytics_enabled=False, elem_id="voice-app") as demo:
        gr.HTML(
            "<div class='eyebrow'>PIPER · LOCAL VOICE TRAINING</div>"
            "<h1>Train your Piper voice</h1>"
            "<p>Record, prepare a dataset, train, and export a Piper voice. Your work stays on this machine.</p>",
            elem_id="studio-header",
        )
        project_status = gr.Markdown(_project_summary(initial_project), elem_id="project-context")
        workflow_status = gr.Markdown(elem_id="workflow-message")
        with gr.Tabs(selected=1, elem_id="workflow") as workflow_tabs:
            with gr.Tab("1 · Project", id=1, elem_classes="step-page") as project_tab:
                step_heading(1, "Choose or create a project", "A project keeps your prompts, recordings, datasets, and training runs together. Choose a saved voice or start a new one.")
                project_select = gr.Dropdown(label="Choose an existing project", choices=_project_choices(), value=initial_project, info="Select a project by name. No project ID is needed.")
                with gr.Accordion("Or create a new project", open=not projects) as project_create_panel:
                    project_name = gr.Textbox(label="Project name", placeholder="e.g. My Polish narrator", max_lines=1)
                    with gr.Accordion("Language settings · Polish by default", open=False):
                        with gr.Row():
                            language = gr.Textbox(label="Language code", value="pl_PL", info="pl_PL for Polish, en_US for American English.")
                            espeak = gr.Textbox(label="eSpeak voice", value="pl", info="Use the pronunciation voice for your language, e.g. pl or en-us.")
                    create_button = gr.Button("Create project", variant="primary")
                create_status = gr.Markdown()
                with gr.Row():
                    resume_button = gr.Button("Resume saved progress")
                    import_shortcut = gr.Button("Import a dataset ZIP")
                with gr.Row(elem_classes="step-footer"):
                    continue_project = gr.Button("Continue to Step 2 · Prepare text →", variant="primary")
                forward_buttons.append((continue_project, 1, 2))

            with gr.Tab("2 · Text", id=2, interactive=progress.blocked_reason(2) is None, elem_classes="step-page") as text_tab:
                step_heading(2, "Prepare your recording text", "Paste text or choose a file, then save the prompts you will read.")
                prompt_text = gr.Textbox(label="Paste your source text", lines=6, placeholder="Paste prose, or put one recording prompt on each line.")
                with gr.Accordion("Use a .txt file or a built-in prompt pack instead", open=False):
                    gr.Markdown("An uploaded file takes priority over pasted text. Clear the file to use your pasted text again.")
                    prompt_upload = gr.File(label="Source text file", file_types=[".txt"], type="filepath")
                    prompt_pack = gr.Dropdown(label="Built-in prompt pack", choices=sorted(path.name for path in PROMPT_DIR.glob("*.txt")), value=None)
                    builtin_button = gr.Button("Load prompt pack")
                    source_status = gr.Markdown()
                parse_mode = gr.Radio([("Prose — split into sentences", "prose"), ("One prompt per line", "lines")], value="prose", label="How should the text become prompts?")
                preview_button = gr.Button("Preview & save prompt queue", variant="primary")
                estimate = gr.Markdown()
                queue_status = gr.Markdown()
                gr.Markdown("### Check your prompts\nEdit the **Text** column if needed, then save your changes.")
                queue_table = gr.Dataframe(headers=["Prompt ID", "Text", "Words", "Status"], datatype=["str", "str", "number", "str"], type="array", interactive=True, static_columns=[0, 2, 3], row_count=(0, "dynamic"), col_count=(4, "fixed"), max_height=320, wrap=True, column_widths=[120, 460, 70, 140], label="Prompt queue", elem_id="prompt-queue")
                save_queue_button = gr.Button("Save text edits")
                queue_action_status = gr.Markdown()
                with gr.Row(elem_classes="step-footer"):
                    back_text = gr.Button("← Step 1 · Project")
                    continue_text = gr.Button("Continue to Step 3 · Record →", variant="primary")
                back_buttons.append((back_text, 1))
                forward_buttons.append((continue_text, 2, 3))

            with gr.Tab("3 · Record", id=3, interactive=progress.blocked_reason(3) is None, elem_classes="step-page") as record_tab:
                step_heading(3, "Record your voice", "Read one prompt at a time. Listen, then accept the take or record it again.")
                active_prompt_id = gr.State(value=None)
                prompt_progress = gr.Markdown("0 / 0")
                current_text = gr.Markdown("Prepare a prompt queue in Step 2 first.", elem_id="recording-prompt")
                gr.Markdown("Use a quiet room and keep the same microphone distance. Play your take before accepting it.")
                recording = gr.Audio(label="Your microphone recording", sources=["microphone"], type="filepath")
                with gr.Row():
                    accept_button = gr.Button("Accept & next prompt", variant="primary")
                    review_button = gr.Button("Save take for review")
                    reject_button = gr.Button("Reject take", variant="stop")
                record_result = gr.Markdown()
                with gr.Accordion("Listen to the last saved take", open=False):
                    sample_player = gr.Audio(label="Last saved recording", interactive=False)
                prompt_position = gr.State(value=0)
                with gr.Row():
                    prev_button = gr.Button("← Previous prompt")
                    next_button = gr.Button("Next prompt →")
                with gr.Accordion("Review saved recordings", open=False):
                    gr.Markdown("Only accepted takes enter a dataset. Choose a recording to listen or change its status.")
                    sample_select = gr.Dropdown(label="Saved recording", choices=_sample_choices(initial_project), value=None, filterable=True)
                    sample_audio_player = gr.Audio(label="Selected recording", interactive=False)
                    with gr.Row():
                        accept_sample_button = gr.Button("Mark accepted")
                        flag_sample_button = gr.Button("Mark for review")
                        reject_sample_button = gr.Button("Mark rejected", variant="stop")
                    sample_review_status = gr.Markdown()
                gr.Markdown("Continue when you have accepted recordings. You can return to record more later.")
                with gr.Row(elem_classes="step-footer"):
                    back_record = gr.Button("← Step 2 · Text")
                    continue_record = gr.Button("Continue to Step 4 · Build dataset →", variant="primary")
                back_buttons.append((back_record, 2))
                forward_buttons.append((continue_record, 3, 4))

            with gr.Tab("4 · Dataset", id=4, interactive=progress.blocked_reason(4) is None, elem_classes="step-page") as dataset_tab:
                step_heading(4, "Build your training dataset", "Turn accepted recordings into a reproducible dataset, or import a dataset ZIP from another machine. The dataset is what you will train on in the next step.")
                gr.Markdown("### Build from your recordings\nTargets use actual accepted audio duration. If you have less audio than the target, the dataset uses what is available.")
                duration_target = gr.Dropdown(["15 minutes", "30 minutes", "60 minutes", "All accepted", "Custom duration"], value="30 minutes", label="How much accepted audio should be included?")
                custom_minutes_input = gr.Number(value=45, precision=0, minimum=1, maximum=10000, label="Custom target (minutes)", visible=False)
                create_dataset_button = gr.Button("Create dataset from accepted takes", variant="primary")
                dataset_status = gr.Markdown()
                with gr.Accordion("Or import a dataset ZIP", open=False):
                    gr.Markdown("Use a ZIP exported by this app. Its files and hashes are validated before it becomes available for training.")
                    dataset_import_file = gr.File(label="Dataset ZIP to import", file_types=[".zip"], type="filepath")
                    import_dataset_button = gr.Button("Validate & import dataset")
                gr.Markdown("### Choose the dataset to use")
                dataset_select = gr.Dropdown(label="Saved dataset", value=None, info="Creating or importing a dataset selects it here and in Step 5.")
                dataset_transfer_status = gr.Markdown()
                with gr.Accordion("Download a dataset to train on another machine", open=False):
                    export_dataset_button = gr.Button("Export selected dataset as ZIP")
                    dataset_archive = gr.File(label="Download dataset ZIP", interactive=False)
                with gr.Row(elem_classes="step-footer"):
                    back_dataset = gr.Button("← Step 3 · Record")
                    continue_dataset = gr.Button("Continue to Step 5 · Train →", variant="primary")
                back_buttons.append((back_dataset, 3))
                forward_buttons.append((continue_dataset, 4, 5))

            with gr.Tab("5 · Train", id=5, interactive=progress.blocked_reason(5) is None, elem_classes="step-page") as train_tab:
                step_heading(5, "Train your Piper voice", "Choose a training mode and review the run settings before starting. Your configuration, checkpoints, and logs are saved with the project.")
                train_dataset = gr.Dropdown(label="Training dataset", value=None)
                training_mode = gr.Radio([("Fine-tune an existing voice (recommended)", "finetune"), ("Train a new voice from scratch", "scratch")], value="finetune", label="Training mode")
                with gr.Group() as finetune_panel:
                    gr.Markdown("### Starting checkpoint\nFine-tuning needs a compatible Piper **.ckpt** file. For Polish, download the suggested medium voice, or provide your own.")
                    download_checkpoint_button = gr.Button("Download Polish medium checkpoint")
                    base_checkpoint_path = gr.Textbox(label="Checkpoint path", placeholder="Download, upload, or enter a local .ckpt path", info="The selected checkpoint is saved with the training run.")
                    with gr.Accordion("Upload your own checkpoint", open=False):
                        checkpoint_upload = gr.File(label="Piper .ckpt file", file_types=[".ckpt"], type="filepath")
                        checkpoint_upload_button = gr.Button("Save uploaded checkpoint")
                scratch_warning = gr.Markdown("**Training from scratch:** No base voice checkpoint will be used. Small datasets may produce poorer results than fine-tuning.", visible=False)
                warmstart_checkbox = gr.Checkbox(label="Optionally warm-start the vocoder from a checkpoint", value=False, visible=False)
                warmstart_path = gr.Textbox(label="Vocoder warm-start checkpoint path", visible=False)
                with gr.Accordion("Device and training settings", open=False):
                    device = gr.Radio([("Automatic", "auto"), ("CPU", "cpu"), ("NVIDIA GPU / CUDA", "cuda")], value="auto", label="Training device", info="Automatic uses CUDA when available. CPU training can take much longer.")
                    batch_size = gr.Number(value=0, precision=0, minimum=0, label="Batch size (0 = automatic)")
                    max_epochs = gr.Number(value=1000, precision=0, minimum=1, maximum=100000, label="Maximum epochs")
                    training_seed = gr.Number(value=42, precision=0, minimum=0, label="Random seed")
                gr.Markdown("### Review and start")
                prepare_button = gr.Button("Review run summary")
                run_summary = gr.Markdown()
                train_button = gr.Button("Start training", variant="primary")
                training_status = gr.Markdown()
                with gr.Accordion("Training progress and controls", open=True) as training_progress_panel:
                    gr.Markdown("Training continues when you close this page. Return to this project to see its saved progress.")
                    run_progress = gr.HTML("No training runs yet.")
                    run_chart = gr.LinePlot(x="epoch", y="loss", color="series", y_aggregate="mean", title="Loss over epochs", x_title="Epoch", y_title="Loss", height=260, visible=False)
                    with gr.Accordion("Recent trainer logs", open=False):
                        run_status = gr.Code(label="Latest log lines", language="shell", value="No training runs yet.")
                    with gr.Row():
                        refresh_run_button = gr.Button("Refresh training status")
                        cancel_button = gr.Button("Stop training", variant="stop")
                with gr.Row(elem_classes="step-footer"):
                    back_train = gr.Button("← Step 4 · Dataset")
                    continue_train = gr.Button("Continue to Step 6 · Export & listen →", variant="primary")
                back_buttons.append((back_train, 4))
                forward_buttons.append((continue_train, 5, 6))

            with gr.Tab("6 · Voice", id=6, interactive=progress.blocked_reason(6) is None, elem_classes="step-page") as voice_tab:
                step_heading(6, "Export and listen to your voice", "Export a saved training checkpoint as a Piper voice, test it locally, and download the model to use in your own applications.")
                gr.Markdown("### 1. Export a trained voice\nSelect a run with a saved checkpoint. The download contains both the ONNX model and its JSON configuration.")
                run_select = gr.Dropdown(label="Training run", value=None)
                model_name = gr.Textbox(label="Voice filename", value="pl_PL-kamil-medium", info="Use letters, numbers, underscores, and hyphens.")
                export_model_button = gr.Button("Export voice & prepare download", variant="primary")
                model_status = gr.Markdown()
                model_archive = gr.File(label="Download voice ZIP", interactive=False)
                gr.Markdown("### 2. Listen to the exported voice")
                model_select = gr.Dropdown(label="Exported voice", choices=[], value=None)
                local_test_text = gr.Textbox(label="Text for your listening test", value="Dzisiaj sprawdzam własny model głosu.", lines=3)
                synth_button = gr.Button("Generate test speech", variant="primary")
                synth_audio = gr.Audio(label="Your generated voice", interactive=False)
                with gr.Accordion("Compare with a recording or another voice", open=False):
                    fixed_test_prompt = gr.Dropdown(label="Held-out test prompt from the Step 4 dataset", choices=[], value=None)
                    load_fixed_prompt_button = gr.Button("Load test text & original recording")
                    reference_audio = gr.Audio(label="Original speaker recording", interactive=False)
                    compare_model_select = gr.Dropdown(label="Second exported voice", choices=[], value=None)
                    compare_button = gr.Button("Generate the same text with both voices")
                    compare_audio_a = gr.Audio(label="First voice", interactive=False)
                    compare_audio_b = gr.Audio(label="Second voice", interactive=False)
                with gr.Accordion("Use your voice through the optional Piper API", open=False):
                    gr.Markdown("Publish the selected voice, then start the API with `docker compose --profile inference up -d`.")
                    publish_button = gr.Button("Publish selected voice to Piper API")
                    publish_status = gr.Markdown()
                gr.Markdown("**Keep improving your voice:** Return to Step 3 to record more, then build a new dataset and train another run. Your earlier runs stay saved.")
                with gr.Row(elem_classes="step-footer"):
                    back_voice = gr.Button("← Step 5 · Train")
                    record_more = gr.Button("Return to Step 3 · Record more")
                back_buttons.extend([(back_voice, 5), (record_more, 3)])
        step_tabs = [project_tab, text_tab, record_tab, dataset_tab, train_tab, voice_tab]

        create_button.click(create_project, [project_name, language, espeak], [project_select, project_status, create_status, project_create_panel])
        project_state_outputs = [
            project_status, queue_table, dataset_select, sample_select,
            queue_action_status, train_dataset, prompt_position, current_text,
            prompt_progress, active_prompt_id, run_select, model_select, compare_model_select,
        ]
        stale_text = [estimate, queue_status, source_status, record_result, sample_review_status, dataset_status, dataset_transfer_status, run_summary, training_status, model_status, publish_status, workflow_status]
        stale_files = [sample_player, sample_audio_player, dataset_archive, dataset_import_file, model_archive, synth_audio, reference_audio, compare_audio_a, compare_audio_b]
        project_select.change(select_project_state, project_select, project_state_outputs).then(
            reset_project_view, project_select, [prompt_text, prompt_upload, parse_mode, recording, run_status],
        ).then(
            lambda: ("",) * len(stale_text) + (None,) * len(stale_files), outputs=stale_text + stale_files,
        ).then(refresh_runs, project_select, [run_status, run_select, run_progress, run_chart]).then(step_availability, project_select, step_tabs).then(lambda: gr.update(selected=1), outputs=workflow_tabs)
        scroll_to_step = "() => { document.getElementById('workflow').scrollIntoView({behavior: 'smooth', block: 'start'}); }"
        for button, current, target in forward_buttons:
            button.click(lambda pid, source=current, destination=target: navigate_step(pid, source, destination), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        for button, target in back_buttons:
            source = 6 if button in (back_voice, record_more) else target + 1
            button.click(lambda pid, current=source, destination=target: navigate_step(pid, current, destination), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        resume_button.click(lambda pid: navigate_step(pid, 1, workflow_progress(pid).next_step), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        import_shortcut.click(lambda pid: navigate_step(pid, 1, 4), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        builtin_button.click(select_prompt_pack, prompt_pack, [prompt_upload, source_status])
        preview_button.click(lambda pid, upload, pasted, mode: preview_source(pid, upload, pasted, mode, 140), [project_select, prompt_upload, prompt_text, parse_mode], [queue_table, estimate, queue_status, project_status]).then(lambda pid: current_prompt(pid, 0), project_select, [prompt_position, current_text, prompt_progress, active_prompt_id]).then(step_availability, project_select, step_tabs)
        save_queue_button.click(save_queue, [project_select, queue_table], [queue_table, project_status, queue_action_status]).then(resume_prompt, project_select, [prompt_position, current_text, prompt_progress, active_prompt_id]).then(step_availability, project_select, step_tabs)
        prev_button.click(lambda pid, idx: current_prompt(pid, max(0, int(idx)-1)), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        next_button.click(lambda pid, idx: current_prompt(pid, int(idx)+1), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        for button, decision in ((accept_button, "accept"), (review_button, "review"), (reject_button, "reject")):
            button.click(lambda pid, idx, prompt_id, path, choice=decision: record_sample(pid, int(idx), prompt_id, path, choice), [project_select, prompt_position, active_prompt_id, recording], [record_result, sample_player, project_status, sample_select, prompt_position, queue_table, recording])
        prompt_position.change(lambda pid, idx: current_prompt(pid, int(idx)) if pid else (0, "", "0 / 0", None), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        sample_select.change(load_sample_audio, [project_select, sample_select], [sample_audio_player, sample_review_status])
        for button, status in ((accept_sample_button, "accepted"), (flag_sample_button, "review"), (reject_sample_button, "rejected")):
            button.click(lambda pid, sid, value=status: review_sample(pid, sid, value), [project_select, sample_select], [sample_select, project_status, sample_review_status, queue_table])
        duration_target.change(lambda target: gr.update(visible=(target == "Custom duration")), duration_target, custom_minutes_input)
        create_dataset_button.click(make_dataset, [project_select, duration_target, custom_minutes_input], [dataset_select, train_dataset, dataset_status, project_status]).then(step_availability, project_select, step_tabs)
        export_dataset_button.click(export_dataset_ui, [project_select, dataset_select], [dataset_archive, dataset_transfer_status])
        import_dataset_button.click(import_dataset_ui, [project_select, dataset_import_file], [dataset_select, train_dataset, dataset_transfer_status]).then(step_availability, project_select, step_tabs)
        download_checkpoint_button.click(lambda: download_base_checkpoint("pl_PL-darkman-medium"), outputs=[base_checkpoint_path, training_status])
        checkpoint_upload_button.click(save_uploaded_checkpoint, checkpoint_upload, [base_checkpoint_path, training_status])
        training_mode.change(lambda mode: (gr.update(visible=(mode == "finetune")), gr.update(visible=(mode == "scratch")), gr.update(visible=(mode == "scratch"), value=False), gr.update(visible=False, value="")), training_mode, [finetune_panel, scratch_warning, warmstart_checkbox, warmstart_path])
        warmstart_checkbox.change(lambda enabled: gr.update(visible=True) if enabled else gr.update(visible=False, value=""), warmstart_checkbox, warmstart_path)
        prepare_button.click(prepare_training_summary, [project_select, train_dataset, training_mode, base_checkpoint_path, warmstart_path, device, batch_size, training_seed, max_epochs], [run_summary, training_status])
        for setting in (train_dataset, training_mode, base_checkpoint_path, warmstart_path, device, batch_size, training_seed, max_epochs):
            setting.change(lambda: "", outputs=run_summary)
        train_button.click(start_training, [project_select, train_dataset, training_mode, base_checkpoint_path, warmstart_path, device, batch_size, training_seed, max_epochs], [training_status, run_status]).then(lambda: gr.update(open=True), outputs=training_progress_panel).then(refresh_runs, project_select, [run_status, run_select, run_progress, run_chart]).then(step_availability, project_select, step_tabs)
        cancel_button.click(cancel_training, project_select, [training_status, run_status]).then(refresh_runs, project_select, [run_status, run_select, run_progress, run_chart])
        refresh_run_button.click(refresh_runs, project_select, [run_status, run_select, run_progress, run_chart]).then(step_availability, project_select, step_tabs)
        gr.Timer(10).tick(refresh_runs, project_select, [run_status, run_select, run_progress, run_chart])
        export_model_button.click(export_run, [project_select, run_select, model_name], [model_archive, model_status, model_select])
        model_select.change(lambda pid: gr.update(choices=model_choices(pid)), project_select, compare_model_select)
        synth_button.click(synthesize_model, [model_select, local_test_text], [synth_audio, model_status])
        compare_button.click(compare_models, [model_select, compare_model_select, local_test_text], [compare_audio_a, compare_audio_b, model_status])
        publish_button.click(publish_selected, model_select, publish_status)
        dataset_select.change(lambda pid, dataset: (gr.update(choices=evaluation_choices(pid, dataset), value=None), gr.update(value=dataset)), [project_select, dataset_select], [fixed_test_prompt, train_dataset])
        train_dataset.input(lambda dataset: gr.update(value=dataset), train_dataset, dataset_select)
        load_fixed_prompt_button.click(load_evaluation_prompt, [project_select, dataset_select, fixed_test_prompt], [local_test_text, reference_audio, model_status])
        demo.load(load_app_state, project_select, [project_select, *project_state_outputs]).then(reset_project_view, project_select, [prompt_text, prompt_upload, parse_mode, recording, run_status]).then(refresh_runs, project_select, [run_status, run_select, run_progress, run_chart]).then(step_availability, project_select, step_tabs)
    return demo


def launch_app():
    return build_app().launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        show_error=True, max_file_size="2gb",
        allowed_paths=[str((DATA_DIR / "projects").resolve()), str((DATA_DIR / "exports").resolve())],
    )


if __name__ == "__main__":
    launch_app()
