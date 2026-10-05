"""Gradio UI for the local Piper voice workflow."""

from __future__ import annotations

import json
from importlib import metadata
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from app.audio import inspect_wav, normalize_audio, trim_edge_silence
from app.bundles import export_bundle, import_bundle
from app.checkpoints import download_checkpoint, inference_config
from app.datasets import create_dataset, saved_split_indices
from app.export import export_onnx, package_model, package_training_run, publish_model, snapshot_checkpoint, validate_voice_name
from app.inference import synthesis_parameters, synthesize
from app.projects import ProjectStore
from app.recording_review import review_rows
from app.text import estimate_text, parse_prompts, prompt_recommendation
from app.training import TrainingJobs, batch_size_for, validate_training_config
from app.performance import cpu_snapshot, effective_cpu_count, gpu_snapshot, physical_cpu_count, workers_for
from app.workflow import WorkflowProgress
from app.ui import training_summary


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
    return [(row["text"][:110], row["sample_id"]) for row in rows if row.get("split") == "test"]


def load_evaluation_prompt(project_id: str, dataset_name: str, sample_id: str):
    try:
        dataset = _selected_dataset(project_id, dataset_name)
        rows = json.loads((dataset / "sample-index.json").read_text(encoding="utf-8"))
        prompt = next(row["text"] for row in rows if row["sample_id"] == sample_id and row["split"] == "test")
        audio = dataset / "audio" / next(row["filename"] for row in rows if row["sample_id"] == sample_id)
        return prompt, str(audio), "Loaded fixed held-out test prompt and reference recording. This sample is excluded from the training split."
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
    imported = imported_audio_metrics(project_id)
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
        + (f"**{accepted_count + imported['count']}** available recordings · "
           f"**{_duration_label(accepted + imported['seconds'])}** available audio  \n" if imported["count"] else "")
        + f"Local recordings: **{recorded}/{prompt_total}** prompts recorded · **{accepted_count}** accepted · "
        f"**{_duration_label(accepted)}** accepted audio"
        + (f" · {details}" if details else "")
        + (f"  \n**{imported['count']}** imported recording{'s' if imported['count'] != 1 else ''} · "
           f"**{_duration_label(imported['seconds'])}** imported audio" if imported["count"] else "")
    )


def imported_audio_metrics(project_id: str) -> dict[str, float]:
    """Count each imported sample once, excluding recordings already in this project."""
    # Registered clips (including rejected/replaced takes) already have review state.
    seen = {value for sample in STORE.list_samples(project_id)
            for value in (sample["id"], sample["quality"].get("source_sample_id")) if value}
    count, seconds = 0, 0.0
    for dataset in _dataset_dirs(project_id):
        if not dataset.name.startswith("imported-"):
            continue
        index = json.loads((dataset / "sample-index.json").read_text(encoding="utf-8"))
        for row in index:
            if row["sample_id"] in seen:
                continue
            audio_path = (dataset / "audio" / row["filename"]).resolve()
            if not audio_path.is_relative_to((dataset / "audio").resolve()):
                raise ValueError("Imported recording path escapes its dataset")
            with wave.open(str(audio_path), "rb") as audio:
                seconds += audio.getnframes() / audio.getframerate()
            count += 1
            seen.add(row["sample_id"])
    return {"count": count, "seconds": seconds}


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
        *refresh_voice_comparison(selected),
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


def recording_review_list(project_id, dataset_name=None, filter_name="Needs review", search="", selected=None, advance=False):
    try:
        choices = _dataset_choices(project_id)
        names = [value for _, value in choices]
        dataset_name = dataset_name if dataset_name in names else (names[-1] if names else None)
        dataset = _selected_dataset(project_id, dataset_name) if dataset_name else None
        if dataset and (dataset / "sample-index.json").is_file():
            STORE.register_dataset_recordings(project_id, dataset)
        rows = review_rows(STORE, project_id, dataset)
        matches = [row for row in rows if (filter_name == "All recordings" or row["needs_review"])
                   and search.casefold().strip() in row["text"].casefold()]
        ids = [row["id"] for row in matches]
        if selected not in ids:
            selected = ids[0] if ids else None
        elif advance:
            selected = ids[(ids.index(selected) + 1) % len(ids)]
        options = [(f"{row['status'].capitalize()} · {row['duration_seconds']:.1f}s · {row['text'][:80]}", row["id"]) for row in matches]
        flagged = sum(row["needs_review"] for row in rows)
        summary = f"**{flagged} need review** · {len(rows)} recordings. Flags are checks to make, not proof of bad audio."
        if not matches:
            summary += " No matching recordings; choose All recordings to browse."
        return gr.update(choices=options, value=selected), summary, gr.update(choices=choices, value=dataset_name)
    except Exception as error:
        return gr.update(choices=[], value=None), f"Could not load review: {error}", gr.update()


def recording_review_details(project_id, sample_id, dataset_name=None):
    try:
        dataset = _selected_dataset(project_id, dataset_name) if dataset_name else None
        row = next((item for item in review_rows(STORE, project_id, dataset) if item["id"] == sample_id), None)
        if row is None:
            return None, None, "Choose a recording.", "", *(gr.update(interactive=False) for _ in range(4))
        reason = " · ".join(row["reasons"]) or "No automatic warnings. Listen for missing words, unnatural delivery and cut beginnings or endings."
        if row["quality"].get("imported"):
            details = "Imported dataset audio; the original untrimmed recording is not in this bundle."
        elif row["removed_seconds"] is not None:
            details = f"Dataset preparation removed {row['removed_seconds']:.2f}s from the edges. This does not measure Piper's later training-time trimming."
        else:
            details = "No matching dataset copy selected."
        return row["original"], row["dataset_audio"], f"**Sentence:** {row['text']}\n\n**{row['status'].capitalize()}** · {reason}", details, *(gr.update(interactive=True) for _ in range(4))
    except Exception as error:
        return None, None, f"Could not load recording: {error}", "", *(gr.update(interactive=False) for _ in range(4))


def rerecord_sample(project_id, sample_id):
    try:
        sample = next(item for item in STORE.list_samples(project_id) if item["id"] == sample_id)
        prompts = STORE.get_project(project_id)["prompts"]
        position = next(i for i, prompt in enumerate(prompts) if prompt["id"] == sample["prompt_id"])
        STORE.set_sample_status(project_id, sample_id, "review")
        return (*current_prompt(project_id, position), "Record new", None, "Record a replacement below. The previous take is excluded from new datasets until kept or replaced.")
    except Exception as error:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Could not prepare re-recording: {error}"


def make_dataset(project_id: str, target: str, custom_minutes: int = 30):
    try:
        for dataset in _dataset_dirs(project_id):
            if (dataset / "sample-index.json").is_file():
                STORE.register_dataset_recordings(project_id, dataset)
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
                if sample["quality"].get("imported"):
                    prepared_samples.append(sample)
                    continue
                trimmed = Path(staging) / f"{sample['id']}.wav"
                trim_edge_silence(source, trimmed)
                prepared_samples.append({
                    **sample,
                    "audio_file": str(trimmed),
                    "duration_seconds": inspect_wav(trimmed)["duration_seconds"],
                })
            fixed_splits = {sample["id"]: sample["quality"]["source_split"] for sample in prepared_samples if sample["quality"].get("source_split")}
            manifest = create_dataset(prepared_samples, output, target_seconds, seed=42, split_by_id=fixed_splits)
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
        with tempfile.TemporaryDirectory(prefix=".import-", dir=target.parent) as temporary:
            staged = import_bundle(archive_path, Path(temporary) / "dataset")
            staged.rename(target)
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


def _setting(value: Any, custom: Any, allowed: set[int], name: str) -> tuple[str, int | None]:
    if value in ("Auto", 0, "0", None):
        return "auto", None
    selected = custom if value == "Custom" else value
    try:
        number = int(selected)
        if float(selected) != number:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a whole number") from None
    if number < 1 or (allowed and value != "Custom" and number not in allowed):
        raise ValueError(f"{name} must be a positive whole number")
    if number > 256:
        raise ValueError(f"{name} cannot exceed 256")
    return "manual", number


def _training_options(device: str, batch: Any, custom_batch: Any, workers: Any, custom_workers: Any,
                      threads: Any, hourly_rate: Any) -> dict[str, Any]:
    batch_mode, batch_value = _setting(batch, custom_batch, {4, 8, 16, 32, 64}, "Batch size")
    worker_mode, worker_value = _setting(workers, custom_workers, {1, 2, 4, 8}, "DataLoader workers")
    thread_mode, thread_value = _setting(threads, threads, set(), "PyTorch CPU threads")
    try:
        rate = None if hourly_rate in (None, "") else float(hourly_rate)
    except (TypeError, ValueError):
        raise ValueError("GPU cost per hour must be a number") from None
    if rate is not None and (not math.isfinite(rate) or rate < 0):
        raise ValueError("GPU cost per hour must be a finite non-negative number")
    return {"batch_size_mode": batch_mode, "batch_size": batch_value or batch_size_for(device),
            "num_workers_mode": worker_mode, "num_workers": worker_value or workers_for(device, physical_cpu_count(), effective_cpu_count()),
            "torch_threads_mode": thread_mode, "torch_threads": thread_value,
            "gpu_hourly_rate": rate}


def start_training(project_id: str, dataset_name: str, mode: str, checkpoint_path: str, warmstart_path: str, device: str, batch_size: Any, seed: int, max_epochs: int,
                   custom_batch: Any = None, workers: Any = "Auto", custom_workers: Any = None, threads: Any = "Auto", hourly_rate: Any = None,
                   checkpoint_interval: Any = 250, learning_rate: Any = 0.0002, learning_rate_d: Any = 0.0001):
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
            "csv_path": str(dataset / "metadata.csv"), "audio_dir": str(dataset / "audio"),
            "cache_dir": str(run_dir / "cache"), "config_path": str(run_dir / "voice.onnx.json"),
            "run_dir": str(run_dir), "training_mode": mode,
            "checkpoint": (checkpoint_path or None) if mode == "finetune" else None,
            "checkpoint_id": ("pl_PL-darkman-medium" if checkpoint_path and "pl_PL-darkman-medium" in checkpoint_path else "local-or-uploaded") if mode == "finetune" and checkpoint_path else None,
            "vocoder_warmstart_checkpoint": warmstart_path or None, "device": selected_device,
            **_training_options(selected_device, batch_size, custom_batch, workers, custom_workers, threads, hourly_rate),
            "seed": int(seed), "max_epochs": int(max_epochs),
            "checkpoint_interval": checkpoint_interval, "learning_rate": learning_rate, "learning_rate_d": learning_rate_d,
            "split_mode": "saved", "validation_split": 0.0, "num_test_examples": 0,
            "piper_revision": os.environ.get("PIPER_REVISION", "unknown"),
            "piper_version": piper_version(),
            "base_checkpoint_hash": _hash_if_exists(Path(checkpoint_path)) if mode == "finetune" and checkpoint_path else None,
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


def piper_version() -> str:
    try:
        return metadata.version("piper-tts")
    except metadata.PackageNotFoundError:
        return "unavailable"


def hardware_diagnostics() -> dict[str, Any]:
    try:
        import torch
        torch_version, cuda_runtime, has_cuda = torch.__version__, torch.version.cuda, torch.cuda.is_available()
        default_threads = torch.get_num_threads()
    except ImportError:
        torch_version, cuda_runtime, has_cuda, default_threads = "unavailable", None, False, None
    cpu, _ = cpu_snapshot(os.getpid())
    try:
        model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        model = "unknown"
    return {"PyTorch version": torch_version, "CUDA runtime": cuda_runtime, "CUDA available": has_cuda,
            "GPU": gpu_snapshot(), "CPU model": model, "Physical CPU cores": physical_cpu_count(),
            "Logical CPU cores available": effective_cpu_count(),
            "RAM used / total bytes": [cpu.get("ram_used_bytes"), cpu.get("ram_total_bytes")],
            "Piper revision": os.environ.get("PIPER_REVISION", "unknown"),
            "Piper version": piper_version(),
            "Auto GPU DataLoader workers": workers_for("cuda", physical_cpu_count(), effective_cpu_count()),
            "Auto CPU DataLoader workers": workers_for("cpu", physical_cpu_count(), effective_cpu_count()),
            "PyTorch default threads": default_threads}


def training_view(project_id: str | None):
    """Return a concise saved summary, optional loss history, and diagnostic logs."""
    if not STORE.has_project(project_id):
        return "Select a project to see training progress.", gr.update(value=None, visible=False), "No training runs yet."
    state = JOBS.status(project_id)
    if not state:
        return "No training runs yet.", gr.update(value=None, visible=False), "No training runs yet."
    summary = training_summary(state)
    losses = state["progress"]["losses"]
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
        if not checkpoint_choices(project_id, path.name):
            continue
        config_path = path / "run-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        mode = "Fine-tune" if config.get("training_mode") == "finetune" else "Scratch" if config.get("training_mode") == "scratch" else "Run"
        created = config.get("created_at")
        saved = datetime.fromisoformat(created).strftime("%d %b %H:%M") if created else datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b %H:%M")
        status_path = path / "status.json"
        status = json.loads(status_path.read_text()).get("status", "saved") if status_path.is_file() else "saved"
        dataset = config.get("dataset_id", "dataset")
        manifest_path = STORE.project_dir(project_id) / "datasets" / Path(dataset).name / "dataset.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            dataset = f"{manifest.get('sample_count', '?')} samples · {_duration_label(manifest.get('total_seconds', 0))}"
        choices.append((f"{config.get('voice_name', 'Voice')} · {mode} · {dataset} · up to {config.get('max_epochs', '?')} epochs · {status} · {saved} · {path.name[:8]}", path.name))
    return choices


def checkpoint_choices(project_id: str, run_id: str | None) -> list[tuple[str, str]]:
    if not STORE.has_project(project_id) or not run_id:
        return []
    run_dir = STORE.project_dir(project_id) / "runs" / Path(run_id).name
    checkpoints = []
    for directory in (run_dir / "checkpoints", run_dir / "metrics/version_0/checkpoints"):
        if directory.is_dir():
            checkpoints.extend(path for path in directory.rglob("*.ckpt") if path.is_file() and path.stat().st_size)
    choices = []
    for path in sorted(checkpoints, key=lambda item: item.stat().st_mtime, reverse=True):
        epoch = re.search(r"^(?:best-)?epoch[-=](\d+)", path.stem)
        saved = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%d %b %H:%M UTC")
        label = f"{'Best validation · ' if path.stem.startswith('best-') else ''}After epoch {int(epoch[1]) + 1}" if epoch else "Final trained checkpoint" if path.stem == "final" else f"Latest saved training checkpoint · {saved}" if path.stem == "last" else path.stem
        choices.append((f"{label} · {path.stat().st_size / 1024**2:.0f} MiB", str(path)))
    return choices


def refresh_checkpoints(project_id: str, run_id: str | None, checkpoint_path: str | None = None):
    choices = checkpoint_choices(project_id, run_id)
    selected = checkpoint_path if checkpoint_path in {value for _, value in choices} else choices[0][1] if choices else None
    return gr.update(choices=choices, value=selected, interactive=bool(choices)), gr.update(interactive=bool(choices))


def _selected_checkpoint(project_id: str, run_id: str, checkpoint_path: str | None = None) -> Path:
    checkpoints = [Path(value) for _, value in checkpoint_choices(project_id, run_id)]
    if not checkpoints:
        raise ValueError("No training checkpoint is available in this run yet")
    chosen = Path(checkpoint_path).resolve() if checkpoint_path else checkpoints[0].resolve()
    if chosen not in {path.resolve() for path in checkpoints}:
        raise ValueError("Selected checkpoint does not belong to this run")
    return chosen


def download_run_checkpoint(project_id: str, run_id: str, checkpoint_path: str | None):
    try:
        chosen = _selected_checkpoint(project_id, run_id, checkpoint_path)
        target = DATA_DIR / "exports" / f"{Path(run_id).name}-{chosen.stem}-{uuid.uuid4().hex[:8]}.ckpt"
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_checkpoint(chosen, target)
        return str(target), "Saved a checkpoint copy for further training. ONNX voices cannot be used as training checkpoints."
    except Exception as error:
        return None, f"Checkpoint download failed: {error}"


def download_training_run(project_id: str, run_id: str):
    try:
        if not STORE.has_project(project_id) or not run_id:
            raise ValueError("Select a training run")
        runs = (STORE.project_dir(project_id) / "runs").resolve()
        run = (runs / run_id).resolve()
        if run.parent != runs or not run.is_dir():
            raise ValueError("Select an existing training run")
        archive = DATA_DIR / "exports" / f"{run.name}-{uuid.uuid4().hex[:8]}.piper-training.zip"
        return str(package_training_run(run, archive)), "Training ZIP ready: checkpoints, metrics, logs and configuration. Keep the dataset ZIP to resume training elsewhere."
    except Exception as error:
        return None, f"Training ZIP failed: {error}"


def export_run(project_id: str, run_id: str, voice_name: str, checkpoint_path: str | None = None):
    try:
        if not run_id:
            raise ValueError("Select a trained run with a saved checkpoint")
        run_dir = STORE.project_dir(project_id) / "runs" / Path(run_id).name
        if not run_dir.is_dir():
            raise ValueError("Select a training run")
        chosen = _selected_checkpoint(project_id, run_id, checkpoint_path)
        config = run_dir / "voice.onnx.json"
        return _export_checkpoint(project_id, chosen, config, voice_name, run_id)
    except Exception as error:
        return None, f"ONNX export failed: {error}", gr.update()


def _export_checkpoint(project_id: str, chosen: Path, config: Path, voice_name: str, source_id: str):
    validate_voice_name(voice_name)
    export_name = voice_name[:40] + f"-{source_id[:8]}"
    epoch = re.search(r"^(?:best-)?epoch[-=](\d+)", chosen.stem)
    if epoch:
        export_name += f"-epoch-{int(epoch[1]) + 1}"
    else:
        export_name += f"-{re.sub(r'[^A-Za-z0-9_-]', '-', chosen.stem)[:20]}"
    export_name += f"-{uuid.uuid4().hex[:8]}"
    export_name = validate_voice_name(export_name)
    output = STORE.project_dir(project_id) / "models" / export_name
    # Export a snapshot so the rolling checkpoint can advance during training.
    with tempfile.TemporaryDirectory(prefix="piper-checkpoint-") as temporary:
        snapshot = Path(temporary) / chosen.name
        snapshot_checkpoint(chosen, snapshot)
        model, config_file = export_onnx(snapshot, config, output, export_name)
    archive = DATA_DIR / "exports" / f"{export_name}.piper-model.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    package_model(model, config_file, archive)
    choices = model_choices(project_id)
    return str(archive), f"Exported model pair: {model.name} ({model.stat().st_size:,} bytes) and {config_file.name} ({config_file.stat().st_size:,} bytes).", gr.update(choices=choices, value=str(model))


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


def synthesize_model(model_path: str, text: str, noise_scale=None, noise_w=None, length_scale=None):
    try:
        if not model_path:
            raise ValueError("Select a saved voice or generate a checkpoint sample first")
        model = Path(model_path)
        (DATA_DIR / "exports").mkdir(parents=True, exist_ok=True)
        output = DATA_DIR / "exports" / f"test-{uuid.uuid4().hex}.wav"
        config = model.with_name(model.name + ".json")
        parameters = synthesis_parameters(noise_scale, noise_w, length_scale)
        audio = synthesize(model, config, text, output, **parameters)
        settings = ", ".join(f"{name}={value:g}" for name, value in parameters.items()) or "model defaults"
        return str(audio), f"Speech generated locally using {settings}."
    except Exception as error:
        return None, f"Synthesis failed: {error}"


def export_checkpoint_file(project_id: str, voice_name: str, checkpoint_path: str, config_path: str | None):
    try:
        if not STORE.has_project(project_id):
            raise ValueError("Choose a project first")
        chosen = Path(checkpoint_path or "").resolve()
        if not chosen.is_relative_to(DATA_DIR.resolve()):
            raise ValueError("Checkpoint must be inside the data directory")
        if config_path and not Path(config_path).resolve().is_relative_to(DATA_DIR.resolve()):
            raise ValueError("Piper config must be inside the data directory")
        config = inference_config(chosen, config_path, DATA_DIR / "checkpoints")
        if not config.resolve().is_relative_to(DATA_DIR.resolve()):
            raise ValueError("Piper config must be inside the data directory")
        return _export_checkpoint(project_id, chosen, config, voice_name, "file")
    except Exception as error:
        return None, f"ONNX export failed: {error}", gr.update()


def generate_checkpoint_file_sample(project_id: str, voice_name: str, checkpoint_path: str, config_path: str,
                                    text: str, noise_scale=None, noise_w=None, length_scale=None):
    return generate_checkpoint_sample(project_id, None, voice_name, checkpoint_path, text,
                                      noise_scale, noise_w, length_scale, from_file=True, config_path=config_path)


def generate_checkpoint_sample(project_id: str, run_id: str, voice_name: str, checkpoint_path: str | None, text: str,
                               noise_scale=None, noise_w=None, length_scale=None, *, from_file=False, config_path=None):
    """One user action exports exactly the selected checkpoint and speaks the test text."""
    if not text or not text.strip():
        return None, "Enter text for your listening test.", None, gr.update(value=None)
    try:
        parameters = synthesis_parameters(noise_scale, noise_w, length_scale)
    except ValueError as error:
        return None, f"Speech not generated: {error}", None, gr.update(value=None)
    label = next((label for label, path in checkpoint_choices(project_id, run_id) if path == checkpoint_path), "Latest checkpoint")
    archive, message, model_update = (export_checkpoint_file(project_id, voice_name, checkpoint_path, config_path)
                                    if from_file else export_run(project_id, run_id, voice_name, checkpoint_path))
    if not archive:
        return None, message, None, gr.update(value=None)
    audio, speech_message = synthesize_model(model_update["value"], text, **parameters)
    source = f"Checkpoint file **{Path(checkpoint_path).parent.name} · {Path(checkpoint_path).name}**" if from_file else f"Run **{run_id[:8]}** · **{label}**"
    status = f"{source}. {speech_message}"
    return audio, status, archive, model_update


def run_dataset(project_id: str, run_id: str | None) -> str | None:
    if not STORE.has_project(project_id) or not run_id:
        return None
    path = STORE.project_dir(project_id) / "runs" / Path(run_id).name / "run-config.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    dataset = config.get("dataset_id")
    return Path(dataset).name if dataset else None


def run_evaluation_choices(project_id: str, run_id: str | None):
    return evaluation_choices(project_id, run_dataset(project_id, run_id))


def refresh_run_reference(project_id: str, run_id: str | None, sample_id: str | None = None):
    choices = run_evaluation_choices(project_id, run_id)
    selected = sample_id if sample_id in {value for _, value in choices} else None
    return gr.update(choices=choices, value=selected, interactive=bool(choices))


def reference_help(project_id: str, run_id: str | None):
    if not run_id:
        return "Select a run to check for reference recordings."
    if not run_evaluation_choices(project_id, run_id):
        return "No held-out reference recording is available for this run. You can still test the checkpoint with your own text."
    return "Choose a reference sentence and load it to hear the same words from the original speaker and checkpoint."


def load_run_evaluation_prompt(project_id: str, run_id: str, sample_id: str):
    return load_evaluation_prompt(project_id, run_dataset(project_id, run_id), sample_id)


def _comparison_sources(project_id: str):
    sources = {path: (f"Saved model · {label}", None) for label, path in model_choices(project_id)}
    for run_label, run_id in run_choices(project_id):
        for label, path in checkpoint_choices(project_id, run_id):
            sources[path] = (f"Checkpoint · {label} · {run_label.split(' · ')[0]} · run {run_id[:8]}", run_id)
    return sources


def refresh_voice_comparison(project_id: str, selected_a: str | None = None, selected_b: str | None = None):
    sources = _comparison_sources(project_id)
    choices = [(label, path) for path, (label, _) in sources.items()]
    return tuple(gr.update(choices=choices, value=selected if selected in sources else None,
                           interactive=bool(choices)) for selected in (selected_a, selected_b))


def compare_voices(project_id: str, voice_name: str, source_a: str, source_b: str, text: str,
                   noise_scale=None, noise_w=None, length_scale=None):
    try:
        sources = _comparison_sources(project_id)
        if not source_a or not source_b or source_a == source_b:
            raise ValueError("Choose two different models or checkpoints for A and B.")
        if source_a not in sources or source_b not in sources:
            raise ValueError("Selected model or checkpoint is no longer available in this project. Refresh the choices.")
        if not text or not text.strip():
            raise ValueError("Enter text for your listening test.")
        parameters = synthesis_parameters(noise_scale, noise_w, length_scale)
        audio = []
        for source in (source_a, source_b):
            _, run_id = sources[source]
            if run_id is None:
                sample, status = synthesize_model(source, text, **parameters)
            else:
                sample, status, _, _ = generate_checkpoint_sample(project_id, run_id, voice_name, source, text, **parameters)
            if not sample:
                raise ValueError(status)
            audio.append(sample)
        settings = ", ".join(f"{name}={value:g}" for name, value in parameters.items()) or "each model's defaults"
        return *audio, f"A: {sources[source_a][0]}\n\nB: {sources[source_b][0]}\n\nBoth read the same test text using {settings}."
    except Exception as error:
        return None, None, f"Comparison failed: {error}"


def publish_selected(model_path: str):
    try:
        model = Path(model_path)
        config = model.with_name(model.name + ".json")
        paths = publish_model(model, config, DATA_DIR / "piper-voices")
        return f"Published {paths[0].name} and {paths[1].name} to the shared Piper voice directory."
    except Exception as error:
        return f"Publish failed: {error}"


def prepare_training_summary(project_id: str, dataset_name: str, mode: str, checkpoint_path: str, warmstart_path: str, device: str, batch_size: Any, seed: int, max_epochs: int,
                             custom_batch: Any = None, workers: Any = "Auto", custom_workers: Any = None, threads: Any = "Auto", hourly_rate: Any = None,
                             checkpoint_interval: Any = 250, learning_rate: Any = 0.0002, learning_rate_d: Any = 0.0001):
    try:
        dataset = _selected_dataset(project_id, dataset_name)
        manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8")) if (dataset / "dataset.json").exists() else {}
        split_sizes = {name: len(values) for name, values in saved_split_indices(dataset).items()}
        selected_device = device
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if selected_device == "auto":
            selected_device = "cuda" if has_cuda else "cpu"
        project = STORE.get_project(project_id)
        options = _training_options(selected_device, batch_size, custom_batch, workers, custom_workers, threads, hourly_rate)
        actual_batch = ("Auto (short training probe)" if selected_device == "cuda" else "Auto (RAM and longest utterance)") if options["batch_size_mode"] == "auto" else options["batch_size"]
        return (
            f"### Review this run before starting\n\n"
            f"Dataset: **{dataset.name}** · Samples: **{manifest.get('sample_count', 'imported')}** · "
            f"Training samples: **{split_sizes['train']}** · "
            f"Total audio: **{_clock(manifest.get('total_seconds', 0))}**\n\n"
            f"Fixed train / validation / test audio: **{_clock(manifest.get('split_durations_seconds', {}).get('train', 0))} / "
            f"{_clock(manifest.get('split_durations_seconds', {}).get('validation', 0))} / "
            f"{_clock(manifest.get('split_durations_seconds', {}).get('test', 0))}** · "
            f"Rows: **{split_sizes['train']} / {split_sizes['validation']} / {split_sizes['test']}**\n\n"
            f"Mode: **{'Fine-tune' if mode == 'finetune' else 'Full training from scratch'}** · "
            f"Checkpoint: **{Path(checkpoint_path).name if mode == 'finetune' and checkpoint_path else 'none'}** · "
            f"Vocoder warm-start: **{Path(warmstart_path).name if mode == 'scratch' and warmstart_path else 'none'}**\n\n"
            f"Sample rate: **22050 Hz** · eSpeak: **{project['espeak_voice']}** · Device: **{selected_device.upper()}** · Batch size: **{actual_batch}** · "
            f"DataLoader workers: **{workers}**\n\n"
            f"Maximum epochs: **{int(max_epochs)}** · Seed: **{int(seed)}** · Uses the saved dataset splits directly; no additional random split.\n\n"
            f"Keep checkpoint every **{checkpoint_interval} epochs**, plus rolling, best-validation and final checkpoints.\n\n"
            f"Generator / discriminator learning rates: **{learning_rate} / {learning_rate_d}**, stepped after each epoch and logged.\n\n"
            f"**Training time estimate:** available after warm-up and at least 10 measured training batches. "
            f"The {int(max_epochs)}-epoch cap is a starting point, not a quality guarantee. "
            f"Piper's fixed per-epoch learning-rate decay gives a different final ratio when this cap changes.",
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


def refresh_training(project_id: str | None):
    summary, chart, logs = training_view(project_id)
    return logs, summary, chart


def refresh_runs(project_id: str | None, run_id: str | None = None, checkpoint_path: str | None = None):
    choices = run_choices(project_id)
    selected = run_id if run_id in {value for _, value in choices} else choices[0][1] if choices else None
    logs, summary, chart = refresh_training(project_id)
    return (
        logs,
        gr.update(choices=choices, value=selected, interactive=bool(choices)),
        summary,
        chart,
        *refresh_checkpoints(project_id, selected, checkpoint_path),
    )


def checkpoint_help(project_id: str | None, run_id: str | None):
    if not checkpoint_choices(project_id, run_id):
        return "No saved checkpoint yet. Keep training; the first rolling checkpoint is normally saved after 25 epochs."
    return "Choose a checkpoint, enter text, and generate a sample. Training can continue. Each sample keeps its own voice download."


def poll_training(project_id: str | None, run_id: str | None, checkpoint_path: str | None, previous):
    updates = list(refresh_runs(project_id, run_id, checkpoint_path))
    signature = [project_id, updates[1].get("choices"), updates[4].get("choices")]
    if previous and signature[0] == previous[0]:
        # Refresh each picker independently, leaving unchanged controls undisturbed.
        if signature[1] == previous[1]:
            updates[1] = gr.update()
        if signature[2] == previous[2]:
            updates[4] = updates[5] = gr.update()
    return *updates, signature


def build_app() -> gr.Blocks:
    from app.pages import build_app as build_pages
    return build_pages(sys.modules[__name__])


def launch_app():
    return build_app().launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        show_error=True, max_file_size="2gb",
        allowed_paths=[str((DATA_DIR / "projects").resolve()), str((DATA_DIR / "exports").resolve())],
    )


if __name__ == "__main__":
    launch_app()
