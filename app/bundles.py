"""Portable dataset bundle export and safe import."""

from __future__ import annotations

import hashlib
import json
import csv
import shutil
import zipfile
import wave
from app.datasets import saved_split_indices
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(dataset_dir: Path, archive_path: Path, project_info: dict[str, Any]) -> Path:
    source = Path(dataset_dir).resolve()
    archive = Path(archive_path)
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = [path for path in source.rglob("*") if path.is_file() and path.relative_to(source).as_posix() != "manifest.json"]
    hashes = {path.relative_to(source).as_posix(): _sha256(path) for path in files}
    dataset_manifest_path = source / "dataset.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8")) if dataset_manifest_path.exists() else {}
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "application_version": "0.1.0",
        "project": project_info,
        "dataset": dataset_manifest,
        "sample_count": dataset_manifest.get("sample_count"),
        "total_duration_seconds": dataset_manifest.get("total_seconds"),
        "splits": dataset_manifest.get("splits"),
        "split_durations_seconds": dataset_manifest.get("split_durations_seconds"),
        "files": hashes,
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(source).as_posix())
        bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return archive


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name}")
    if "\\" in name or ":" in path.parts[0]:
        raise ValueError(f"unsafe archive path: {name}")
    return path


def import_bundle(archive_path: Path, destination: Path) -> Path:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    total_size = 0
    with zipfile.ZipFile(archive_path) as bundle:
        members = bundle.infolist()
        if len(members) > 100_000:
            raise ValueError("dataset bundle contains too many files")
        for member in members:
            relative = _safe_member(member.filename)
            total_size += member.file_size
            if total_size > 20 * 1024 * 1024 * 1024:
                raise ValueError("dataset bundle exceeds the 20 GiB limit")
            target = (root / Path(*relative.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe archive path: {member.filename}")
        required = {"manifest.json", "metadata.csv", "dataset.json", "sample-index.json", "project-info.json", "splits/train.json", "splits/validation.json", "splits/test.json"}
        missing = required - set(bundle.namelist())
        if missing:
            raise ValueError(f"bundle is missing required files: {', '.join(sorted(missing))}")
        manifest = json.loads(bundle.read("manifest.json"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported dataset bundle schema")
        archive_files = set(bundle.namelist()) - {"manifest.json"}
        hashed_files = set(manifest.get("files", {}))
        if archive_files != hashed_files:
            raise ValueError("dataset bundle must include a SHA-256 hash for every file")
        for member in members:
            relative = _safe_member(member.filename)
            target = (root / Path(*relative.parts)).resolve()
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        for filename, expected_hash in manifest.get("files", {}).items():
            relative = _safe_member(filename)
            path = root.joinpath(*relative.parts)
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(f"dataset bundle hash mismatch: {filename}")
        all_entries: list[list[str]] = []
        for metadata_file in (root / "metadata.csv", root / "train_metadata.csv"):
            if not metadata_file.exists():
                continue
            with metadata_file.open(encoding="utf-8", newline="") as stream:
                entries = list(csv.reader(stream, delimiter="|"))
            if not entries and metadata_file.name == "metadata.csv":
                raise ValueError("dataset metadata is empty")
            if metadata_file.name == "metadata.csv":
                all_entries = entries
            for row in entries:
                if len(row) != 2 or not row[1].strip():
                    raise ValueError("dataset metadata must contain audio filename and non-empty prompt text")
                audio_name = _safe_member(row[0])
                audio_path = root / "audio" / audio_name.name
                if len(audio_name.parts) != 1 or not audio_path.is_file():
                    raise ValueError(f"dataset audio file is missing or unsafe: {row[0]}")
                try:
                    with wave.open(str(audio_path), "rb") as audio:
                        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != 22050:
                            raise ValueError(f"dataset audio must be mono, 16-bit, 22050 Hz WAV: {row[0]}")
                except wave.Error as error:
                    raise ValueError(f"dataset audio is not a valid WAV file: {row[0]}") from error
        project_info = json.loads((root / "project-info.json").read_text(encoding="utf-8"))
        if project_info.get("sample_rate") != 22050 or not project_info.get("language") or not project_info.get("espeak_voice"):
            raise ValueError("project-info.json must declare language, eSpeak voice, and 22050 Hz sample rate")
        dataset_manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        if dataset_manifest.get("schema_version") != 1 or dataset_manifest.get("sample_count") != len(all_entries):
            raise ValueError("dataset.json sample count or schema version is invalid")
        if (
            manifest.get("sample_count") != dataset_manifest.get("sample_count")
            or manifest.get("total_duration_seconds") != dataset_manifest.get("total_seconds")
            or manifest.get("splits") != dataset_manifest.get("splits")
            or manifest.get("split_durations_seconds") != dataset_manifest.get("split_durations_seconds")
        ):
            raise ValueError("dataset bundle manifest summary does not match dataset.json")
        if manifest.get("project", {}).get("language") not in (None, project_info["language"]):
            raise ValueError("dataset bundle project metadata does not match project-info.json")
        saved_split_indices(root)
    return root
