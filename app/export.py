"""Piper ONNX export and model publication helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def validate_voice_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
        raise ValueError("voice name may contain only letters, numbers, underscores, and hyphens")
    return name


def snapshot_checkpoint(source: Path, destination: Path) -> Path:
    """Copy saved weights without serving a file the trainer may overwrite."""
    before = source.stat()
    shutil.copyfile(source, destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        destination.unlink(missing_ok=True)
        raise ValueError("Checkpoint changed while being copied. Please try again.")
    return destination


def export_onnx(checkpoint: Path, config_json: Path, output_dir: Path, voice_name: str) -> tuple[Path, Path]:
    name = validate_voice_name(voice_name)
    checkpoint = Path(checkpoint)
    config_json = Path(config_json)
    if not checkpoint.is_file() or not config_json.is_file():
        raise FileNotFoundError("training checkpoint or generated Piper config is missing")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"{name}.onnx"
    config = root / f"{name}.onnx.json"
    result = subprocess.run(
        [sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(checkpoint), "--output-file", str(model)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        model.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX export failed: {result.stderr[-1200:]}")
    shutil.copy2(config_json, config)
    return model, config


def package_model(model: Path, config: Path, archive: Path) -> Path:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(model, Path(model).name)
        bundle.write(config, Path(config).name)
    return archive


def publish_model(model: Path, config: Path, voice_dir: Path) -> tuple[Path, Path]:
    root = Path(voice_dir)
    root.mkdir(parents=True, exist_ok=True)
    destinations = (root / Path(model).name, root / Path(config).name)
    shutil.copy2(model, destinations[0])
    shutil.copy2(config, destinations[1])
    return destinations
