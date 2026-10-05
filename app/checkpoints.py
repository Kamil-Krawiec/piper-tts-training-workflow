"""Curated, pinned Piper checkpoint registry and persistent cache."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


CHECKPOINTS: dict[str, dict[str, Any]] = {
    "pl_PL-darkman-medium": {
        "id": "pl_PL-darkman-medium",
        "language": "pl_PL",
        "quality": "medium",
        "sample_rate": 22050,
        "speakers": 1,
        "source": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/pl/pl_PL/darkman/medium",
        "checkpoint_source": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/6a9b3be/pl/pl_PL/darkman/medium/epoch=4909-step=1454360.ckpt?download=true",
        "config_source": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/6a9b3be/pl/pl_PL/darkman/medium/config.json?download=true",
        "license": "MIT",
        "size_bytes": 845898328,
        "sha256": "9e5bd955fb920a6feccede80f7cc961caa540a2f75e75269c465c37fa5884a63",
    }
}


def curated_checkpoints() -> list[dict[str, Any]]:
    return [dict(item) for item in CHECKPOINTS.values()]


def download_checkpoint(checkpoint_id: str, cache_dir: Path, timeout: int = 60) -> Path:
    item = CHECKPOINTS.get(checkpoint_id)
    if not item:
        raise ValueError("checkpoint is not in the curated registry")
    root = Path(cache_dir) / checkpoint_id
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "checkpoint.ckpt"
    if destination.is_file() and destination.stat().st_size == item["size_bytes"] and _sha256(destination) == item["sha256"]:
        return destination
    temporary = root / "checkpoint.ckpt.part"
    request = urllib.request.Request(item["checkpoint_source"], headers={"User-Agent": "piper-voice-trainer/0.1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size != item["size_bytes"] or _sha256(temporary) != item["sha256"]:
            raise ValueError("upstream checkpoint did not match the pinned size and SHA-256")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    file = Path(path)
    if not file.is_file() or file.suffix != ".ckpt":
        raise ValueError("select an existing .ckpt file")
    return {"path": str(file.resolve()), "size_bytes": file.stat().st_size, "sha256": _sha256(file)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_config(checkpoint: Path, config_path: str | None, cache_dir: Path) -> Path:
    """Use this checkpoint's inference config, fetching only a known base voice's config."""
    if checkpoint.suffix != ".ckpt" or not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError("Select an existing, nonempty .ckpt file")
    candidates = [Path(config_path).resolve()] if config_path else [checkpoint.with_name("voice.onnx.json"), checkpoint.with_name("config.json")]
    config = next((path for path in candidates if path.is_file()), None)
    if config is None and not config_path:
        for checkpoint_id, item in CHECKPOINTS.items():
            if checkpoint == (cache_dir / checkpoint_id / "checkpoint.ckpt").resolve():
                request = urllib.request.Request(item["config_source"], headers={"User-Agent": "piper-voice-trainer"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = json.load(response)
                config = checkpoint.with_name("config.json")
                config.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                break
    if config is None:
        raise ValueError("Matching Piper config is missing. Provide its JSON path, or place config.json beside the checkpoint.")
    data = json.loads(config.read_text(encoding="utf-8"))
    if not (data.get("audio", {}).get("sample_rate") and data.get("espeak", {}).get("voice") and data.get("phoneme_id_map")):
        raise ValueError("Select a Piper inference config with audio, espeak and phoneme_id_map settings")
    return config
