"""Deterministic Piper dataset materialization."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import wave
from pathlib import Path
from typing import Any, Iterable


def saved_split_indices(dataset_dir: Path) -> dict[str, list[int]]:
    """Validate the saved partition and map sample IDs to the full metadata CSV."""
    root = Path(dataset_dir)
    with (root / "metadata.csv").open(encoding="utf-8", newline="") as stream:
        entries = list(csv.reader(stream, delimiter="|"))
    index = json.loads((root / "sample-index.json").read_text(encoding="utf-8"))
    if not isinstance(index, list) or len(index) != len(entries):
        raise ValueError("sample-index.json must map every dataset metadata row")
    positions = {tuple(row): i for i, row in enumerate(entries)}
    if len(positions) != len(entries):
        raise ValueError("Dataset metadata contains duplicate rows")
    by_id = {}
    for row in index:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key]
                for key in ("sample_id", "filename", "text", "split")):
            raise ValueError("sample-index.json contains an invalid recording")
        pair = (row["filename"], row["text"])
        if pair not in positions:
            raise ValueError("sample-index.json filenames and text must match metadata.csv")
        if row["sample_id"] in by_id:
            raise ValueError("sample-index.json contains duplicate IDs")
        by_id[row["sample_id"]] = (positions[pair], row["split"])
    result = {}
    seen = set()
    for name in ("train", "validation", "test"):
        ids = json.loads((root / "splits" / f"{name}.json").read_text(encoding="utf-8"))
        if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
            raise ValueError(f"Saved {name} split must contain sample IDs")
        for value in ids:
            if value not in by_id or value in seen or by_id[value][1] != name:
                raise ValueError("Saved splits contain unknown, duplicate or inconsistent sample IDs")
            seen.add(value)
        result[name] = sorted(by_id[value][0] for value in ids)
    if seen != set(by_id) or len({i for values in result.values() for i in values}) != len(entries):
        raise ValueError("Saved splits must partition every metadata row exactly once")
    if not result["train"]:
        raise ValueError("Saved dataset has no training samples")
    return result


def _rank(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()


def create_dataset(
    samples: Iterable[dict[str, Any]], output_dir: Path, target_seconds: int | None,
    seed: int = 42,
) -> dict[str, Any]:
    """Select a stable, nested subset and write Piper audio/metadata files."""
    accepted = [sample for sample in samples if sample.get("status", "accepted") == "accepted"]
    evaluation_ranked = sorted(accepted, key=lambda sample: _rank(seed + 1, str(sample["id"])))
    if len(evaluation_ranked) >= 3:
        test_count = max(1, round(len(evaluation_ranked) * 0.05))
        validation_count = max(1, round(len(evaluation_ranked) * 0.10))
        test_count = min(test_count, len(evaluation_ranked) - 2)
        validation_count = min(validation_count, len(evaluation_ranked) - test_count - 1)
    elif len(evaluation_ranked) == 2:
        test_count, validation_count = 1, 0
    else:
        test_count, validation_count = 0, 0
    test_samples = evaluation_ranked[:test_count]
    validation_samples = evaluation_ranked[test_count:test_count + validation_count]
    evaluation_samples = test_samples + validation_samples
    evaluation_ids = {str(sample["id"]) for sample in evaluation_samples}
    train_ranked = sorted((sample for sample in accepted if str(sample["id"]) not in evaluation_ids), key=lambda sample: _rank(seed, str(sample["id"])))
    selected: list[dict[str, Any]] = list(evaluation_samples)
    duration = sum(float(sample.get("duration_seconds", 0)) for sample in selected)
    for sample in train_ranked:
        sample_duration = float(sample.get("duration_seconds", 0))
        if target_seconds is not None and selected and duration >= target_seconds:
            break
        selected.append(sample)
        duration += sample_duration

    root = Path(output_dir)
    splits: dict[str, list[str]] = {
        "train": [],
        "validation": [str(sample["id"]) for sample in validation_samples],
        "test": [str(sample["id"]) for sample in test_samples],
    }
    for sample in selected:
        if str(sample["id"]) not in evaluation_ids:
            splits["train"].append(str(sample["id"]))
    training_ids = set(splits["train"])
    training_samples = [sample for sample in selected if str(sample["id"]) in training_ids]
    filename_by_id = {str(sample["id"]): f"{index:06}.wav" for index, sample in enumerate(selected, 1)}
    write_metadata(selected, root, filename_by_id=filename_by_id)
    write_metadata(training_samples, root, "train_metadata.csv", filename_by_id=filename_by_id)
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        (root / "splits" / f"{split}.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "target_seconds": target_seconds,
        "total_seconds": round(duration, 3),
        "sample_count": len(selected),
        "training_sample_count": len(training_samples),
        "sample_ids": [str(sample["id"]) for sample in selected],
        "evaluation_sample_ids": splits["validation"] + splits["test"],
        "splits": {name: len(ids) for name, ids in splits.items()},
        "split_durations_seconds": {
            name: round(sum(float(sample.get("duration_seconds", 0)) for sample in selected if str(sample["id"]) in set(ids)), 3)
            for name, ids in splits.items()
        },
    }
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    split_by_id = {sample_id: split for split, ids in splits.items() for sample_id in ids}
    index_rows = [
        {"sample_id": str(sample["id"]), "filename": filename_by_id[str(sample["id"])], "text": str(sample["text"]), "split": split_by_id[str(sample["id"])]}
        for sample in selected
    ]
    (root / "sample-index.json").write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_metadata(samples: Iterable[dict[str, Any]], output_dir: Path, metadata_name: str = "metadata.csv", filename_by_id: dict[str, str] | None = None) -> Path:
    root = Path(output_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = root / metadata_name
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="|", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        for index, sample in enumerate(samples, 1):
            audio_path = Path(sample.get("audio_file") or sample.get("normalized_file") or sample.get("raw_file", ""))
            if not audio_path.is_file():
                raise FileNotFoundError(f"audio file is missing for sample {sample.get('id', index)}")
            if audio_path.suffix.lower() != ".wav":
                raise ValueError(f"sample {sample.get('id', index)} is not a normalized WAV file")
            try:
                with wave.open(str(audio_path), "rb") as wav:
                    if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 22050:
                        raise ValueError(f"sample {sample.get('id', index)} must be mono, 16-bit, 22050 Hz WAV")
            except wave.Error as error:
                raise ValueError(f"sample {sample.get('id', index)} is not a valid WAV file") from error
            filename = (filename_by_id or {}).get(str(sample["id"]), f"{index:06}.wav")
            destination = audio_dir / filename
            if audio_path.resolve() != destination.resolve():
                shutil.copy2(audio_path, destination)
            writer.writerow((filename, str(sample["text"]).replace("\n", " ").strip()))
    return metadata
