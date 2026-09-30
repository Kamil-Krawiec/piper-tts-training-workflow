"""Non-destructive technical normalization and recording diagnostics."""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any


def normalize_audio(source: Path, destination: Path, sample_rate: int = 22050) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to normalize browser recordings")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target)],
        capture_output=True, text=True, check=False,
    )
    if process.returncode:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Audio conversion failed: {process.stderr.strip()[-800:]}")
    return target


def trim_edge_silence(source: Path, destination: Path, sample_rate: int = 22050) -> Path:
    """Create a trimmed copy retaining 250 ms of silence at each speech edge."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to prepare training audio")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
            "-af",
            "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB:start_silence=0.25,areverse,silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB:start_silence=0.25,areverse",
            "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target),
        ],
        capture_output=True, text=True, check=False,
    )
    if process.returncode:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Silence trimming failed: {process.stderr.strip()[-800:]}")
    try:
        with wave.open(str(target), "rb") as audio:
            has_audio = audio.getnframes() > 0
    except (OSError, wave.Error):
        has_audio = False
    if not has_audio:
        shutil.copy2(source, target)
    return target


def inspect_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frame_rate = audio.getframerate()
        frame_count = audio.getnframes()
        raw = audio.readframes(frame_count)
    if sample_width != 2 or channels != 1:
        raise ValueError("normalized audio must be mono 16-bit PCM WAV")
    values = [int.from_bytes(raw[index:index + 2], "little", signed=True) for index in range(0, len(raw) - 1, 2)]
    if not values:
        raise ValueError("recording contains no audio samples")
    peak = max(abs(value) for value in values)
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    duration = frame_count / frame_rate
    silence_threshold = 32767 * 10 ** (-45 / 20)
    silent = sum(abs(value) <= silence_threshold for value in values)
    clipping = sum(abs(value) >= 32760 for value in values) / len(values)
    peak_dbfs = 20 * math.log10(peak / 32767) if peak else -120.0
    rms_dbfs = 20 * math.log10(rms / 32767) if rms else -120.0
    silence_ratio = silent / len(values)
    warnings = []
    if duration < 1.2:
        warnings.append("Recording is very short (< 1.2 s).")
    if duration > 15:
        warnings.append("Recording is unusually long (> 15 s).")
    if clipping > 0.001:
        warnings.append("Audio contains clipped peaks.")
    if rms_dbfs < -45:
        warnings.append("Recording level is very quiet.")
    if silence_ratio > 0.75:
        warnings.append("More than 75% of the recording is near silence.")
    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": frame_rate,
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "clipping_ratio": round(clipping, 5),
        "silence_ratio": round(silence_ratio, 4),
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
    }
