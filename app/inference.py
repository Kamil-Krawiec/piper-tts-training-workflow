"""Local Piper synthesis and optional OpenAI-compatible API checks."""

from __future__ import annotations

import math
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def synthesis_parameters(noise_scale=None, noise_w=None, length_scale=None) -> dict[str, float]:
    """Omitted overrides preserve the voice config's inference defaults."""
    parameters = {}
    for name, raw in {"noise_scale": noise_scale, "noise_w": noise_w, "length_scale": length_scale}.items():
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a number") from None
        if not math.isfinite(value) or value < 0 or (name == "length_scale" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'length_scale' else 'non-negative'}")
        parameters[name] = value
    return parameters


def synthesize(model: Path, config: Path, text: str, output: Path,
               noise_scale=None, noise_w=None, length_scale=None) -> Path:
    if not text.strip():
        raise ValueError("enter text to synthesize")
    command = [sys.executable, "-m", "piper", "-m", str(model), "-c", str(config), "-f", str(output)]
    for name, value in synthesis_parameters(noise_scale, noise_w, length_scale).items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    result = subprocess.run(
        [*command, "--", text],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Piper synthesis failed: {result.stderr[-1000:]}")
    return Path(output)


def api_health(url: str = "http://piper-api:5000") -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/audio/voices", timeout=3) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError):
        return False


def api_synthesize(text: str, voice: str, output: Path, url: str = "http://piper-api:5000") -> Path:
    import json

    body = json.dumps({"model": "piper", "voice": voice, "input": text, "response_format": "wav"}).encode("utf-8")
    request = urllib.request.Request(url.rstrip("/") + "/v1/audio/speech", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            Path(output).write_bytes(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Piper API returned HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:600]}") from error
    return Path(output)
