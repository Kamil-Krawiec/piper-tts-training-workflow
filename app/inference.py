"""Local Piper synthesis and optional OpenAI-compatible API checks."""

from __future__ import annotations

import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def synthesize(model: Path, config: Path, text: str, output: Path) -> Path:
    if not text.strip():
        raise ValueError("enter text to synthesize")
    result = subprocess.run(
        [sys.executable, "-m", "piper", "-m", str(model), "-c", str(config), "-f", str(output), "--", text],
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
