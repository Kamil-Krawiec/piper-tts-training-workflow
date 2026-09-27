"""Deterministic prompt parsing and recording-time estimates."""

from __future__ import annotations

import re
import os
from typing import Iterable

_ABBREVIATIONS = (
    "dr", "prof", "mgr", "inż", "lek", "hab", "np", "itd", "itp", "tj",
    "tzn", "ul", "al", "pl", "nr", "godz", "r", "str", "św", "ks",
)
_ABBREVIATION_PATTERN = re.compile(
    rf"(?i)\b(?:m\.in|w\.w|e\.g|i\.e|{'|'.join(_ABBREVIATIONS)})\.(?=\s+[\wĄĆĘŁŃÓŚŹŻąćęłńóśźż])"
)
_DECIMAL_DOT = re.compile(r"(?<=\d)\.(?=\d)")
_BOUNDARY = re.compile(r"(?<=[.!?…])(?:[\"'»”’)]*)\s+(?=[\"'„“(]*[A-ZĄĆĘŁŃÓŚŹŻ0-9])")
MIN_RECOMMENDED_WORDS = max(1, int(os.environ.get("PIPER_PROMPT_MIN_WORDS", "2")))
MAX_RECOMMENDED_WORDS = max(MIN_RECOMMENDED_WORDS, int(os.environ.get("PIPER_PROMPT_MAX_WORDS", "30")))


def parse_prompts(text: str, mode: str) -> list[str]:
    """Split imported text without rewriting prompt contents."""
    if mode == "lines":
        return [line.strip() for line in text.splitlines() if line.strip()]
    if mode != "prose":
        raise ValueError("mode must be 'prose' or 'lines'")

    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text)]
    prose = " ".join(part for part in paragraphs if part)
    if not prose:
        return []

    protected: list[str] = []

    def preserve(match: re.Match[str]) -> str:
        protected.append(match.group())
        return f"\x00ABBR{len(protected) - 1}\x00"

    prose = _ABBREVIATION_PATTERN.sub(preserve, prose)
    decimal_dots: list[str] = []

    def preserve_decimal(match: re.Match[str]) -> str:
        decimal_dots.append(match.group())
        return f"\x00DEC{len(decimal_dots) - 1}\x00"

    prose = _DECIMAL_DOT.sub(preserve_decimal, prose)
    sentences = [part.strip() for part in _BOUNDARY.split(prose) if part.strip()]
    restored: list[str] = []
    for sentence in sentences:
        for index, token in enumerate(protected):
            sentence = sentence.replace(f"\x00ABBR{index}\x00", token)
        for index, token in enumerate(decimal_dots):
            sentence = sentence.replace(f"\x00DEC{index}\x00", token)
        restored.append(sentence)
    return restored


def estimate_text(prompts: Iterable[str], words_per_minute: int = 140) -> dict[str, int]:
    if words_per_minute < 1 or words_per_minute > 500:
        raise ValueError("speaking rate must be between 1 and 500 words per minute")
    prompt_list = list(prompts)
    word_count = sum(len(re.findall(r"\b\w+\b", prompt, flags=re.UNICODE)) for prompt in prompt_list)
    characters = sum(len(prompt) for prompt in prompt_list)
    seconds = round(word_count * 60 / words_per_minute)
    return {
        "prompts": len(prompt_list),
        "words": word_count,
        "characters": characters,
        "estimated_seconds": seconds,
        "usable_min_seconds": round(seconds * 0.86),
        "usable_max_seconds": round(seconds * 0.97),
    }


def prompt_recommendation(prompt: str, seen: set[str] | None = None) -> str:
    text = prompt.strip()
    words = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    if not text:
        return "Empty"
    if seen is not None and text.casefold() in seen:
        return "Duplicate"
    if words < MIN_RECOMMENDED_WORDS:
        return "Too short (recommendation)"
    if words > MAX_RECOMMENDED_WORDS:
        return "Too long (recommendation)"
    if re.search(r"[!?.,;:]{3,}|^[,.;:!?]", text):
        return "Unusual punctuation"
    return "Ready"
