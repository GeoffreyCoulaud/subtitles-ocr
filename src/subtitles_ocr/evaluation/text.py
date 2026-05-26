"""Text content sub-scores (ADR-0006 §5.1-5.2)."""

from __future__ import annotations

import re
import unicodedata

_OVERRIDE_BLOCK = re.compile(r"\{[^{}]*\}")
_LINE_BREAK = re.compile(r"\\[Nn]")
_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[.,;:!?\"'`]")

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"',
    "…": "...",
}


def _strip_overrides_and_breaks(raw: str) -> str:
    s = _OVERRIDE_BLOCK.sub("", raw)
    s = _LINE_BREAK.sub(" ", s)
    return s


def normalise_plain(raw: str) -> str:
    s = _strip_overrides_and_breaks(raw)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    for src, dst in _QUOTES.items():
        s = s.replace(src, dst)
    s = _PUNCTUATION.sub("", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def normalise_exact(raw: str) -> str:
    s = _strip_overrides_and_breaks(raw)
    s = unicodedata.normalize("NFC", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / m


def text_plain_pair_score(out_text: str, ref_text: str) -> float:
    return _similarity(normalise_plain(out_text), normalise_plain(ref_text))


def text_exact_pair_score(out_text: str, ref_text: str) -> float:
    return _similarity(normalise_exact(out_text), normalise_exact(ref_text))


def text_plain_score(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(text_plain_pair_score(o, r) for o, r in pairs) / len(pairs)


def text_exact_score(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(text_exact_pair_score(o, r) for o, r in pairs) / len(pairs)
