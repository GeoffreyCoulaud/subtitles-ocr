"""Stage 11 - deterministic text normalization (ADR-0005).

Supersedes the LLM-based whole-document cleanup of ADR-0002 section 3
Stage 10. Pure functions on event text, no external dependencies, no LLM
client.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel

# Invisible characters to strip after NFKC normalization. NFKC already folds
# NBSP (U+00A0) into a regular space, so it is absent here. ZWJ (U+200D) is
# kept (potentially legitimate in combined glyphs / emoji sequences).
_INVISIBLE_CHARS: tuple[str, ...] = (
    chr(0x200B),  # ZERO WIDTH SPACE
    chr(0x200C),  # ZERO WIDTH NON-JOINER
    chr(0xFEFF),  # BYTE ORDER MARK / ZWNBSP
    chr(0x2060),  # WORD JOINER
)

# Collapse runs of horizontal whitespace (spaces, tabs) into a single space.
# Newlines are explicitly NOT collapsed: they are meaningful (multi-line
# subtitles, exported as \N).
_HSPACE_RUN = re.compile(r"[ \t]+")

# Trim horizontal whitespace around newlines so " \n " becomes "\n".
_NEWLINE_WITH_SPACES = re.compile(r" *\n *")


class NormalizedEvent(BaseModel):
    event_id: int
    cleaned_text: str


class NormalizeResult(BaseModel):
    events: list[NormalizedEvent]


def normalize_text(text: str) -> str:
    """Apply NFKC + invisible-char strip + whitespace normalization."""
    text = unicodedata.normalize("NFKC", text)
    for c in _INVISIBLE_CHARS:
        text = text.replace(c, "")
    text = _HSPACE_RUN.sub(" ", text)
    text = _NEWLINE_WITH_SPACES.sub("\n", text)
    return text.strip()


def is_noise(text: str) -> bool:
    """Return True if `text` is OCR noise that should be pruned.

    Criterion: shorter than 2 characters, or zero alphabetic characters.
    """
    if len(text) < 2:
        return True
    if not any(c.isalpha() for c in text):
        return True
    return False
