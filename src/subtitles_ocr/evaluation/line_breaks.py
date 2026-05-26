"""Line-break sub-score (ADR-0006 §5.6, v1 count-only variant)."""

from __future__ import annotations

import re

_OVERRIDE_BLOCK = re.compile(r"\{[^{}]*\}")
_LINE_BREAK = re.compile(r"\\[Nn]")


def count_breaks(raw: str) -> int:
    s = _OVERRIDE_BLOCK.sub("", raw)
    return len(_LINE_BREAK.findall(s))


def line_breaks_pair_score(out_text: str, ref_text: str) -> float:
    return 1.0 if count_breaks(out_text) == count_breaks(ref_text) else 0.0


def line_breaks_score(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(line_breaks_pair_score(o, r) for o, r in pairs) / len(pairs)
