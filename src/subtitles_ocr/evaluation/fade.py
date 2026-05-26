"""Fade sub-score (ADR-0006 §5.9)."""

from __future__ import annotations

from fractions import Fraction

from subtitles_ocr.evaluation._tags import parse_event_text
from subtitles_ocr.evaluation.timing import endpoint_score


def fade_pair_score(out_text: str, ref_text: str, fps: Fraction, K: int = 10) -> float | None:
    out = parse_event_text(out_text).fade
    ref = parse_event_text(ref_text).fade
    if out is None and ref is None:
        return None
    if out is None or ref is None:
        return 0.0
    fps_f = float(fps)
    d_in = abs(out[0] - ref[0]) / 1000.0 * fps_f
    d_out = abs(out[1] - ref[1]) / 1000.0 * fps_f
    return (endpoint_score(d_in, K) + endpoint_score(d_out, K)) / 2.0


def fade_score(
    pairs: list[tuple[str, str]],
    fps: Fraction,
    K: int = 10,
) -> float | None:
    scored = [
        s
        for s in (fade_pair_score(o, r, fps, K) for o, r in pairs)
        if s is not None
    ]
    if not scored:
        return None
    return sum(scored) / len(scored)
