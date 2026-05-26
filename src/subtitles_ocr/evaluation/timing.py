"""Timing sub-score (ADR-0006 §5.3)."""

from __future__ import annotations

from fractions import Fraction

from subtitles_ocr.evaluation.alignment import Cue


def endpoint_score(delta_frames: float, K: int) -> float:
    d = abs(delta_frames)
    if d < 1.0:
        return 1.0
    return max(0.0, 1.0 - d / K)


def timing_pair_score(out: Cue, ref: Cue, fps: Fraction, K: int = 10) -> float:
    fps_f = float(fps)
    d_start = abs(out.start_s - ref.start_s) * fps_f
    d_end = abs(out.end_s - ref.end_s) * fps_f
    return (endpoint_score(d_start, K) + endpoint_score(d_end, K)) / 2.0


def timing_score(
    pairs: list[tuple[Cue, Cue]],
    fps: Fraction,
    K: int = 10,
) -> float | None:
    if not pairs:
        return None
    return sum(timing_pair_score(o, r, fps, K) for o, r in pairs) / len(pairs)
