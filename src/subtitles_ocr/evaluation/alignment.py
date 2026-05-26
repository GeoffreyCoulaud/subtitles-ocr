"""Temporal-IoU alignment of output cues to reference cues (ADR-0006 §4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class Cue(NamedTuple):
    start_s: float
    end_s: float


@dataclass(frozen=True)
class AlignmentResult:
    pairs: list[tuple[int, int, float]]  # (ref_index, out_index, iou)
    unmatched_ref: list[int]
    unmatched_out: list[int]


def _iou(a: Cue, b: Cue) -> float:
    overlap = max(0.0, min(a.end_s, b.end_s) - max(a.start_s, b.start_s))
    if overlap <= 0.0:
        return 0.0
    union = max(a.end_s, b.end_s) - min(a.start_s, b.start_s)
    if union <= 0.0:
        return 0.0
    return overlap / union


def align_by_iou(refs: list[Cue], outs: list[Cue]) -> AlignmentResult:
    # For each ref, find best output cue by IoU. Resolve contention by IoU.
    candidates: list[tuple[float, int, int]] = []
    for ri, r in enumerate(refs):
        best_iou = 0.0
        best_oi = -1
        for oi, o in enumerate(outs):
            iou = _iou(r, o)
            if iou > best_iou:
                best_iou = iou
                best_oi = oi
        if best_oi >= 0:
            candidates.append((best_iou, ri, best_oi))

    # Sort by IoU descending; greedily assign, dropping refs whose output is taken.
    candidates.sort(reverse=True)
    used_out: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    matched_ref: set[int] = set()
    for iou, ri, oi in candidates:
        if oi in used_out:
            continue
        pairs.append((ri, oi, iou))
        used_out.add(oi)
        matched_ref.add(ri)

    pairs.sort(key=lambda p: p[0])  # back to ref order for stable reports

    unmatched_ref = [i for i in range(len(refs)) if i not in matched_ref]
    unmatched_out = [i for i in range(len(outs)) if i not in used_out]
    return AlignmentResult(pairs=pairs, unmatched_ref=unmatched_ref, unmatched_out=unmatched_out)
