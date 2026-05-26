"""Stage 2b/2c — phash refinement and phash-only fallback (ADR-0002 §3 Stage 2).

The `FrameSource` Protocol decouples the matching logic from real frame
decoding + `cv2.img_hash.PHash` so tests can inject deterministic hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class FrameSource(Protocol):
    def get_phash(self, source: str, frame_idx: int) -> int: ...


def hamming_distance64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


@dataclass
class PhashMatch:
    fansub_frame_idx: int
    raw_frame_idx: int | None
    distance: int | None


@dataclass
class RefinementResult:
    per_frame_offsets: list[int]
    per_frame_distances: list[int]
    disagreement_ratio: float


def refine_phash(
    fansub_frames: Iterable[int],
    *,
    predicted_offset: int,
    frame_source: FrameSource,
    window: int,
    thresh_agree: int,
) -> RefinementResult:
    offsets: list[int] = []
    distances: list[int] = []
    disagreements = 0
    total = 0
    for n in fansub_frames:
        h_fan = frame_source.get_phash("fansub", n)
        best_off = predicted_offset
        best_dist = 65
        for k in range(-window, window + 1):
            cand_raw = n + predicted_offset + k
            if cand_raw < 0:
                continue
            h_raw = frame_source.get_phash("raw", cand_raw)
            d = hamming_distance64(h_fan, h_raw)
            if d < best_dist:
                best_dist = d
                best_off = predicted_offset + k
        offsets.append(best_off)
        distances.append(best_dist)
        if best_dist > thresh_agree:
            disagreements += 1
        total += 1
    ratio = disagreements / total if total > 0 else 0.0
    return RefinementResult(
        per_frame_offsets=offsets,
        per_frame_distances=distances,
        disagreement_ratio=ratio,
    )


def phash_fallback(
    fansub_frames: Iterable[int],
    *,
    raw_total_frames: int,
    frame_source: FrameSource,
    w_initial: int,
    w_min: int,
    w_max: int,
    grow_step: int,
    shrink_step: int,
    thresh_match: int,
) -> list[PhashMatch]:
    """Per-frame phash matching with adaptive search window (ADR-0002 §3 Stage 2c)."""
    matches: list[PhashMatch] = []
    current_offset = 0
    window = w_initial
    consecutive_hits = 0
    last_match: int | None = None

    fansub_list = list(fansub_frames)
    for n in fansub_list:
        h_fan = frame_source.get_phash("fansub", n)
        anchor = last_match + 1 if last_match is not None else n + current_offset
        best_dist = 65
        best_raw: int | None = None
        for k in range(-window, window + 1):
            cand = anchor + k
            if cand < 0 or cand >= raw_total_frames:
                continue
            h_raw = frame_source.get_phash("raw", cand)
            d = hamming_distance64(h_fan, h_raw)
            if d < best_dist:
                best_dist = d
                best_raw = cand

        if best_raw is not None and best_dist <= thresh_match:
            matches.append(PhashMatch(fansub_frame_idx=n, raw_frame_idx=best_raw, distance=best_dist))
            consecutive_hits += 1
            current_offset = best_raw - n
            last_match = best_raw
            if consecutive_hits >= 3 and window > w_min:
                window = max(w_min, window - shrink_step)
        else:
            matches.append(PhashMatch(fansub_frame_idx=n, raw_frame_idx=None, distance=None))
            consecutive_hits = 0
            if window < w_max:
                window = min(w_max, window + grow_step)
    return matches
