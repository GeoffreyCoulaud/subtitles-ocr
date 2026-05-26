"""Stage 2a — audio cross-correlation primitives (ADR-0002 §3 Stage 2a).

Pure-function module: VAD probability post-processing and hierarchical
cross-correlation. Real VAD inference (silero-vad ONNX) lives behind the
`VadModel` Protocol, so tests can inject synthetic probabilities directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


class VadModel(Protocol):
    def compute_probs(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


class AudioLoader(Protocol):
    def load(self, path) -> tuple[np.ndarray, int]: ...


class WindowVerdict(str, Enum):
    NO_MATCH = "no_match"
    CONFIDENT_MATCH = "confident_match"
    AMBIGUOUS = "ambiguous"


@dataclass
class WindowResult:
    start: int  # inclusive sample index (in VAD-rate frames)
    end: int  # exclusive
    offset: int | None  # offset of B vs A in VAD-rate samples (None if no_match)
    peak: float
    snr: float
    verdict: WindowVerdict


def normalize_zscore(probs: np.ndarray) -> np.ndarray:
    """Centre on mean, scale to unit std. Returns zeros if std is degenerate."""
    arr = np.asarray(probs, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if std <= 1e-9:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _normalized_xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear cross-correlation normalized by the geometric mean of the L2 norms.

    Result peak is in `[-1, 1]` for ideal signals (1.0 = perfect match).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 1e-12:
        return np.zeros(a.size + b.size - 1, dtype=np.float64)
    raw = np.correlate(a, b, mode="full")
    return raw / norm


def _verdict_from_curve(
    curve: np.ndarray,
    *,
    thresh_low: float,
    thresh_high: float,
    thresh_snr: float,
    a_len: int,
) -> tuple[int | None, float, float, WindowVerdict]:
    if curve.size == 0:
        return None, 0.0, 0.0, WindowVerdict.NO_MATCH
    peak_idx = int(np.argmax(curve))
    peak_val = float(curve[peak_idx])
    # `np.correlate(a, b, "full")` index k corresponds to lag (k - (b_len-1)).
    # We pre-computed against b of length b_len: offset of b vs a is `b_len - 1 - peak_idx`.
    b_len = curve.size - a_len + 1
    offset = (b_len - 1) - peak_idx

    # peak_to_noise: peak / std over the curve excluding ±1 sample neighbourhood
    mask = np.ones_like(curve, dtype=bool)
    lo = max(0, peak_idx - 1)
    hi = min(curve.size, peak_idx + 2)
    mask[lo:hi] = False
    noise = curve[mask]
    noise_std = float(noise.std()) if noise.size > 0 else 0.0
    snr = peak_val / noise_std if noise_std > 1e-9 else float("inf")

    if peak_val < thresh_low:
        return None, peak_val, snr, WindowVerdict.NO_MATCH
    if peak_val >= thresh_high and snr >= thresh_snr:
        return offset, peak_val, snr, WindowVerdict.CONFIDENT_MATCH
    return offset, peak_val, snr, WindowVerdict.AMBIGUOUS


def hierarchical_cross_correlate(
    probs_fansub: np.ndarray,
    probs_raw: np.ndarray,
    *,
    sample_rate_hz: float,
    window_init_samples: int,
    window_floor_samples: int,
    thresh_low: float,
    thresh_high: float,
    thresh_snr: float,
) -> list[WindowResult]:
    """3-pass recursive subdivision of windows, per ADR-0002 §3 Stage 2a."""
    n = probs_fansub.size
    if n == 0:
        return []

    results: list[WindowResult] = []

    def recurse(start: int, end: int, depth: int) -> None:
        window_size = end - start
        if window_size <= 0:
            return
        a = probs_fansub[start:end]
        # Search a wider raw slice so positive AND negative offsets fit.
        raw_start = max(0, start - window_size)
        raw_end = min(probs_raw.size, end + window_size)
        b = probs_raw[raw_start:raw_end]
        curve = _normalized_xcorr(a, b)
        offset_local, peak, snr, verdict = _verdict_from_curve(
            curve,
            thresh_low=thresh_low,
            thresh_high=thresh_high,
            thresh_snr=thresh_snr,
            a_len=a.size,
        )
        # Re-base offset to absolute coordinates: position of b in absolute index
        # is raw_start + local match; reported offset is "raw_index - fansub_index".
        abs_offset: int | None
        if offset_local is None:
            abs_offset = None
        else:
            # Local convention: offset_local = raw_local - 0 (since a starts at 0 of slice)
            # In absolute terms: (raw_start + raw_local) - start
            abs_offset = raw_start + offset_local - start

        if verdict == WindowVerdict.AMBIGUOUS and window_size > window_floor_samples:
            mid = start + window_size // 2
            recurse(start, mid, depth + 1)
            recurse(mid, end, depth + 1)
            return

        # Convert ambiguous-at-floor → orphan (conservative, per ADR-0002)
        final_verdict = verdict
        final_offset = abs_offset
        if verdict == WindowVerdict.AMBIGUOUS:
            final_verdict = WindowVerdict.NO_MATCH
            final_offset = None

        results.append(
            WindowResult(
                start=start,
                end=end,
                offset=final_offset,
                peak=peak,
                snr=snr,
                verdict=final_verdict,
            )
        )

    # Pass 1: coarse split
    step = window_init_samples
    cursor = 0
    while cursor < n:
        recurse(cursor, min(cursor + step, n), depth=0)
        cursor += step

    results.sort(key=lambda r: r.start)
    return results


def post_filter_isolated_matches(
    windows: list[WindowResult],
    *,
    sample_rate_hz: float,
    min_match_s: float,
    offset_tolerance_frames: int,
) -> list[WindowResult]:
    """Reclassify tiny isolated matches as ORPHAN.

    A match is "isolated" if its duration < `min_match_s` AND neither of its
    immediate neighbours (left/right) is a CONFIDENT_MATCH with an offset that
    differs by no more than `offset_tolerance_frames`.
    """
    out = [
        WindowResult(
            start=w.start, end=w.end, offset=w.offset, peak=w.peak, snr=w.snr, verdict=w.verdict
        )
        for w in windows
    ]
    min_samples = min_match_s * sample_rate_hz
    for i, w in enumerate(out):
        if w.verdict != WindowVerdict.CONFIDENT_MATCH:
            continue
        duration = w.end - w.start
        if duration >= min_samples:
            continue
        has_coherent_neighbour = False
        for j in (i - 1, i + 1):
            if 0 <= j < len(out):
                n = out[j]
                if (
                    n.verdict == WindowVerdict.CONFIDENT_MATCH
                    and n.offset is not None
                    and w.offset is not None
                    and abs(n.offset - w.offset) <= offset_tolerance_frames
                ):
                    has_coherent_neighbour = True
                    break
        if not has_coherent_neighbour:
            out[i] = WindowResult(
                start=w.start,
                end=w.end,
                offset=None,
                peak=w.peak,
                snr=w.snr,
                verdict=WindowVerdict.NO_MATCH,
            )
    return out
