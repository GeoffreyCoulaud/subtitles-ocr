"""Stage 4 — hysteresis-thresholded mask formation (ADR-0002 §3 Stage 4).

Pipeline:
    1. Gaussian smoothing of the diff map (sigma = mask_smoothing_sigma).
    2. Hysteresis thresholding (Canny-style):
       - pixels above T_high are seeds.
       - pixels above T_low are accepted iff 8-connected to a seed.
    3. Connected-components filter by size (AREA_MIN <= area <= AREA_MAX).
    4. Morphological dilation with a 3x3 kernel, `mask_dilation_iter` iterations.

Output: uint8 mask in {0, 255}.
"""

from __future__ import annotations

import cv2
import numpy as np

from subtitles_ocr.config import FrameProcessingConfig


def _hysteresis(diff: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    seeds = (diff >= t_high).astype(np.uint8)
    candidates = (diff >= t_low).astype(np.uint8)
    # Mark each candidate component containing at least one seed.
    num_labels, labels = cv2.connectedComponents(candidates, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(diff, dtype=np.uint8)
    seed_labels = np.unique(labels[seeds > 0])
    keep = np.isin(labels, seed_labels[seed_labels > 0])
    return keep.astype(np.uint8) * 255


def _filter_by_area(mask: np.ndarray, area_min: int, area_max: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(mask, dtype=np.uint8)
    # label 0 is background; iterate over the foreground components.
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area_min <= area <= area_max:
            out[labels == label_idx] = 255
    return out


def make_mask(diff_map: np.ndarray, config: FrameProcessingConfig) -> np.ndarray:
    if diff_map.dtype != np.float32:
        diff = diff_map.astype(np.float32)
    else:
        diff = diff_map

    sigma = float(config.mask_smoothing_sigma)
    if sigma > 0:
        ksize = max(3, int(round(sigma * 6)) | 1)
        smoothed = cv2.GaussianBlur(diff, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    else:
        smoothed = diff

    binary = _hysteresis(smoothed, t_low=config.mask_t_low, t_high=config.mask_t_high)
    filtered = _filter_by_area(binary, config.mask_area_min, config.mask_area_max)

    if config.mask_dilation_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        filtered = cv2.dilate(filtered, kernel, iterations=config.mask_dilation_iter)

    return filtered
