"""Stage 3 — gradient-domain diff (ADR-0002 §3 Stage 3).

Recipe:
    1. BT.709 luma conversion (RGB uint8 → float32 grayscale).
    2. Local contrast normalization (LCN): (px - mean_local) / max(std_local, std_floor).
    3. Sobel 3x3 gradient magnitude.
    4. abs(gmag_fansub - gmag_raw) as float32.
"""

from __future__ import annotations

import cv2
import numpy as np

from subtitles_ocr.config import FrameProcessingConfig


def _bt709_luma(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={image.shape}")
    img = image.astype(np.float32)
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def _local_contrast_normalize(gray: np.ndarray, sigma: float, std_floor: float) -> np.ndarray:
    # ksize derived from sigma per OpenCV convention; ensure odd >=3.
    ksize = max(3, int(round(sigma * 6)) | 1)
    mean_local = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    sq_local = cv2.GaussianBlur(gray * gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    var_local = np.maximum(sq_local - mean_local * mean_local, 0.0)
    std_local = np.sqrt(var_local)
    return (gray - mean_local) / np.maximum(std_local, std_floor)


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def compute_diff(
    fansub: np.ndarray, raw: np.ndarray, config: FrameProcessingConfig
) -> np.ndarray:
    if fansub.shape != raw.shape:
        raise ValueError(
            f"fansub and raw shapes must match, got {fansub.shape} vs {raw.shape}"
        )
    fansub_gray = _bt709_luma(fansub)
    raw_gray = _bt709_luma(raw)
    fansub_lcn = _local_contrast_normalize(fansub_gray, config.lcn_sigma, config.std_floor)
    raw_lcn = _local_contrast_normalize(raw_gray, config.lcn_sigma, config.std_floor)
    gmag_fansub = _sobel_magnitude(fansub_lcn)
    gmag_raw = _sobel_magnitude(raw_lcn)
    return np.abs(gmag_fansub - gmag_raw).astype(np.float32)
