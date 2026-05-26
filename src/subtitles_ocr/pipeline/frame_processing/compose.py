"""Stage 5 — masked compose on black background (ADR-0002 §3 Stage 5).

`composed = fansub * (mask / 255)[..., None]`, with background pure black RGB(0,0,0).
"""

from __future__ import annotations

import numpy as np


def compose(fansub: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if fansub.ndim != 3 or fansub.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 fansub image, got shape={fansub.shape}")
    if mask.shape != fansub.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match fansub spatial shape {fansub.shape[:2]}"
        )
    scale = (mask.astype(np.float32) / 255.0)[..., None]
    composed = fansub.astype(np.float32) * scale
    return composed.clip(0, 255).astype(np.uint8)
