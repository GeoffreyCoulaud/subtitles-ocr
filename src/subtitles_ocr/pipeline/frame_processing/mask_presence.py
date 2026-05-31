"""Persisted per-frame mask-presence grids — input to ADR-0010's fade detector.

Stage 4's `make_mask` produces a per-frame uint8 mask in ``{0, 255}`` marking
which pixels currently belong to a burnt-in subtitle. To detect fades, the
animation stage needs to know how much of that mask covers a given bounding
box on frames outside the OCR-grouped event (the pre/post-event fade
windows). Recomputing the mask there would re-read both source videos
thousands of times; instead the OCR stage records a 16×16 mean-pooled view
of the mask for every ALIGNED frame and writes a single ``mask_grid.npz``
sidecar at end-of-stage. The animation stage reads it via
``PersistedMaskSource``.

The grid stores the **fraction** of pixels equal to 255 in each cell, as
``float32`` in ``[0, 1]``. A whole-bbox query averages those fractions
over the grid cells the bbox covers.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class MaskPresenceRecorder:
    """Collects per-frame mean-pooled mask coverage; writes one npz on save.

    Used as a sink inside ``iter_composed_frames`` so the mask coverage is
    observed without bloating ``ComposedFrame`` (which is the OCR input).
    """

    def __init__(self, grid_size: int = 16) -> None:
        self.grid_size = int(grid_size)
        self._frame_indices: list[int] = []
        self._grids: list[np.ndarray] = []
        self._frame_height: int | None = None
        self._frame_width: int | None = None

    def record(self, frame_idx: int, mask: np.ndarray) -> None:
        if mask.ndim != 2:
            raise ValueError(f"expected 2-D mask array, got shape={mask.shape}")
        h, w = mask.shape
        if self._frame_height is None:
            self._frame_height = h
            self._frame_width = w
        # Convert {0, 255} → [0, 1] before mean pooling so each grid cell
        # holds the *fraction* of pixels that were "subtitle" in its source
        # region.
        normalized = mask.astype(np.float32) / 255.0
        grid = cv2.resize(
            normalized,
            (self.grid_size, self.grid_size),
            interpolation=cv2.INTER_AREA,
        )
        self._frame_indices.append(int(frame_idx))
        self._grids.append(grid)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._grids:
            stack = np.empty((0, self.grid_size, self.grid_size), dtype=np.float32)
            indices = np.empty((0,), dtype=np.int32)
            fh = fw = 0
        else:
            stack = np.stack(self._grids).astype(np.float32, copy=False)
            indices = np.asarray(self._frame_indices, dtype=np.int32)
            fh = int(self._frame_height or 0)
            fw = int(self._frame_width or 0)
        np.savez(
            path,
            grid=stack,
            frame_indices=indices,
            frame_height=np.int32(fh),
            frame_width=np.int32(fw),
            grid_size=np.int32(self.grid_size),
        )


class PersistedMaskSource:
    """Implements ``MaskPresenceSource`` Protocol (declared in animation.py)."""

    def __init__(self, path: Path) -> None:
        data = np.load(Path(path))
        self._grid: np.ndarray = data["grid"]
        indices: np.ndarray = data["frame_indices"]
        self._row_by_frame: dict[int, int] = {
            int(idx): i for i, idx in enumerate(indices.tolist())
        }
        self._frame_height: int = int(data["frame_height"])
        self._frame_width: int = int(data["frame_width"])
        self._grid_size: int = int(data["grid_size"])

    def mask_alpha(
        self, frame_idx: int, bbox: tuple[int, int, int, int]
    ) -> float:
        row = self._row_by_frame.get(int(frame_idx))
        if row is None or self._frame_height <= 0 or self._frame_width <= 0:
            return 0.0
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, min(self._frame_width, int(x_min)))
        x_max = max(0, min(self._frame_width, int(x_max)))
        y_min = max(0, min(self._frame_height, int(y_min)))
        y_max = max(0, min(self._frame_height, int(y_max)))
        sx = self._grid_size / self._frame_width
        sy = self._grid_size / self._frame_height
        cx1 = int(np.floor(x_min * sx))
        cy1 = int(np.floor(y_min * sy))
        cx2 = int(np.ceil(x_max * sx))
        cy2 = int(np.ceil(y_max * sy))
        # Ensure at least one cell, even for sub-cell bboxes.
        cx2 = max(cx2, cx1 + 1)
        cy2 = max(cy2, cy1 + 1)
        cx2 = min(cx2, self._grid_size)
        cy2 = min(cy2, self._grid_size)
        if cx2 <= cx1 or cy2 <= cy1:
            return 0.0
        return float(self._grid[row, cy1:cy2, cx1:cx2].mean())
