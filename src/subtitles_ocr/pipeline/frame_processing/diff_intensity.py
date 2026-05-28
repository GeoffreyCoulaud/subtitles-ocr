"""Persisted per-frame diff-intensity grids — fade detection's input.

The animation stage's `_detect_fades` needs `mean_intensity(frame_idx, bbox)`
for frames outside the OCR-grouped events (pre/post-event fade windows).
Recomputing the diff on demand would re-read both source videos thousands of
times. Instead the OCR stage records a 16×16 mean-pooled diff for every
ALIGNED frame it processes and writes a single sidecar at end-of-stage; the
animation stage reads it via `PersistedDiffSource`.

Resolution rationale: a typical subtitle bbox spans ~5 columns × ~2 rows on
a 16×16 grid of a 1280×720 fansub frame, enough mean-pooling resolution for
fade-fitting on the bbox while keeping the sidecar small (~32 MB at 31k
frames).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class DiffIntensityRecorder:
    """Collects per-frame mean-pooled diffs in memory; writes one npz on save.

    Used as a sink inside ``iter_composed_frames`` so the diff array can be
    observed without bloating ComposedFrame (which is the OCR input).
    """

    def __init__(self, grid_size: int = 16) -> None:
        self.grid_size = int(grid_size)
        self._frame_indices: list[int] = []
        self._grids: list[np.ndarray] = []
        self._frame_height: int | None = None
        self._frame_width: int | None = None

    def record(self, frame_idx: int, diff: np.ndarray) -> None:
        if diff.ndim != 2:
            raise ValueError(f"expected 2-D diff array, got shape={diff.shape}")
        h, w = diff.shape
        if self._frame_height is None:
            self._frame_height = h
            self._frame_width = w
        # cv2.resize with INTER_AREA = mean pooling. Cast to float32 to keep
        # the sidecar compact and to match the stored array dtype.
        grid = cv2.resize(
            diff.astype(np.float32, copy=False),
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


class PersistedDiffSource:
    """Implements ``DiffIntensitySource`` Protocol from animation.py."""

    def __init__(self, path: Path) -> None:
        data = np.load(Path(path))
        self._grid: np.ndarray = data["grid"]
        indices: np.ndarray = data["frame_indices"]
        # frame_idx → row in self._grid
        self._row_by_frame: dict[int, int] = {
            int(idx): i for i, idx in enumerate(indices.tolist())
        }
        self._frame_height: int = int(data["frame_height"])
        self._frame_width: int = int(data["frame_width"])
        self._grid_size: int = int(data["grid_size"])

    def mean_intensity(
        self, frame_idx: int, bbox: tuple[int, int, int, int]
    ) -> float:
        row = self._row_by_frame.get(int(frame_idx))
        if row is None or self._frame_height <= 0 or self._frame_width <= 0:
            return 0.0
        x_min, y_min, x_max, y_max = bbox
        # Clip to frame bounds.
        x_min = max(0, min(self._frame_width, int(x_min)))
        x_max = max(0, min(self._frame_width, int(x_max)))
        y_min = max(0, min(self._frame_height, int(y_min)))
        y_max = max(0, min(self._frame_height, int(y_max)))
        # Map to grid coords (axis-aligned bbox in grid space).
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
