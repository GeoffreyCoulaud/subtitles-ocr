"""Tests for the persisted diff-intensity sidecar (fade detection input).

The recorder collects per-frame mean-pooled diff grids during OCR's
frame_processing stream and writes a single npz at end-of-stage. The source
reads that npz and answers ``mean_intensity(frame_idx, bbox)`` queries from
the animation stage's fade detector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from subtitles_ocr.pipeline.frame_processing.diff_intensity import (
    DiffIntensityRecorder,
    PersistedDiffSource,
)


def test_recorder_pools_uniform_diff_into_uniform_grid(tmp_path: Path) -> None:
    rec = DiffIntensityRecorder(grid_size=16)
    diff = np.full((64, 96), 0.5, dtype=np.float32)
    rec.record(frame_idx=10, diff=diff)
    rec.save(tmp_path / "diff_grid.npz")

    src = PersistedDiffSource(tmp_path / "diff_grid.npz")
    # Whole-frame bbox → mean is the constant value.
    mean = src.mean_intensity(10, (0, 0, 96, 64))
    assert mean == pytest.approx(0.5, abs=1e-5)


def test_recorder_pools_split_diff_preserving_structure(tmp_path: Path) -> None:
    rec = DiffIntensityRecorder(grid_size=16)
    diff = np.zeros((64, 64), dtype=np.float32)
    # Bright in the bottom half, dark in the top half.
    diff[32:, :] = 1.0
    rec.record(frame_idx=0, diff=diff)
    rec.save(tmp_path / "diff_grid.npz")

    src = PersistedDiffSource(tmp_path / "diff_grid.npz")
    top_mean = src.mean_intensity(0, (0, 0, 64, 32))
    bot_mean = src.mean_intensity(0, (0, 32, 64, 64))
    assert top_mean == pytest.approx(0.0, abs=1e-5)
    assert bot_mean == pytest.approx(1.0, abs=1e-5)


def test_mean_intensity_returns_zero_for_unknown_frame(tmp_path: Path) -> None:
    rec = DiffIntensityRecorder(grid_size=16)
    rec.record(frame_idx=5, diff=np.ones((32, 32), dtype=np.float32))
    rec.save(tmp_path / "diff_grid.npz")

    src = PersistedDiffSource(tmp_path / "diff_grid.npz")
    # Frame 99 was never recorded (e.g. ORPHAN). Source returns 0.0 so the
    # fade detector's "anchor_intensity <= 0 → skip" branch handles it.
    assert src.mean_intensity(99, (0, 0, 32, 32)) == 0.0


def test_mean_intensity_rounds_subpixel_bbox_to_at_least_one_cell(tmp_path: Path) -> None:
    rec = DiffIntensityRecorder(grid_size=16)
    diff = np.zeros((64, 64), dtype=np.float32)
    diff[0:4, 0:4] = 2.0  # bright corner that occupies < 1 grid cell
    rec.record(frame_idx=0, diff=diff)
    rec.save(tmp_path / "diff_grid.npz")

    src = PersistedDiffSource(tmp_path / "diff_grid.npz")
    # A bbox smaller than one grid cell must still query the grid cell that
    # covers it (never return NaN, never raise).
    val = src.mean_intensity(0, (0, 0, 2, 2))
    assert val > 0.0
    assert not np.isnan(val)


def test_recorder_save_writes_expected_arrays(tmp_path: Path) -> None:
    rec = DiffIntensityRecorder(grid_size=16)
    rec.record(frame_idx=3, diff=np.ones((48, 64), dtype=np.float32))
    rec.record(frame_idx=4, diff=np.zeros((48, 64), dtype=np.float32))
    path = tmp_path / "diff_grid.npz"
    rec.save(path)

    npz = np.load(path)
    assert sorted(npz.files) == sorted(
        ["grid", "frame_indices", "frame_height", "frame_width", "grid_size"]
    )
    assert npz["grid"].shape == (2, 16, 16)
    assert npz["grid"].dtype == np.float32
    assert npz["frame_indices"].tolist() == [3, 4]
    assert int(npz["frame_height"]) == 48
    assert int(npz["frame_width"]) == 64
    assert int(npz["grid_size"]) == 16


def test_source_handles_bbox_clipped_to_frame_bounds(tmp_path: Path) -> None:
    rec = DiffIntensityRecorder(grid_size=16)
    rec.record(frame_idx=0, diff=np.ones((64, 64), dtype=np.float32))
    rec.save(tmp_path / "diff_grid.npz")

    src = PersistedDiffSource(tmp_path / "diff_grid.npz")
    # bbox extends past the frame edge; the source clips silently.
    mean = src.mean_intensity(0, (-10, -10, 200, 200))
    assert mean == pytest.approx(1.0, abs=1e-5)
