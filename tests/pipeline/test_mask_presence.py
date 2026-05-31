"""Tests for the persisted mask-presence sidecar (ADR-0010).

The recorder collects per-frame mean-pooled mask coverage during the OCR
frame_processing stream. The source reads the saved sidecar and answers
``mask_alpha(frame_idx, bbox)`` queries from the new fade detector.

A mask comes in as a uint8 array in ``{0, 255}`` (per Stage 4). After
pooling, each grid cell holds the *fraction* of pixels that were 255 in
the cell's source region — a number in ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from subtitles_ocr.pipeline.frame_processing.mask_presence import (
    MaskPresenceRecorder,
    PersistedMaskSource,
)


def _mask(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def test_recorder_pools_fully_covered_mask_to_alpha_one(tmp_path: Path) -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    mask = np.full((64, 96), 255, dtype=np.uint8)
    rec.record(frame_idx=10, mask=mask)
    rec.save(tmp_path / "mask_grid.npz")

    src = PersistedMaskSource(tmp_path / "mask_grid.npz")
    assert src.mask_alpha(10, (0, 0, 96, 64)) == pytest.approx(1.0, abs=1e-5)


def test_recorder_pools_empty_mask_to_alpha_zero(tmp_path: Path) -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    rec.record(frame_idx=0, mask=_mask((64, 64)))
    rec.save(tmp_path / "mask_grid.npz")

    src = PersistedMaskSource(tmp_path / "mask_grid.npz")
    assert src.mask_alpha(0, (0, 0, 64, 64)) == pytest.approx(0.0, abs=1e-5)


def test_recorder_pools_half_covered_mask_to_alpha_half(tmp_path: Path) -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    mask = _mask((64, 64))
    mask[32:, :] = 255  # bottom half fully covered
    rec.record(frame_idx=0, mask=mask)
    rec.save(tmp_path / "mask_grid.npz")

    src = PersistedMaskSource(tmp_path / "mask_grid.npz")
    top = src.mask_alpha(0, (0, 0, 64, 32))
    bot = src.mask_alpha(0, (0, 32, 64, 64))
    full = src.mask_alpha(0, (0, 0, 64, 64))
    assert top == pytest.approx(0.0, abs=1e-5)
    assert bot == pytest.approx(1.0, abs=1e-5)
    assert full == pytest.approx(0.5, abs=1e-5)


def test_mask_alpha_returns_zero_for_unknown_frame(tmp_path: Path) -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    mask = np.full((32, 32), 255, dtype=np.uint8)
    rec.record(frame_idx=5, mask=mask)
    rec.save(tmp_path / "mask_grid.npz")

    src = PersistedMaskSource(tmp_path / "mask_grid.npz")
    # Frame 99 was never recorded (e.g. ORPHAN). Source returns 0.0 so the
    # caller's "alpha == 0 → no signal" branch handles it.
    assert src.mask_alpha(99, (0, 0, 32, 32)) == 0.0


def test_mask_alpha_handles_subpixel_bbox(tmp_path: Path) -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    mask = _mask((64, 64))
    mask[0:4, 0:4] = 255  # a tiny patch
    rec.record(frame_idx=0, mask=mask)
    rec.save(tmp_path / "mask_grid.npz")

    src = PersistedMaskSource(tmp_path / "mask_grid.npz")
    val = src.mask_alpha(0, (0, 0, 2, 2))
    assert val > 0.0
    assert not np.isnan(val)


def test_mask_alpha_clips_bbox_to_frame_bounds(tmp_path: Path) -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    rec.record(frame_idx=0, mask=np.full((64, 64), 255, dtype=np.uint8))
    rec.save(tmp_path / "mask_grid.npz")

    src = PersistedMaskSource(tmp_path / "mask_grid.npz")
    # bbox extends past the frame edge; source clips silently.
    assert src.mask_alpha(0, (-10, -10, 200, 200)) == pytest.approx(1.0, abs=1e-5)


def test_recorder_save_writes_expected_arrays(tmp_path: Path) -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    rec.record(frame_idx=3, mask=np.full((48, 64), 255, dtype=np.uint8))
    rec.record(frame_idx=4, mask=_mask((48, 64)))
    path = tmp_path / "mask_grid.npz"
    rec.save(path)

    npz = np.load(path)
    assert sorted(npz.files) == sorted(
        ["grid", "frame_indices", "frame_height", "frame_width", "grid_size"]
    )
    assert npz["grid"].shape == (2, 16, 16)
    assert npz["grid"].dtype == np.float32
    assert npz["grid"][0].mean() == pytest.approx(1.0, abs=1e-5)
    assert npz["grid"][1].mean() == pytest.approx(0.0, abs=1e-5)
    assert npz["frame_indices"].tolist() == [3, 4]
    assert int(npz["frame_height"]) == 48
    assert int(npz["frame_width"]) == 64
    assert int(npz["grid_size"]) == 16


def test_recorder_save_empty_produces_loadable_file(tmp_path: Path) -> None:
    # The recorder never received a frame (every aligned frame was an
    # ORPHAN before reaching the sink, for instance). Save must still
    # produce a valid empty sidecar that the source can open.
    rec = MaskPresenceRecorder(grid_size=16)
    rec.save(tmp_path / "mask_grid.npz")
    src = PersistedMaskSource(tmp_path / "mask_grid.npz")
    assert src.mask_alpha(0, (0, 0, 100, 100)) == 0.0


def test_recorder_rejects_non_2d_mask() -> None:
    rec = MaskPresenceRecorder(grid_size=16)
    with pytest.raises(ValueError):
        rec.record(frame_idx=0, mask=np.zeros((10, 10, 3), dtype=np.uint8))
