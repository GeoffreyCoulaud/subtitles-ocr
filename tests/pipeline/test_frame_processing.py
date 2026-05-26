"""Tests for stages 3-5: diff, mask, compose, and the iter_composed_frames driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from subtitles_ocr.config import FrameProcessingConfig
from subtitles_ocr.pipeline.alignment.stage import AlignmentResult, AlignmentSegment
from subtitles_ocr.pipeline.frame_processing import ComposedFrame, iter_composed_frames
from subtitles_ocr.pipeline.frame_processing.compose import compose
from subtitles_ocr.pipeline.frame_processing.diff import compute_diff
from subtitles_ocr.pipeline.frame_processing.mask import make_mask


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def _flat_rgb(h: int, w: int, rgb: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = rgb[0]
    img[..., 1] = rgb[1]
    img[..., 2] = rgb[2]
    return img


def _rect_glyph(h: int, w: int, *, top: int, left: int, height: int, width: int) -> np.ndarray:
    img = _flat_rgb(h, w, (40, 40, 40))
    img[top : top + height, left : left + width] = (240, 240, 240)
    return img


def test_compute_diff_identical_images_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    cfg = FrameProcessingConfig()
    diff = compute_diff(img, img.copy(), cfg)
    assert diff.dtype == np.float32
    assert diff.shape == (64, 64)
    assert float(np.max(np.abs(diff))) < 1e-3


def test_compute_diff_rectangle_glyph_is_nonzero_on_edges() -> None:
    raw = _flat_rgb(64, 64, (40, 40, 40))
    fansub = _rect_glyph(64, 64, top=20, left=20, height=24, width=24)
    cfg = FrameProcessingConfig()
    diff = compute_diff(fansub, raw, cfg)
    # On the glyph edges the diff must be substantially positive,
    # while far from the rectangle (e.g. corner) it must stay near zero.
    edge_value = float(np.max(diff[18:22, 18:46]))
    corner_value = float(np.max(diff[0:4, 0:4]))
    assert edge_value > 1.0
    assert corner_value < edge_value / 5.0


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------


def test_make_mask_returns_binary_uint8_with_seeds_above_t_high() -> None:
    cfg = FrameProcessingConfig(
        mask_smoothing_sigma=0.5,
        mask_t_high=0.3,
        mask_t_low=0.1,
        mask_area_min=20,
        mask_area_max=10_000,
        mask_dilation_iter=1,
    )
    diff = np.zeros((64, 64), dtype=np.float32)
    # A solid 20x20 high-intensity blob.
    diff[20:40, 20:40] = 1.0
    mask = make_mask(diff, cfg)
    assert mask.dtype == np.uint8
    assert mask.shape == (64, 64)
    assert set(np.unique(mask).tolist()).issubset({0, 255})
    # Mask covers the high-intensity blob (after dilation expands it slightly).
    assert mask[30, 30] == 255
    # Far away from the blob is background.
    assert mask[0, 0] == 0


def test_make_mask_filters_components_below_area_min() -> None:
    cfg = FrameProcessingConfig(
        mask_smoothing_sigma=0.1,
        mask_t_high=0.3,
        mask_t_low=0.1,
        mask_area_min=200,
        mask_area_max=10_000,
        mask_dilation_iter=0,
    )
    diff = np.zeros((64, 64), dtype=np.float32)
    # Single bright pixel — area 1, well below mask_area_min=200.
    diff[10, 10] = 1.0
    mask = make_mask(diff, cfg)
    assert mask[10, 10] == 0
    assert int(mask.sum()) == 0


def test_make_mask_hysteresis_keeps_low_pixels_connected_to_seed() -> None:
    cfg = FrameProcessingConfig(
        mask_smoothing_sigma=0.1,
        mask_t_high=0.8,
        mask_t_low=0.2,
        mask_area_min=1,
        mask_area_max=10_000,
        mask_dilation_iter=0,
    )
    diff = np.zeros((40, 40), dtype=np.float32)
    # A high-intensity seed connected to a tail of mid-intensity pixels.
    diff[10, 10] = 1.0
    diff[10, 11:20] = 0.5  # above T_low, below T_high — accepted via connection
    diff[30, 30] = 0.5  # above T_low but isolated — must NOT be accepted
    mask = make_mask(diff, cfg)
    assert mask[10, 10] == 255
    assert mask[10, 15] == 255
    assert mask[30, 30] == 0


def test_make_mask_dilation_grows_seed() -> None:
    base_cfg = dict(
        mask_smoothing_sigma=0.1,
        mask_t_high=0.3,
        mask_t_low=0.1,
        mask_area_min=1,
        mask_area_max=10_000,
    )
    diff = np.zeros((40, 40), dtype=np.float32)
    diff[20, 20] = 1.0

    no_dilation = make_mask(diff, FrameProcessingConfig(**base_cfg, mask_dilation_iter=0))
    one_dilation = make_mask(diff, FrameProcessingConfig(**base_cfg, mask_dilation_iter=1))
    assert int(one_dilation.sum()) > int(no_dilation.sum())


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


def test_compose_pixels_in_mask_keep_fansub_pixels_outside_mask_black() -> None:
    fansub = _flat_rgb(8, 8, (200, 150, 100))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    out = compose(fansub, mask)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    # Inside mask: original fansub pixels.
    assert tuple(out[3, 3].tolist()) == (200, 150, 100)
    # Outside mask: pure black RGB(0,0,0).
    assert tuple(out[0, 0].tolist()) == (0, 0, 0)
    assert tuple(out[7, 7].tolist()) == (0, 0, 0)


# ---------------------------------------------------------------------------
# iter_composed_frames
# ---------------------------------------------------------------------------


def test_default_frame_reader_can_decode_a_real_mp4(tmp_path: Path) -> None:
    """The default reader must decode codecs that opencv's bundled ffmpeg
    sometimes lacks (AV1, certain H.264 profiles). We generate a synthetic
    mp4 via PyAV and verify the reader returns a frame whose pixel matches
    the colour we encoded."""
    import av

    path = tmp_path / "synth.mp4"
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(5):
            arr = np.full((16, 16, 3), i * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    from subtitles_ocr.pipeline.frame_processing.iterator import _OpenCvFrameReader

    reader = _OpenCvFrameReader()
    frame = reader.read(path, 2)
    assert frame.shape == (16, 16, 3)
    assert frame.dtype == np.uint8
    # rgb pixel should be near (80, 80, 80) — allow generous tolerance for codec losses
    mean = float(frame.mean())
    assert 60.0 <= mean <= 100.0, f"unexpected mean intensity {mean}"


class _FakeFrameReader:
    """In-memory frame reader matching the FrameReader Protocol used by the iterator."""

    def __init__(self, frames_by_path: dict[Path, list[np.ndarray]]) -> None:
        self._frames = frames_by_path

    def read(self, path: Path, frame_idx: int) -> np.ndarray:
        return self._frames[path][frame_idx]


class _RecordingFrameReader:
    """Frame reader that records every (path, frame_idx) call."""

    def __init__(self, frames_by_path: dict[Path, list[np.ndarray]]) -> None:
        self._frames = frames_by_path
        self.calls: list[tuple[Path, int]] = []

    def read(self, path: Path, frame_idx: int) -> np.ndarray:
        self.calls.append((path, frame_idx))
        return self._frames[path][frame_idx]


def test_iter_composed_frames_reads_raw_from_conformed_workdir(mock_globals) -> None:
    """The iterator must read raw frames from the conformed mkv in the workdir
    (`01_conform/raw.mkv`), not from `globals.raw_path` which still points at
    the original bluray (potentially different resolution / codec)."""
    n_frames = 3
    conformed_raw_path = mock_globals.workdir / "01_conform" / "raw.mkv"
    rng = np.random.default_rng(0)
    fansub_frames = [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(n_frames)]
    raw_frames = [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(n_frames)]
    reader = _RecordingFrameReader(
        {mock_globals.hardsub_path: fansub_frames, conformed_raw_path: raw_frames}
    )

    alignment = AlignmentResult(
        fansub_total_frames=n_frames,
        raw_total_frames=n_frames,
        method_used="audio_only",
        aligned_ratio=1.0,
        orphan_ratio=0.0,
        user_skipped_ratio=0.0,
        segments=[
            AlignmentSegment(
                fansub_frame_start=0,
                fansub_frame_end=n_frames,
                raw_frame_start=0,
                raw_frame_end=n_frames,
                offset_frames=0,
                status="ALIGNED",
                confidence_avg=None,
            )
        ],
        warnings=[],
    )

    list(iter_composed_frames(mock_globals, alignment, FrameProcessingConfig(), frame_reader=reader))

    raw_call_paths = {p for p, _ in reader.calls if p != mock_globals.hardsub_path}
    assert raw_call_paths == {conformed_raw_path}, (
        f"raw frames must be read from {conformed_raw_path}, got {raw_call_paths}"
    )


def test_iter_composed_frames_yields_one_per_aligned_fansub_frame(mock_globals) -> None:
    n_frames = 5
    raw_path = mock_globals.workdir / "01_conform" / "raw.mkv"
    hardsub_path = mock_globals.hardsub_path
    rng = np.random.default_rng(42)
    fansub_frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(n_frames)]
    raw_frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(n_frames)]
    reader = _FakeFrameReader({hardsub_path: fansub_frames, raw_path: raw_frames})

    alignment = AlignmentResult(
        fansub_total_frames=n_frames,
        raw_total_frames=n_frames,
        method_used="phash_only",
        aligned_ratio=1.0,
        orphan_ratio=0.0,
        user_skipped_ratio=0.0,
        segments=[
            AlignmentSegment(
                fansub_frame_start=0,
                fansub_frame_end=n_frames,
                raw_frame_start=0,
                raw_frame_end=n_frames,
                offset_frames=0,
                status="ALIGNED",
                confidence_avg=0.9,
            )
        ],
        warnings=[],
    )

    cfg = FrameProcessingConfig()
    out = list(
        iter_composed_frames(
            mock_globals,
            alignment,
            cfg,
            frame_reader=reader,
        )
    )
    assert [cf.fansub_frame_idx for cf in out] == [0, 1, 2, 3, 4]
    for cf in out:
        assert isinstance(cf, ComposedFrame)
        assert cf.image.shape == (32, 32, 3)
        assert cf.image.dtype == np.uint8


def test_iter_composed_frames_skips_orphan_segments(mock_globals) -> None:
    n_frames = 6
    raw_path = mock_globals.workdir / "01_conform" / "raw.mkv"
    hardsub_path = mock_globals.hardsub_path
    rng = np.random.default_rng(1)
    fansub_frames = [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(n_frames)]
    raw_frames = [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(n_frames)]
    reader = _FakeFrameReader({hardsub_path: fansub_frames, raw_path: raw_frames})

    alignment = AlignmentResult(
        fansub_total_frames=n_frames,
        raw_total_frames=n_frames,
        method_used="phash_only",
        aligned_ratio=4 / 6,
        orphan_ratio=2 / 6,
        user_skipped_ratio=0.0,
        segments=[
            AlignmentSegment(
                fansub_frame_start=0,
                fansub_frame_end=2,
                raw_frame_start=0,
                raw_frame_end=2,
                offset_frames=0,
                status="ALIGNED",
                confidence_avg=0.9,
            ),
            AlignmentSegment(
                fansub_frame_start=2,
                fansub_frame_end=4,
                raw_frame_start=None,
                raw_frame_end=None,
                offset_frames=None,
                status="ORPHAN",
                confidence_avg=None,
            ),
            AlignmentSegment(
                fansub_frame_start=4,
                fansub_frame_end=6,
                raw_frame_start=4,
                raw_frame_end=6,
                offset_frames=0,
                status="ALIGNED",
                confidence_avg=0.9,
            ),
        ],
        warnings=[],
    )

    cfg = FrameProcessingConfig()
    out = list(
        iter_composed_frames(
            mock_globals,
            alignment,
            cfg,
            frame_reader=reader,
        )
    )
    assert [cf.fansub_frame_idx for cf in out] == [0, 1, 4, 5]


def test_iter_composed_frames_respects_start_at_fansub_idx(mock_globals) -> None:
    n_frames = 4
    raw_path = mock_globals.workdir / "01_conform" / "raw.mkv"
    hardsub_path = mock_globals.hardsub_path
    fansub_frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n_frames)]
    raw_frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n_frames)]
    reader = _FakeFrameReader({hardsub_path: fansub_frames, raw_path: raw_frames})

    alignment = AlignmentResult(
        fansub_total_frames=n_frames,
        raw_total_frames=n_frames,
        method_used="phash_only",
        aligned_ratio=1.0,
        orphan_ratio=0.0,
        user_skipped_ratio=0.0,
        segments=[
            AlignmentSegment(
                fansub_frame_start=0,
                fansub_frame_end=n_frames,
                raw_frame_start=0,
                raw_frame_end=n_frames,
                offset_frames=0,
                status="ALIGNED",
                confidence_avg=1.0,
            )
        ],
        warnings=[],
    )

    out = list(
        iter_composed_frames(
            mock_globals,
            alignment,
            FrameProcessingConfig(),
            start_at_fansub_idx=2,
            frame_reader=reader,
        )
    )
    assert [cf.fansub_frame_idx for cf in out] == [2, 3]


def test_iter_composed_frames_debug_images_writes_pngs(tmp_path: Path) -> None:
    from fractions import Fraction

    from subtitles_ocr.config import PipelineGlobals

    # Pre-create the debug subdirs the iterator writes to.
    for d in ("01_conform", "02_alignment", "03_diff", "04_mask", "05_compose", "06_ocr"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    hardsub_path = tmp_path / "fake_hardsub.avi"
    raw_path = tmp_path / "fake_raw.mkv"
    conformed_raw_path = tmp_path / "01_conform" / "raw.mkv"
    globals_ = PipelineGlobals(
        workdir=tmp_path,
        hardsub_path=hardsub_path,
        raw_path=raw_path,
        out_path=tmp_path / "out.ass",
        fps=Fraction(24, 1),
        fansub_width=32,
        fansub_height=32,
        fansub_total_frames=2,
        debug_images=True,
    )
    rng = np.random.default_rng(7)
    fansub_frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
    raw_frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(2)]
    reader = _FakeFrameReader({hardsub_path: fansub_frames, conformed_raw_path: raw_frames})

    alignment = AlignmentResult(
        fansub_total_frames=2,
        raw_total_frames=2,
        method_used="phash_only",
        aligned_ratio=1.0,
        orphan_ratio=0.0,
        user_skipped_ratio=0.0,
        segments=[
            AlignmentSegment(
                fansub_frame_start=0,
                fansub_frame_end=2,
                raw_frame_start=0,
                raw_frame_end=2,
                offset_frames=0,
                status="ALIGNED",
                confidence_avg=1.0,
            )
        ],
        warnings=[],
    )

    out = list(
        iter_composed_frames(
            globals_,
            alignment,
            FrameProcessingConfig(),
            frame_reader=reader,
        )
    )
    assert len(out) == 2
    assert (tmp_path / "03_diff" / "debug" / "00000000.png").exists()
    assert (tmp_path / "04_mask" / "frames" / "00000000.png").exists()
    assert (tmp_path / "05_compose" / "frames" / "00000000.png").exists()
