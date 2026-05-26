"""Tests for Stage 9 — ColorStage (per-event interior/outline color extraction).

Reference: ADR-0002 §3 Stage 8 (renumbered Stage 9 by ADR-0003 §4.3).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from subtitles_ocr.config import ColorConfig
from subtitles_ocr.pipeline.animation import AnimatedEvent, AnimationAnalysisResult
from subtitles_ocr.pipeline.color import ColorStage


# ----------------- Fakes -----------------


class FakeFrameReader:
    """In-memory frame source keyed by (path, frame_idx) → RGB uint8 array."""

    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self.frames = frames
        self.calls: list[int] = []

    def read(self, path: Path, frame_idx: int) -> np.ndarray:
        self.calls.append(frame_idx)
        return self.frames[frame_idx]


# ----------------- Helpers -----------------


def _write_animation_json(workdir: Path, events: list[AnimatedEvent]) -> None:
    result = AnimationAnalysisResult(events=events, stats={})
    out = workdir / "08_animation" / "animation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json())


def _make_white_on_black_frame(
    width: int = 320,
    height: int = 160,
    text_box: tuple[int, int, int, int] = (20, 30, 300, 130),
    text: str = "Hi",
) -> np.ndarray:
    """Frame mimicking a real hardsubbed glyph: cv2.putText with a thick black
    outline pass followed by a thinner white fill pass on a dark background.

    The text is drawn anti-aliased — this AA is what the ColorStage relies on to
    pick up the outline color from the eroded ring of the fill mask.
    """
    img = np.full((height, width, 3), 16, dtype=np.uint8)
    # Place text near the center of text_box
    org = (text_box[0] + 10, text_box[3] - 25)
    # Thick black stroke (outline)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10, cv2.LINE_AA)
    # White fill on top, thinner stroke
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def _make_gradient_interior_frame(
    width: int = 320,
    height: int = 160,
    text_box: tuple[int, int, int, int] = (20, 30, 300, 130),
    outline_thickness: int = 8,
) -> np.ndarray:
    """A bright rainbow-filled rectangle on a dark background. The filled rectangle
    is large enough that the eroded interior pool contains the entire rainbow,
    yielding very high hue variance → style_supported should be False.
    """
    img = np.full((height, width, 3), 16, dtype=np.uint8)
    x0, y0, x1, y1 = text_box
    # The outline ring is not relevant here — the test exercises the interior
    # hue variance gate.
    inner_x0 = x0 + outline_thickness
    inner_x1 = x1 - outline_thickness
    inner_y0 = y0 + outline_thickness
    inner_y1 = y1 - outline_thickness
    w = inner_x1 - inner_x0
    h = inner_y1 - inner_y0
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        hsv[:, i, 0] = int(i * 179 / max(1, w - 1))
        hsv[:, i, 1] = 255
        hsv[:, i, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img[inner_y0:inner_y1, inner_x0:inner_x1] = rgb
    return img


def _axis_aligned_quad(text_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """TL, TR, BR, BL in image coordinates."""
    x0, y0, x1, y1 = text_box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ----------------- Fixtures -----------------


@pytest.fixture
def reader_factory():
    def _factory(frames: dict[int, np.ndarray]) -> FakeFrameReader:
        return FakeFrameReader(frames)

    return _factory


# ----------------- Tests -----------------


def test_white_fill_black_outline_extracted(tmp_workdir, mock_globals, reader_factory):
    """Synthetic white-filled glyph with black outline → fill=white, outline=black."""
    text_box = (20, 30, 300, 130)
    quad = _axis_aligned_quad(text_box)
    # 5 identical frames so the temporal median is trivial
    frames = {i: _make_white_on_black_frame(text_box=text_box) for i in range(5)}
    reader = reader_factory(frames)

    event = AnimatedEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=5,
        raw_ocr_texts=["hi"] * 5,
        raw_ocr_confidences=[0.9] * 5,
        quads_per_frame={i: quad for i in range(5)},
        quad_median=quad,
        member_frame_indices=list(range(5)),
        motion=None,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    _write_animation_json(tmp_workdir, [event])

    stage = ColorStage(frame_reader=reader)
    result = stage.run(mock_globals, ColorConfig())

    assert len(result.events) == 1
    ec = result.events[0]
    assert ec.event_id == 0
    assert ec.style_supported is True
    # White fill (allow some quantization tolerance)
    assert ec.fill_color is not None
    fr, fg, fb = ec.fill_color
    assert fr >= 200 and fg >= 200 and fb >= 200
    # Black outline
    assert ec.outline_color is not None
    orr, og, ob = ec.outline_color
    assert orr <= 60 and og <= 60 and ob <= 60


def test_gradient_interior_marks_unsupported(tmp_workdir, mock_globals, reader_factory):
    """High hue variance in interior pool → style_supported=False."""
    text_box = (20, 30, 300, 130)
    quad = _axis_aligned_quad(text_box)
    frames = {i: _make_gradient_interior_frame(text_box=text_box) for i in range(5)}
    reader = reader_factory(frames)

    event = AnimatedEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=5,
        raw_ocr_texts=["x"] * 5,
        raw_ocr_confidences=[0.9] * 5,
        quads_per_frame={i: quad for i in range(5)},
        quad_median=quad,
        member_frame_indices=list(range(5)),
        motion=None,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    _write_animation_json(tmp_workdir, [event])

    stage = ColorStage(frame_reader=reader)
    result = stage.run(mock_globals, ColorConfig())

    ec = result.events[0]
    assert ec.style_supported is False
    assert ec.fill_color is None
    assert ec.outline_color is None


def test_fade_in_excludes_initial_frames(tmp_workdir, mock_globals, reader_factory):
    """fade_in_ms=200 at 24fps excludes first 5 frames (round(200*24/1000) = 5)."""
    text_box = (20, 30, 300, 130)
    quad = _axis_aligned_quad(text_box)
    # Frames 0..4 are *junk* (red) — must be excluded so result reflects white-on-black
    junk = np.full((160, 320, 3), 64, dtype=np.uint8)
    junk[:, :, 0] = 255  # noisy red noise that would skew Otsu
    frames: dict[int, np.ndarray] = {i: junk.copy() for i in range(5)}
    for i in range(5, 12):
        frames[i] = _make_white_on_black_frame(text_box=text_box)
    reader = reader_factory(frames)

    event = AnimatedEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=12,
        raw_ocr_texts=["t"] * 12,
        raw_ocr_confidences=[0.9] * 12,
        quads_per_frame={i: quad for i in range(12)},
        quad_median=quad,
        member_frame_indices=list(range(12)),
        motion=None,
        fade_in_ms=200,  # ms_to_frame(200, 24fps) = 5
        fade_out_ms=0,
    )
    _write_animation_json(tmp_workdir, [event])

    stage = ColorStage(frame_reader=reader)
    result = stage.run(mock_globals, ColorConfig())

    # The frames asked for must be ONLY the non-fade members (5..11)
    assert sorted(set(reader.calls)) == list(range(5, 12))
    ec = result.events[0]
    assert ec.style_supported is True
    assert ec.fill_color is not None and ec.fill_color[0] >= 200


def test_fewer_than_three_remaining_frames_marks_unsupported(
    tmp_workdir, mock_globals, reader_factory
):
    """Exclusion that leaves < 3 frames → style_supported=False, no read attempts."""
    text_box = (20, 30, 300, 130)
    quad = _axis_aligned_quad(text_box)
    frames = {i: _make_white_on_black_frame(text_box=text_box) for i in range(2)}
    reader = reader_factory(frames)

    # 2 member frames, fade_in_ms=0 → only 2 remaining (< 3)
    event = AnimatedEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=2,
        raw_ocr_texts=["t"] * 2,
        raw_ocr_confidences=[0.9] * 2,
        quads_per_frame={i: quad for i in range(2)},
        quad_median=quad,
        member_frame_indices=[0, 1],
        motion=None,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    _write_animation_json(tmp_workdir, [event])

    stage = ColorStage(frame_reader=reader)
    result = stage.run(mock_globals, ColorConfig())

    ec = result.events[0]
    assert ec.style_supported is False
    assert ec.fill_color is None
    assert ec.outline_color is None


def test_rotated_quad_is_rectified(tmp_workdir, mock_globals, reader_factory):
    """A glyph drawn rotated in the source → upright in the canonical crop, color extracted."""
    width, height = 320, 320
    img = np.full((height, width, 3), 16, dtype=np.uint8)
    # The text spans roughly this bounding box (axis-aligned), then we rotate.
    text_box = (40, 130, 280, 230)
    org = (text_box[0] + 10, text_box[3] - 25)
    cv2.putText(img, "Hi", org, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(img, "Hi", org, cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3, cv2.LINE_AA)

    # Rotate the whole image by 30° around its center
    cx, cy = width / 2, height / 2
    angle = 30.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height), borderValue=(128, 128, 128))

    # Rotate the quad vertices the same way
    src_quad = np.array(_axis_aligned_quad(text_box), dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    src_h = np.hstack([src_quad, ones])
    rotated_quad_f = (M @ src_h.T).T
    rotated_quad = [
        (int(round(p[0])), int(round(p[1]))) for p in rotated_quad_f
    ]

    frames = {i: rotated for i in range(4)}
    reader = reader_factory(frames)

    event = AnimatedEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=4,
        raw_ocr_texts=["t"] * 4,
        raw_ocr_confidences=[0.9] * 4,
        quads_per_frame={i: rotated_quad for i in range(4)},
        quad_median=rotated_quad,
        member_frame_indices=list(range(4)),
        motion=None,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    _write_animation_json(tmp_workdir, [event])

    stage = ColorStage(frame_reader=reader)
    result = stage.run(mock_globals, ColorConfig())

    ec = result.events[0]
    assert ec.style_supported is True
    assert ec.fill_color is not None
    # After perspective warp the AA gets thicker; the recovered fill is still
    # much brighter than the outline, but not pristine 255.
    assert ec.fill_color[0] >= 180 and ec.fill_color[1] >= 180 and ec.fill_color[2] >= 180
    assert ec.outline_color is not None
    # Outline should be clearly darker than the fill (separation > 80 on R).
    assert ec.fill_color[0] - ec.outline_color[0] > 80


def test_resume_skips_when_sidecar_matches(tmp_workdir, mock_globals, reader_factory):
    """If colors.json + meta exist and meta matches, skip recomputation."""
    text_box = (20, 30, 300, 130)
    quad = _axis_aligned_quad(text_box)
    frames = {i: _make_white_on_black_frame(text_box=text_box) for i in range(4)}
    reader = reader_factory(frames)

    event = AnimatedEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=4,
        raw_ocr_texts=["t"] * 4,
        raw_ocr_confidences=[0.9] * 4,
        quads_per_frame={i: quad for i in range(4)},
        quad_median=quad,
        member_frame_indices=list(range(4)),
        motion=None,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    _write_animation_json(tmp_workdir, [event])

    stage = ColorStage(frame_reader=reader)
    first = stage.run(mock_globals, ColorConfig())
    reads_first = len(reader.calls)
    assert reads_first > 0

    # Second invocation must short-circuit and reuse cached result
    second = stage.run(mock_globals, ColorConfig())
    assert second.model_dump() == first.model_dump()
    assert len(reader.calls) == reads_first  # no new reads
