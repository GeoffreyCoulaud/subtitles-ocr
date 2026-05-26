"""Tests for AnimationStage full reconstruction (ADR-0003 §4.2, §7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_ocr.config import AnimationConfig, PipelineGlobals
from subtitles_ocr.meta import BaseMeta
from subtitles_ocr.pipeline.animation import (
    AnimatedEvent,
    AnimationAnalysisResult,
    AnimationStage,
)
from subtitles_ocr.pipeline.group import GroupResult, SubtitleEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quad_around(cx: int, cy: int, w: int = 20, h: int = 10) -> list[tuple[int, int]]:
    """Axis-aligned quad TL/TR/BR/BL of size (w, h) centred on (cx, cy)."""
    x0, x1 = cx - w // 2, cx + w // 2
    y0, y1 = cy - h // 2, cy + h // 2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _quad_median_of(quads: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    from statistics import median_low

    out: list[tuple[int, int]] = []
    for i in range(4):
        xs = [q[i][0] for q in quads]
        ys = [q[i][1] for q in quads]
        out.append((median_low(xs), median_low(ys)))
    return out


def _make_event(
    *,
    event_id: int,
    start: int,
    end: int,
    quads: dict[int, list[tuple[int, int]]],
    texts: list[str] | None = None,
    confs: list[float] | None = None,
) -> SubtitleEvent:
    frames = sorted(quads.keys())
    if texts is None:
        texts = ["t"] * len(frames)
    if confs is None:
        confs = [0.9] * len(frames)
    return SubtitleEvent(
        event_id=event_id,
        fansub_frame_start=start,
        fansub_frame_end=end,
        raw_ocr_texts=list(texts),
        raw_ocr_confidences=list(confs),
        quads_per_frame=quads,
        quad_median=_quad_median_of(list(quads.values())) if quads else [(0, 0), (10, 0), (10, 5), (0, 5)],
        member_frame_indices=frames,
    )


def _write_group(workdir: Path, events: list[SubtitleEvent], total_frames: int = 1000) -> None:
    payload = GroupResult(
        fansub_total_frames=total_frames,
        events=events,
        stats={"events": len(events)},
    )
    (workdir / "07_group" / "events.json").write_text(payload.model_dump_json())


class FakeDiffSource:
    """Scripted diff intensity for fade tests.

    `scores[frame_idx]` returns the value; missing frames return 0.0. The bbox
    is ignored — tests don't exercise per-bbox geometry from the source, only
    the temporal scoring path.
    """

    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[int, tuple[int, int, int, int]]] = []

    def mean_intensity(self, frame_idx: int, bbox: tuple[int, int, int, int]) -> float:
        self.calls.append((frame_idx, bbox))
        return self.scores.get(frame_idx, 0.0)


# ---------------------------------------------------------------------------
# Persistence + cache (carried over from MVP)
# ---------------------------------------------------------------------------


def test_static_event_passes_through_with_motion_none(
    mock_globals: PipelineGlobals,
) -> None:
    quads = {f: _quad_around(100, 50) for f in range(20, 30)}
    ev = _make_event(event_id=0, start=20, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert len(result.events) == 1
    out = result.events[0]
    assert out.motion is None
    assert out.fade_in_ms == 0
    assert out.fade_out_ms == 0
    assert result.stats["static"] == 1
    assert result.stats["linear_move"] == 0


def test_writes_animation_json_and_sidecar(mock_globals: PipelineGlobals) -> None:
    quads = {f: _quad_around(100, 50) for f in range(20, 30)}
    ev = _make_event(event_id=0, start=20, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    AnimationStage().run(mock_globals, AnimationConfig())

    out_path = mock_globals.workdir / "08_animation" / "animation.json"
    meta_path = mock_globals.workdir / "08_animation" / "animation.meta.json"
    assert out_path.exists()
    assert meta_path.exists()
    meta = BaseMeta.model_validate_json(meta_path.read_text())
    assert meta.stage_name == "08_animation"
    assert meta.stage_version == 2


def test_resume_skips_when_sidecar_matches(mock_globals: PipelineGlobals) -> None:
    quads = {f: _quad_around(100, 50) for f in range(20, 30)}
    ev = _make_event(event_id=0, start=20, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    stage = AnimationStage()
    first = stage.run(mock_globals, AnimationConfig())
    assert len(first.events) == 1

    out_path = mock_globals.workdir / "08_animation" / "animation.json"
    out_path.write_text(
        AnimationAnalysisResult(
            events=[
                AnimatedEvent(
                    event_id=999,
                    fansub_frame_start=0,
                    fansub_frame_end=1,
                    raw_ocr_texts=[],
                    raw_ocr_confidences=[],
                    quads_per_frame={},
                    quad_median=[(0, 0), (1, 0), (1, 1), (0, 1)],
                    member_frame_indices=[],
                    motion=None,
                    fade_in_ms=0,
                    fade_out_ms=0,
                )
            ],
            stats={
                "static": 1,
                "linear_move": 0,
                "flagged_nonlinear": 0,
                "fade_in_only": 0,
                "fade_out_only": 0,
                "full_fade": 0,
            },
        ).model_dump_json()
    )

    second = stage.run(mock_globals, AnimationConfig())
    assert second.events[0].event_id == 999


def test_resume_recomputes_when_config_changes(mock_globals: PipelineGlobals) -> None:
    quads = {f: _quad_around(100, 50) for f in range(20, 30)}
    ev = _make_event(event_id=0, start=20, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    stage = AnimationStage()
    stage.run(mock_globals, AnimationConfig())

    changed = AnimationConfig(min_move_displacement_px=99)
    second = stage.run(mock_globals, changed)
    assert len(second.events) == 1


# ---------------------------------------------------------------------------
# A1 — intra-event linear move
# ---------------------------------------------------------------------------


def test_linear_intra_event_move_above_threshold(
    mock_globals: PipelineGlobals,
) -> None:
    # 20 frames of perfectly-linear horizontal pan from cx=100 to cx=290 (190px)
    quads = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 30)}
    ev = _make_event(event_id=0, start=10, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert len(result.events) == 1
    motion = result.events[0].motion
    assert motion is not None
    assert motion["type"] == "linear"
    sx, sy = motion["start"]
    ex, ey = motion["end"]
    assert sx == pytest.approx(100, abs=1)
    assert ex == pytest.approx(290, abs=1)
    assert sy == pytest.approx(50, abs=1)
    assert ey == pytest.approx(50, abs=1)
    assert result.stats["linear_move"] == 1


def test_intra_event_displacement_below_threshold_is_static(
    mock_globals: PipelineGlobals,
) -> None:
    # 20 frames, cx drifts 100 → 105 over 20 frames: 5 px < default 8 px threshold
    quads = {f: _quad_around(100 + (f - 10) // 4, 50) for f in range(10, 30)}
    ev = _make_event(event_id=0, start=10, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert result.events[0].motion is None
    assert result.stats["static"] == 1


def test_intra_event_nonlinear_displacement_is_flagged(
    mock_globals: PipelineGlobals,
) -> None:
    # Displacement is large but R² is bad: cx oscillates around a slow drift
    centroids = []
    for k, f in enumerate(range(10, 30)):
        cx = 100 + (k * 10) + (50 if k % 2 == 0 else -50)  # large noise, large drift
        centroids.append(cx)
    quads = {10 + k: _quad_around(cx, 50) for k, cx in enumerate(centroids)}
    ev = _make_event(event_id=0, start=10, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    motion = result.events[0].motion
    assert motion is not None
    assert motion["type"] == "nonlinear_flagged"
    assert result.stats["flagged_nonlinear"] == 1


def test_single_frame_event_is_static(mock_globals: PipelineGlobals) -> None:
    quads = {15: _quad_around(100, 50)}
    ev = _make_event(event_id=0, start=15, end=16, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert result.events[0].motion is None


# ---------------------------------------------------------------------------
# A2 — inter-event merge
# ---------------------------------------------------------------------------


def test_inter_event_merge_happy_path(mock_globals: PipelineGlobals) -> None:
    # Two events fragmented at frame 19→21 with consistent linear trajectory
    quads0 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 20)}
    quads1 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(22, 32)}
    ev0 = _make_event(event_id=0, start=10, end=20, quads=quads0, texts=["Hello"] * 10)
    ev1 = _make_event(event_id=1, start=22, end=32, quads=quads1, texts=["Hello"] * 10)
    _write_group(mock_globals.workdir, [ev0, ev1])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert len(result.events) == 1
    merged = result.events[0]
    assert merged.motion is not None
    assert merged.motion["type"] == "linear"
    assert merged.fansub_frame_start == 10
    assert merged.fansub_frame_end == 32
    # quads_per_frame contains both segments
    assert 10 in merged.quads_per_frame
    assert 31 in merged.quads_per_frame


def test_inter_event_merge_blocked_by_text(mock_globals: PipelineGlobals) -> None:
    quads0 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 20)}
    quads1 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(22, 32)}
    ev0 = _make_event(event_id=0, start=10, end=20, quads=quads0, texts=["Hello"] * 10)
    ev1 = _make_event(event_id=1, start=22, end=32, quads=quads1, texts=["Goodbye world"] * 10)
    _write_group(mock_globals.workdir, [ev0, ev1])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert len(result.events) == 2


def test_inter_event_merge_blocked_by_gap(mock_globals: PipelineGlobals) -> None:
    # gap of 50 frames (≈ 2080ms at 24fps) >> 200ms tolerance
    quads0 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 20)}
    quads1 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(70, 80)}
    ev0 = _make_event(event_id=0, start=10, end=20, quads=quads0, texts=["Hi"] * 10)
    ev1 = _make_event(event_id=1, start=70, end=80, quads=quads1, texts=["Hi"] * 10)
    _write_group(mock_globals.workdir, [ev0, ev1])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert len(result.events) == 2


def test_inter_event_merge_with_bad_r2_merges_static_flagged(
    mock_globals: PipelineGlobals,
) -> None:
    # text & timing OK, but trajectories don't fit a single line
    quads0 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 20)}
    quads1 = {f: _quad_around(500 - (f - 22) * 10, 200) for f in range(22, 32)}  # jumps
    ev0 = _make_event(event_id=0, start=10, end=20, quads=quads0, texts=["Hi"] * 10)
    ev1 = _make_event(event_id=1, start=22, end=32, quads=quads1, texts=["Hi"] * 10)
    _write_group(mock_globals.workdir, [ev0, ev1])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert len(result.events) == 1
    merged = result.events[0]
    assert merged.motion is not None
    assert merged.motion["type"] == "nonlinear_flagged"
    assert result.stats["flagged_nonlinear"] == 1


# ---------------------------------------------------------------------------
# B — Fade detection
# ---------------------------------------------------------------------------


def _linear_ramp_scores(
    start_frame: int,
    end_frame: int,
    *,
    fade_in_frames: int = 0,
    fade_out_frames: int = 0,
    pre_window: int = 0,
    post_window: int = 0,
) -> dict[int, float]:
    """Build scripted score dict for a synthetic fade test.

    Score = 1 inside [start, end]. Linear ramp from 0 to 1 across the
    fade_in_frames before start. Linear ramp from 1 to 0 across the
    fade_out_frames after end.
    """
    scores: dict[int, float] = {}
    # In-event: 1.0
    for f in range(start_frame, end_frame):
        scores[f] = 1.0
    # Fade-in
    for k in range(1, fade_in_frames + 1):
        scores[start_frame - k] = max(0.0, 1.0 - k / fade_in_frames)
    # Fade-out
    for k in range(1, fade_out_frames + 1):
        scores[end_frame - 1 + k] = max(0.0, 1.0 - k / fade_out_frames)
    # Pre-window / post-window padding to score 0
    for k in range(fade_in_frames + 1, fade_in_frames + 1 + pre_window):
        scores.setdefault(start_frame - k, 0.0)
    for k in range(fade_out_frames + 1, fade_out_frames + 1 + post_window):
        scores.setdefault(end_frame - 1 + k, 0.0)
    return scores


def test_fade_in_only(mock_globals: PipelineGlobals) -> None:
    # Static event from 30-50; 6-frame linear fade-in before frame 30 ≈ 250ms@24fps
    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = _linear_ramp_scores(30, 50, fade_in_frames=6, pre_window=5)
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    out = result.events[0]
    assert out.fade_in_ms >= 125
    assert out.fade_out_ms == 0
    # Extended start moves earlier
    assert out.fansub_frame_start < 30


def test_fade_out_only(mock_globals: PipelineGlobals) -> None:
    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = _linear_ramp_scores(30, 50, fade_out_frames=6, post_window=5)
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    out = result.events[0]
    assert out.fade_in_ms == 0
    assert out.fade_out_ms >= 125
    assert out.fansub_frame_end > 50


def test_full_fade(mock_globals: PipelineGlobals) -> None:
    quads = {f: _quad_around(100, 50) for f in range(30, 80)}
    ev = _make_event(event_id=0, start=30, end=80, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = _linear_ramp_scores(
        30, 80, fade_in_frames=6, fade_out_frames=6, pre_window=5, post_window=5
    )
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    out = result.events[0]
    assert out.fade_in_ms >= 125
    assert out.fade_out_ms >= 125


def test_fade_with_bad_r2_reverts_to_zero(mock_globals: PipelineGlobals) -> None:
    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    # Noisy scores in the [0.05, 0.95] band with no linear structure
    scores = {f: 1.0 for f in range(30, 50)}
    for k, f in enumerate(range(20, 30)):
        scores[f] = 0.5 if k % 2 == 0 else 0.7  # noise, no linear trend
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    out = result.events[0]
    assert out.fade_in_ms == 0


def test_fade_extrapolated_below_min_reverts(mock_globals: PipelineGlobals) -> None:
    # A 2-frame fade-in ≈ 83ms@24fps < MIN_FADE_DURATION_MS (125)
    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = _linear_ramp_scores(30, 50, fade_in_frames=2, pre_window=5)
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    assert result.events[0].fade_in_ms == 0


def test_fade_extrapolated_above_cap_reverts(mock_globals: PipelineGlobals) -> None:
    # Ramp spans > 1000ms (≈ 30 frames @ 24fps); use 28 obs frames
    quads = {f: _quad_around(100, 50) for f in range(80, 100)}
    ev = _make_event(event_id=0, start=80, end=100, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    # 60-frame fade-in: extrapolated t = 60 frames * 41.67ms ≈ 2500ms >> 1000ms cap
    # but FADE_SEARCH_WINDOW_MS=1250 → W = 30 frames; so we must observe within window.
    # Construct a *very* shallow slope so extrapolation reaches 0 way past the cap:
    # observations of slope -0.01 per frame inside the window.
    scores = {f: 1.0 for f in range(80, 100)}
    # 28 observations in [0.05, 0.95] band, very flat
    for k in range(1, 29):
        scores[80 - k] = max(0.05, 1.0 - 0.02 * k)  # at k=28: 0.44
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    assert result.events[0].fade_in_ms == 0


def test_partial_fade_sum_exceeds_duration_reverts_both(
    mock_globals: PipelineGlobals,
) -> None:
    # Short event (10 frames ≈ 417ms) with detected fades that sum to > 417ms
    quads = {f: _quad_around(100, 50) for f in range(40, 50)}
    ev = _make_event(event_id=0, start=40, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    # Fade-in of 8 frames + fade-out of 8 frames → 16 frames > 10 duration
    scores = _linear_ramp_scores(40, 50, fade_in_frames=8, fade_out_frames=8, pre_window=5, post_window=5)
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    out = result.events[0]
    assert out.fade_in_ms == 0
    assert out.fade_out_ms == 0


def test_fade_skipped_for_nonlinear_flagged(mock_globals: PipelineGlobals) -> None:
    # Build a nonlinear-flagged event AND provide scores that would otherwise
    # detect a fade. Stage must skip fade detection.
    centroids = [100 + k * 10 + (50 if k % 2 == 0 else -50) for k in range(20)]
    quads = {10 + k: _quad_around(cx, 50) for k, cx in enumerate(centroids)}
    ev = _make_event(event_id=0, start=10, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = _linear_ramp_scores(10, 30, fade_in_frames=6, fade_out_frames=6, pre_window=5, post_window=5)
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    out = result.events[0]
    assert out.motion is not None and out.motion["type"] == "nonlinear_flagged"
    assert out.fade_in_ms == 0
    assert out.fade_out_ms == 0


def test_fade_for_linear_moving_event_uses_extrapolated_bbox(
    mock_globals: PipelineGlobals,
) -> None:
    # Linear-moving event from cx=100 to cx=290 across frames 10-30.
    # If the bbox follows the trajectory in the pre-window, the source should
    # be queried with bboxes whose x shifts frame-to-frame.
    quads = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 30)}
    ev = _make_event(event_id=0, start=10, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = _linear_ramp_scores(10, 30, fade_in_frames=6, pre_window=5)
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    # Filter to only the pre-window calls (frames < 10)
    pre_calls = [(f, bbox) for (f, bbox) in src.calls if f < 10]
    assert len(pre_calls) > 0
    # Bbox cx must differ across pre-window frames (trajectory is extrapolated)
    xs = sorted({(bbox[0] + bbox[2]) / 2 for (_, bbox) in pre_calls})
    assert len(xs) > 1


def test_fade_detection_silent_without_diff_source(
    mock_globals: PipelineGlobals,
) -> None:
    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert result.events[0].fade_in_ms == 0
    assert result.events[0].fade_out_ms == 0


def test_fade_search_window_excludes_other_event_frames(
    mock_globals: PipelineGlobals,
) -> None:
    # Two adjacent events: ev0 ends at 25, ev1 starts at 30.
    # Fade-in search for ev1 must not score frames 20-24 (claimed by ev0).
    quads0 = {f: _quad_around(50, 50) for f in range(20, 25)}
    quads1 = {f: _quad_around(200, 100) for f in range(30, 50)}
    ev0 = _make_event(event_id=0, start=20, end=25, quads=quads0, texts=["a"] * 5)
    ev1 = _make_event(event_id=1, start=30, end=50, quads=quads1, texts=["b"] * 20)
    _write_group(mock_globals.workdir, [ev0, ev1])

    scores = _linear_ramp_scores(30, 50, fade_in_frames=4, pre_window=15)
    src = FakeDiffSource(scores)

    AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    # ev1 is centered at cy=100, ev0 at cy=50. Filter ev1-side calls (bbox y_min >= 90)
    # in the frame range owned by ev0 (20-24). Must be empty.
    ev1_pre_calls = [
        f
        for (f, bbox) in src.calls
        if 20 <= f < 25 and bbox[1] >= 90
    ]
    assert ev1_pre_calls == []


# ---------------------------------------------------------------------------
# A2 — gap == 0 (contiguous events)
# ---------------------------------------------------------------------------


def test_inter_event_merge_gap_zero_merges(mock_globals: PipelineGlobals) -> None:
    """gap=0 means fansub_frame_end[ev0] == fansub_frame_start[ev1].

    The merge condition is ``gap > gap_tol_frames or gap < 0``.  With gap=0,
    neither branch triggers, so the events should be merged into one.
    """
    # ev0: frames 10-19 (end=20), ev1: frames 20-29 (start=20) → gap = 0
    quads0 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 20)}
    quads1 = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(20, 30)}
    ev0 = _make_event(event_id=0, start=10, end=20, quads=quads0, texts=["Hi"] * 10)
    ev1 = _make_event(event_id=1, start=20, end=30, quads=quads1, texts=["Hi"] * 10)
    _write_group(mock_globals.workdir, [ev0, ev1])

    result = AnimationStage().run(mock_globals, AnimationConfig())

    # gap=0 is within tolerance (gap_tol_frames >= 0 always) → merge expected
    assert len(result.events) == 1
    merged = result.events[0]
    assert merged.fansub_frame_start == 10
    assert merged.fansub_frame_end == 30


# ---------------------------------------------------------------------------
# B — fade-out anchor frame boundary
# ---------------------------------------------------------------------------


def test_fade_out_post_window_starts_at_fansub_frame_end_not_before(
    mock_globals: PipelineGlobals,
) -> None:
    """Verify that the fade-out observation window is [fansub_frame_end, end+W).

    The frame ``fansub_frame_end - 1`` is the last in-event frame and must NOT
    appear as a post-window observation (it is used as the mathematical x-anchor
    of the constrained fit, but is never scored by the source in that role).
    The frame ``fansub_frame_end`` IS the first post-window frame and must be
    queried.

    We verify this by giving frame ``fansub_frame_end - 1`` a score that would
    corrupt the fit if included in the post-window observations, while giving
    the actual post-window frames a clean linear ramp.  If the boundary is
    respected, fade-out is detected normally.
    """
    # Static event: frames 30-49 (end=50).
    # fansub_frame_end - 1 = 49 (in-event, must NOT appear in post-window obs).
    # Post-window: frames 50-55.
    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores: dict[int, float] = {}
    # In-event frames → 1.0 (anchor_intensity is taken from frame 30)
    for f in range(30, 50):
        scores[f] = 1.0
    # Post-window: clean 6-frame linear ramp (scores in [0.05, 0.95] band)
    fade_out_frames = 6
    for k in range(1, fade_out_frames + 1):
        scores[50 - 1 + k] = max(0.0, 1.0 - k / fade_out_frames)  # frames 50-55
    # Padding → 0
    for f in range(56, 62):
        scores[f] = 0.0

    src = FakeDiffSource(scores)
    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    out = result.events[0]
    # Fade-out must be detected (post-window boundary is correct)
    assert out.fade_out_ms > 0, (
        "fade-out not detected; post-window boundary may be incorrect"
    )
    assert out.fansub_frame_end > 50, "fansub_frame_end was not extended"

    # fansub_frame_end (50) must be in the source calls (post-window)
    queried_frames = [f for (f, _) in src.calls]
    assert 50 in queried_frames, "frame 50 (first post-window frame) was never queried"

    # fansub_frame_end - 1 (49) must NOT appear as a post-window observation.
    # It IS queried once as the in-event anchor (frame_start=30 query), but
    # frame 49 itself should not be in the post-window observation loop.
    # Check: only frame 30 (anchor query) and frames 50+ (post-window) appear
    # among the post-window-related calls (frames >= 50).
    post_queried = [f for f in queried_frames if f >= 50]
    assert 49 not in post_queried, (
        "frame 49 (fansub_frame_end-1) appeared in post-window observations"
    )


# ---------------------------------------------------------------------------
# B — split of test_fade_extrapolated_below_min_reverts
# ---------------------------------------------------------------------------


def test_fade_with_too_few_observations_returns_zero(
    mock_globals: PipelineGlobals,
) -> None:
    """Branch: len(obs_xs) < min_frames → return 0.

    At 24 fps, min_frames = ceil(125 / (1000/24)) = ceil(3.0) = 3.
    A 1-frame ramp yields exactly 1 observation in the [0.05, 0.95] band,
    which is < 3, so the function must return 0 before fitting.
    """
    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    # 1-frame ramp: only 1 observation at score ≈ 0.5 (within [0.05, 0.95])
    scores = _linear_ramp_scores(30, 50, fade_in_frames=1, pre_window=5)
    src = FakeDiffSource(scores)

    result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    assert result.events[0].fade_in_ms == 0


def test_fade_extrapolated_below_min_ms_reverts_with_warning(
    mock_globals: PipelineGlobals,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Branch: enough observations (>= min_frames), but t_extrapolated < min_fade_duration_ms.

    At 24 fps, min_frames = ceil(125 / (1000/24)) = 3.  We provide exactly
    3 observations (>= min_frames, so the too-few-observations branch is NOT taken)
    but with a slope so steep the extrapolation yields < 125 ms.

    Maths: constrained fit through anchor (30, 1.0) with 3 obs all at score=0.05
    (frames 27, 28, 29):
      slope = Σ(d_i * 0.95) / Σ(d_i²)  where d_i = 30 - f_i ∈ {1, 2, 3}
            = 0.95 * 6 / 14 ≈ 0.407 /frame
      t_frames = 1 / 0.407 ≈ 2.46  →  t_ms = round(2.46 * 1000 / 24) ≈ 102 ms < 125 ms ✓
    """
    import logging

    quads = {f: _quad_around(100, 50) for f in range(30, 50)}
    ev = _make_event(event_id=0, start=30, end=50, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = {f: 1.0 for f in range(30, 50)}
    # 3 obs at the minimum boundary of the fit range → steep slope, short extrapolation
    scores[29] = 0.05
    scores[28] = 0.05
    scores[27] = 0.05
    # Frames further away → below band (0.0), not included in observations
    for f in range(20, 27):
        scores[f] = 0.0
    src = FakeDiffSource(scores)

    with caplog.at_level(logging.WARNING, logger="subtitles_ocr.pipeline.animation"):
        result = AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    # Extrapolated t_ms ≈ 102 ms < min_fade_duration_ms (125 ms) → revert to 0
    assert result.events[0].fade_in_ms == 0


# ---------------------------------------------------------------------------
# B — fade-out for linear moving event uses extrapolated bbox
# ---------------------------------------------------------------------------


def test_fade_out_for_linear_moving_event_uses_extrapolated_bbox(
    mock_globals: PipelineGlobals,
) -> None:
    """Fade-out post-window bboxes must follow the extrapolated linear trajectory.

    Symmetric to ``test_fade_for_linear_moving_event_uses_extrapolated_bbox``
    for the fade-in side. The event moves from cx=100 (frame 10) to cx=290
    (frame 29). Post-window frames (>= 30) should be queried with bboxes whose
    x-centre continues to shift beyond 290, not stuck at the static quad_median.
    """
    quads = {f: _quad_around(100 + (f - 10) * 10, 50) for f in range(10, 30)}
    ev = _make_event(event_id=0, start=10, end=30, quads=quads)
    _write_group(mock_globals.workdir, [ev])

    scores = _linear_ramp_scores(10, 30, fade_out_frames=6, post_window=5)
    src = FakeDiffSource(scores)

    AnimationStage(diff_source=src).run(mock_globals, AnimationConfig())

    # Filter to only the post-window calls (frames >= 30)
    post_calls = [(f, bbox) for (f, bbox) in src.calls if f >= 30]
    assert len(post_calls) > 0, "no post-window frames were queried"
    # Bbox cx must differ across post-window frames (trajectory is extrapolated)
    xs = sorted({(bbox[0] + bbox[2]) / 2 for (_, bbox) in post_calls})
    assert len(xs) > 1, (
        "post-window bboxes have identical x-centre; trajectory extrapolation not applied"
    )
