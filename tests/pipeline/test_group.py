"""Tests for Stage 7 — GroupStage (per-quad trajectory grouping).

Covers:
 - basic single-trajectory: 3 consecutive frames, same text + same quad → 1 event
 - simultaneous top + bottom subs on same frames → 2 distinct trajectories
 - ORPHAN frame in the middle of a textually continuous trajectory → single event
 - ALIGNED frame with no OCR detection: carried within max_gap_frames, broken when budget exhausted (max_gap_frames=0)
 - text Levenshtein above threshold → trajectory break
 - quad IoU below threshold → trajectory break
 - `quads_per_frame` keyed by `fansub_frame_idx`
 - `quad_median` computed per coordinate from member quads
 - resume: events.json + valid sidecar → re-run skips computation
"""

from __future__ import annotations

import json
from pathlib import Path

from subtitles_ocr.config import GroupConfig, PipelineGlobals
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.pipeline.alignment.stage import AlignmentResult, AlignmentSegment
from subtitles_ocr.pipeline.group import GroupResult, GroupStage, SubtitleEvent
from subtitles_ocr.pipeline.ocr import FrameOcrResult, OcrDetection


# ----------------------------- helpers -----------------------------


def _quad(x: int, y: int, w: int = 100, h: int = 30) -> list[tuple[int, int]]:
    """Axis-aligned quad at (x, y) of size (w, h), TL/TR/BR/BL order."""
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _write_ocr_jsonl(path: Path, frames: list[FrameOcrResult]) -> None:
    with JsonlWriter(path, FrameOcrResult) as w:
        for f in frames:
            w.append(f)


def _write_alignment(
    path: Path,
    *,
    fansub_total_frames: int,
    segments: list[AlignmentSegment],
) -> None:
    aligned = sum(
        (s.fansub_frame_end - s.fansub_frame_start) for s in segments if s.status == "ALIGNED"
    )
    orphan = sum(
        (s.fansub_frame_end - s.fansub_frame_start) for s in segments if s.status == "ORPHAN"
    )
    skipped = sum(
        (s.fansub_frame_end - s.fansub_frame_start) for s in segments if s.status == "USER_SKIPPED"
    )
    total = max(fansub_total_frames, 1)
    result = AlignmentResult(
        fansub_total_frames=fansub_total_frames,
        raw_total_frames=fansub_total_frames,
        method_used="phash_only",
        aligned_ratio=aligned / total,
        orphan_ratio=orphan / total,
        user_skipped_ratio=skipped / total,
        segments=segments,
        warnings=[],
    )
    path.write_text(result.model_dump_json(), encoding="utf-8")


def _all_aligned_segment(n: int) -> list[AlignmentSegment]:
    return [
        AlignmentSegment(
            fansub_frame_start=0,
            fansub_frame_end=n,
            raw_frame_start=0,
            raw_frame_end=n,
            offset_frames=0,
            status="ALIGNED",
            confidence_avg=1.0,
        )
    ]


def _setup_workdir(
    workdir: Path,
    *,
    fansub_total_frames: int,
    segments: list[AlignmentSegment],
    ocr_frames: list[FrameOcrResult],
) -> None:
    _write_alignment(
        workdir / "02_alignment" / "alignment.json",
        fansub_total_frames=fansub_total_frames,
        segments=segments,
    )
    _write_ocr_jsonl(workdir / "06_ocr" / "results.jsonl", ocr_frames)


def _globals_with(workdir: Path, total: int) -> PipelineGlobals:
    from fractions import Fraction

    return PipelineGlobals(
        workdir=workdir,
        hardsub_path=Path("/dev/null/fake_hardsub.avi"),
        raw_path=Path("/dev/null/fake_raw.mkv"),
        out_path=workdir / "out.ass",
        fps=Fraction(24, 1),
        fansub_width=1920,
        fansub_height=1080,
        fansub_total_frames=total,
        debug_images=False,
    )


# ----------------------------- tests -----------------------------


def test_three_consecutive_same_text_same_quad_yields_one_event(tmp_workdir: Path) -> None:
    q = _quad(100, 800)
    frames = [
        FrameOcrResult(fansub_frame_idx=i, detections=[OcrDetection(text="hello", confidence=0.9, quad=q)])
        for i in range(3)
    ]
    _setup_workdir(
        tmp_workdir,
        fansub_total_frames=3,
        segments=_all_aligned_segment(3),
        ocr_frames=frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 3), GroupConfig())

    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.fansub_frame_start == 0
    assert ev.fansub_frame_end == 3  # half-open: [0, 3)
    assert ev.raw_ocr_texts == ["hello", "hello", "hello"]
    assert ev.raw_ocr_confidences == [0.9, 0.9, 0.9]
    assert ev.member_frame_indices == [0, 1, 2]
    assert ev.quads_per_frame == {0: q, 1: q, 2: q}
    assert ev.quad_median == q


def test_simultaneous_top_and_bottom_subs_yield_two_events(tmp_workdir: Path) -> None:
    top = _quad(100, 50)
    bot = _quad(100, 800)
    frames = [
        FrameOcrResult(
            fansub_frame_idx=i,
            detections=[
                OcrDetection(text="signe", confidence=0.8, quad=top),
                OcrDetection(text="dialogue", confidence=0.9, quad=bot),
            ],
        )
        for i in range(3)
    ]
    _setup_workdir(
        tmp_workdir,
        fansub_total_frames=3,
        segments=_all_aligned_segment(3),
        ocr_frames=frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 3), GroupConfig())

    assert len(result.events) == 2
    by_text = {ev.raw_ocr_texts[0]: ev for ev in result.events}
    assert set(by_text.keys()) == {"signe", "dialogue"}
    assert by_text["signe"].quad_median == top
    assert by_text["dialogue"].quad_median == bot
    assert by_text["signe"].fansub_frame_end == 3
    assert by_text["dialogue"].fansub_frame_end == 3


def test_orphan_frame_does_not_break_trajectory(tmp_workdir: Path) -> None:
    q = _quad(100, 800)
    # Frames 0, 1 ALIGNED; frame 2 ORPHAN (no OCR); frames 3, 4 ALIGNED.
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=3, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=4, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    segments = [
        AlignmentSegment(
            fansub_frame_start=0, fansub_frame_end=2, raw_frame_start=0, raw_frame_end=2,
            offset_frames=0, status="ALIGNED", confidence_avg=1.0,
        ),
        AlignmentSegment(
            fansub_frame_start=2, fansub_frame_end=3, raw_frame_start=None, raw_frame_end=None,
            offset_frames=None, status="ORPHAN", confidence_avg=None,
        ),
        AlignmentSegment(
            fansub_frame_start=3, fansub_frame_end=5, raw_frame_start=2, raw_frame_end=4,
            offset_frames=-1, status="ALIGNED", confidence_avg=1.0,
        ),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=5, segments=segments, ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 5), GroupConfig())

    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.fansub_frame_start == 0
    assert ev.fansub_frame_end == 5
    assert ev.member_frame_indices == [0, 1, 3, 4]
    assert ev.raw_ocr_texts == ["hi", "hi", "hi", "hi"]


def test_aligned_frame_without_detection_breaks_trajectory_when_gap_budget_zero(
    tmp_workdir: Path,
) -> None:
    """Pre-gap-tolerance semantics, re-expressed: with max_gap_frames=0, a
    single ALIGNED frame with no detection breaks the trajectory immediately."""
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=2, detections=[]),
        FrameOcrResult(fansub_frame_idx=3, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=4, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=5,
        segments=_all_aligned_segment(5), ocr_frames=ocr_frames,
    )

    cfg = GroupConfig(max_gap_frames=0)
    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 5), cfg)

    assert len(result.events) == 2
    a, b = sorted(result.events, key=lambda e: e.fansub_frame_start)
    assert (a.fansub_frame_start, a.fansub_frame_end) == (0, 2)
    assert a.member_frame_indices == [0, 1]
    assert (b.fansub_frame_start, b.fansub_frame_end) == (3, 5)
    assert b.member_frame_indices == [3, 4]


def test_text_levenshtein_above_threshold_breaks_trajectory(tmp_workdir: Path) -> None:
    q = _quad(100, 800)
    # Frame 0: "hello"; frame 1: "world" (Lev distance 4, normalized = 0.8 > 0.2 → break).
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hello", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="world", confidence=0.9, quad=q)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=2,
        segments=_all_aligned_segment(2), ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 2), GroupConfig())

    assert len(result.events) == 2
    texts = {ev.raw_ocr_texts[0] for ev in result.events}
    assert texts == {"hello", "world"}


def test_quad_iou_below_threshold_breaks_trajectory(tmp_workdir: Path) -> None:
    q0 = _quad(0, 0, w=10, h=10)
    q1 = _quad(100, 100, w=10, h=10)  # disjoint with q0 → IoU = 0 < 0.5 → break
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="x", confidence=0.9, quad=q0)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="x", confidence=0.9, quad=q1)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=2,
        segments=_all_aligned_segment(2), ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 2), GroupConfig())

    assert len(result.events) == 2


def test_quads_per_frame_is_keyed_by_fansub_frame_idx(tmp_workdir: Path) -> None:
    q = _quad(100, 800)
    # Use non-zero starting frames to make the keying explicit.
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=10, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=11, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=12, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    segments = [
        AlignmentSegment(
            fansub_frame_start=0, fansub_frame_end=10, raw_frame_start=None, raw_frame_end=None,
            offset_frames=None, status="ORPHAN", confidence_avg=None,
        ),
        AlignmentSegment(
            fansub_frame_start=10, fansub_frame_end=13, raw_frame_start=0, raw_frame_end=3,
            offset_frames=-10, status="ALIGNED", confidence_avg=1.0,
        ),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=13, segments=segments, ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 13), GroupConfig())

    assert len(result.events) == 1
    ev = result.events[0]
    assert set(ev.quads_per_frame.keys()) == {10, 11, 12}
    assert ev.member_frame_indices == [10, 11, 12]


def test_quad_median_is_per_coordinate_median(tmp_workdir: Path) -> None:
    # Three quads, slightly different; median along each coord independently.
    q0 = [(0, 0), (10, 0), (10, 5), (0, 5)]
    q1 = [(1, 1), (11, 1), (11, 6), (1, 6)]
    q2 = [(2, 2), (12, 2), (12, 7), (2, 7)]
    expected_median = [(1, 1), (11, 1), (11, 6), (1, 6)]
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="t", confidence=0.9, quad=q0)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="t", confidence=0.9, quad=q1)]),
        FrameOcrResult(fansub_frame_idx=2, detections=[OcrDetection(text="t", confidence=0.9, quad=q2)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=3,
        segments=_all_aligned_segment(3), ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 3), GroupConfig())

    assert len(result.events) == 1
    assert result.events[0].quad_median == expected_median


def test_writes_events_json_and_sidecar(tmp_workdir: Path) -> None:
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=i, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)])
        for i in range(2)
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=2,
        segments=_all_aligned_segment(2), ocr_frames=ocr_frames,
    )

    GroupStage().run(_globals_with(tmp_workdir, 2), GroupConfig())

    events_path = tmp_workdir / "07_group" / "events.json"
    sidecar_path = tmp_workdir / "07_group" / "events.meta.json"
    assert events_path.exists()
    assert sidecar_path.exists()
    # The on-disk events.json round-trips through the Pydantic schema.
    GroupResult.model_validate_json(events_path.read_text(encoding="utf-8"))
    # The sidecar carries stage_name and stage_version.
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert meta["stage_name"] == "07_group"
    assert meta["stage_version"] == 2


def test_resume_reuses_cached_output_when_sidecar_matches(tmp_workdir: Path) -> None:
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=2,
        segments=_all_aligned_segment(2), ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    globs = _globals_with(tmp_workdir, 2)
    first = stage.run(globs, GroupConfig())
    events_path = tmp_workdir / "07_group" / "events.json"

    # Sabotage the persisted events.json with a sentinel value that differs from
    # what a recompute would produce. If the resume logic re-reads cache, this
    # sentinel survives. If it recomputes, the sentinel is wiped.
    sentinel = GroupResult(
        fansub_total_frames=999,
        events=[],
        stats={"resumed": True},
    )
    events_path.write_text(sentinel.model_dump_json(), encoding="utf-8")

    second = stage.run(globs, GroupConfig())
    assert second.fansub_total_frames == 999
    assert second.stats == {"resumed": True}
    # And confirm a recompute would have produced something different.
    assert first.fansub_total_frames == 2


def test_group_config_exposes_max_gap_frames_default() -> None:
    """GroupConfig ships with a max_gap_frames knob (default 10 frames, ~0.4 s @ 24fps).

    The value bridges OCR misses inside a continuous subtitle (typical fade-in
    durations are 0.5 s but OCR usually catches at least one frame in that
    window) without merging consecutive dialogue lines that are separated by
    longer silences.

    Documents the cache-invalidating contract: changing this value must trigger
    a re-run of Stage 7. Lives next to the other group thresholds.
    """
    cfg = GroupConfig()
    assert cfg.max_gap_frames == 15


def test_group_stage_version_is_bumped_for_gap_tolerance() -> None:
    """Bumped from 1 to 2 to invalidate cached events.json produced by the
    pre-gap-tolerance grouping logic."""
    from subtitles_ocr.pipeline.group import STAGE_VERSION
    assert STAGE_VERSION == 2


def test_single_aligned_gap_within_tolerance_keeps_trajectory(tmp_workdir: Path) -> None:
    """Frame N has no detection on an ALIGNED frame; with gap tolerance >= 1,
    the trajectory survives and re-acquires on the next frame.

    Members exclude the gap frame; event end_exclusive = last_matched_frame + 1.
    """
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=2, detections=[]),
        FrameOcrResult(fansub_frame_idx=3, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=4, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=5,
        segments=_all_aligned_segment(5), ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 5), GroupConfig())

    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.fansub_frame_start == 0
    assert ev.fansub_frame_end == 5  # last_matched (4) + 1
    assert ev.member_frame_indices == [0, 1, 3, 4]
    assert ev.raw_ocr_texts == ["hi", "hi", "hi", "hi"]
    assert ev.raw_ocr_confidences == [0.9, 0.9, 0.9, 0.9]
    assert set(ev.quads_per_frame.keys()) == {0, 1, 3, 4}


def test_gap_exceeding_max_finalizes_at_last_matched_plus_one(tmp_workdir: Path) -> None:
    """With max_gap_frames=2, a 3-frame gap exceeds the budget and finalizes
    the first trajectory at last_matched + 1 (NOT at the gap-closing frame).

    Frames 0,1 match → 2,3,4 empty (3-frame gap) → 5,6 match same text+quad.
    The first event ends at frame 2 (last_matched=1, +1=2). A new trajectory
    starts at frame 5.
    """
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=2, detections=[]),
        FrameOcrResult(fansub_frame_idx=3, detections=[]),
        FrameOcrResult(fansub_frame_idx=4, detections=[]),
        FrameOcrResult(fansub_frame_idx=5, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=6, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=7,
        segments=_all_aligned_segment(7), ocr_frames=ocr_frames,
    )

    cfg = GroupConfig(max_gap_frames=2)
    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 7), cfg)

    assert len(result.events) == 2
    a, b = sorted(result.events, key=lambda e: e.fansub_frame_start)
    assert (a.fansub_frame_start, a.fansub_frame_end) == (0, 2)
    assert a.member_frame_indices == [0, 1]
    assert (b.fansub_frame_start, b.fansub_frame_end) == (5, 7)
    assert b.member_frame_indices == [5, 6]


def test_gap_equal_to_max_keeps_trajectory(tmp_workdir: Path) -> None:
    """With max_gap_frames=2, exactly 2 empty frames stay within the budget;
    the trajectory survives. Boundary test against the > comparison."""
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[]),
        FrameOcrResult(fansub_frame_idx=2, detections=[]),
        FrameOcrResult(fansub_frame_idx=3, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=4,
        segments=_all_aligned_segment(4), ocr_frames=ocr_frames,
    )

    cfg = GroupConfig(max_gap_frames=2)
    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 4), cfg)

    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.fansub_frame_start == 0
    assert ev.fansub_frame_end == 4
    assert ev.member_frame_indices == [0, 3]


def test_trajectory_with_trailing_gap_ends_at_last_matched_plus_one(tmp_workdir: Path) -> None:
    """A trajectory whose last match is at frame K, followed by gap frames
    that do not exceed max, then end-of-video → event ends at K+1, not at
    fansub_total_frames."""
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=2, detections=[]),
        FrameOcrResult(fansub_frame_idx=3, detections=[]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=4,
        segments=_all_aligned_segment(4), ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 4), GroupConfig())  # default max_gap_frames=60

    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.fansub_frame_start == 0
    assert ev.fansub_frame_end == 2  # last_matched=1, +1=2; NOT fansub_total_frames=4
    assert ev.member_frame_indices == [0, 1]


def test_orphan_frames_do_not_consume_gap_budget(tmp_workdir: Path) -> None:
    """ORPHAN frames are neutral: they don't match a trajectory and they don't
    increment its stale counter. A gap composed entirely of ORPHAN frames
    survives any max_gap_frames value, including 0.

    Frames 0,1: ALIGNED, matched. Frames 2-10: ORPHAN. Frame 11: ALIGNED, matched.
    With max_gap_frames=0 the trajectory still survives because ORPHANs don't
    count.
    """
    q = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
        FrameOcrResult(fansub_frame_idx=11, detections=[OcrDetection(text="hi", confidence=0.9, quad=q)]),
    ]
    segments = [
        AlignmentSegment(
            fansub_frame_start=0, fansub_frame_end=2,
            raw_frame_start=0, raw_frame_end=2,
            offset_frames=0, status="ALIGNED", confidence_avg=1.0,
        ),
        AlignmentSegment(
            fansub_frame_start=2, fansub_frame_end=11,
            raw_frame_start=None, raw_frame_end=None,
            offset_frames=None, status="ORPHAN", confidence_avg=None,
        ),
        AlignmentSegment(
            fansub_frame_start=11, fansub_frame_end=12,
            raw_frame_start=2, raw_frame_end=3,
            offset_frames=-9, status="ALIGNED", confidence_avg=1.0,
        ),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=12, segments=segments, ocr_frames=ocr_frames,
    )

    cfg = GroupConfig(max_gap_frames=0)
    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 12), cfg)

    assert len(result.events) == 1
    ev = result.events[0]
    assert ev.fansub_frame_start == 0
    assert ev.fansub_frame_end == 12  # last_matched=11, +1=12
    assert ev.member_frame_indices == [0, 1, 11]


def test_unrelated_detection_during_gap_does_not_resurrect_or_kill_trajectory(
    tmp_workdir: Path,
) -> None:
    """During a gap, a detection that fails BOTH Lev (text) and IoU (position)
    versus the carried trajectory must:
      - not be matched to that trajectory (no false merge);
      - not reset its stale counter to 0;
      - spawn its own new trajectory;
      - leave the original trajectory eligible to re-acquire later.

    Setup: trajectory "hi" at quad_top. Mid-gap: detection "totallyelse" at
    quad_bottom (disjoint quad, very different text). Then "hi" at quad_top
    again, within budget. Expected: 2 events. The "hi" event spans 0..end with
    members at the two "hi" frames.
    """
    q_top = _quad(100, 50)
    q_bot = _quad(100, 800)
    ocr_frames = [
        FrameOcrResult(fansub_frame_idx=0, detections=[OcrDetection(text="hi", confidence=0.9, quad=q_top)]),
        # Frame 1: an unrelated detection at a disjoint quad with very different text.
        FrameOcrResult(fansub_frame_idx=1, detections=[OcrDetection(text="totallyelse", confidence=0.9, quad=q_bot)]),
        # Frame 2: empty (deliberate, to confirm stale ticks past the unrelated detection).
        FrameOcrResult(fansub_frame_idx=2, detections=[]),
        # Frame 3: "hi" back at quad_top → re-acquires.
        FrameOcrResult(fansub_frame_idx=3, detections=[OcrDetection(text="hi", confidence=0.9, quad=q_top)]),
    ]
    _setup_workdir(
        tmp_workdir, fansub_total_frames=4,
        segments=_all_aligned_segment(4), ocr_frames=ocr_frames,
    )

    stage = GroupStage()
    result = stage.run(_globals_with(tmp_workdir, 4), GroupConfig())  # default max_gap_frames=60

    by_text = {tuple(sorted(set(ev.raw_ocr_texts))): ev for ev in result.events}
    assert ("hi",) in by_text
    assert ("totallyelse",) in by_text

    hi_ev = by_text[("hi",)]
    assert hi_ev.member_frame_indices == [0, 3]
    assert hi_ev.fansub_frame_start == 0
    assert hi_ev.fansub_frame_end == 4  # last_matched=3, +1=4

    else_ev = by_text[("totallyelse",)]
    assert else_ev.member_frame_indices == [1]
    assert else_ev.fansub_frame_start == 1
    assert else_ev.fansub_frame_end == 2  # last_matched=1, +1=2
