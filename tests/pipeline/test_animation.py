"""Tests for AnimationStage MVP passthrough (ADR-0003 §3)."""

from __future__ import annotations

import json
from pathlib import Path

from subtitles_ocr.config import AnimationConfig, PipelineGlobals
from subtitles_ocr.meta import BaseMeta
from subtitles_ocr.pipeline.animation import (
    AnimatedEvent,
    AnimationAnalysisResult,
    AnimationStage,
)
from subtitles_ocr.pipeline.group import GroupResult, SubtitleEvent


def _write_group_result(workdir: Path, events: list[SubtitleEvent]) -> None:
    payload = GroupResult(
        fansub_total_frames=1000,
        events=events,
        stats={"events": len(events)},
    )
    (workdir / "07_group" / "events.json").write_text(payload.model_dump_json())


def _three_events() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(
            event_id=0,
            fansub_frame_start=10,
            fansub_frame_end=30,
            raw_ocr_texts=["hi"],
            raw_ocr_confidences=[0.9],
            quads_per_frame={
                10: [(0, 0), (10, 0), (10, 5), (0, 5)],
                11: [(1, 0), (11, 0), (11, 5), (1, 5)],
            },
            quad_median=[(0, 0), (10, 0), (10, 5), (0, 5)],
            member_frame_indices=[10, 11],
        ),
        SubtitleEvent(
            event_id=1,
            fansub_frame_start=40,
            fansub_frame_end=60,
            raw_ocr_texts=["hello", "hello"],
            raw_ocr_confidences=[0.8, 0.85],
            quads_per_frame={
                40: [(100, 100), (200, 100), (200, 130), (100, 130)],
            },
            quad_median=[(100, 100), (200, 100), (200, 130), (100, 130)],
            member_frame_indices=[40, 41, 42],
        ),
        SubtitleEvent(
            event_id=2,
            fansub_frame_start=80,
            fansub_frame_end=85,
            raw_ocr_texts=["bye"],
            raw_ocr_confidences=[0.7],
            quads_per_frame={},
            quad_median=[(50, 50), (60, 50), (60, 60), (50, 60)],
            member_frame_indices=[80],
        ),
    ]


def test_passthrough_produces_one_animated_event_per_input(
    mock_globals: PipelineGlobals,
) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert isinstance(result, AnimationAnalysisResult)
    assert len(result.events) == 3


def test_passthrough_motion_is_none_and_fades_are_zero(
    mock_globals: PipelineGlobals,
) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    result = AnimationStage().run(mock_globals, AnimationConfig())

    for ev in result.events:
        assert ev.motion is None
        assert ev.fade_in_ms == 0
        assert ev.fade_out_ms == 0


def test_passthrough_preserves_quads_per_frame_strictly(
    mock_globals: PipelineGlobals,
) -> None:
    src = _three_events()
    _write_group_result(mock_globals.workdir, src)

    result = AnimationStage().run(mock_globals, AnimationConfig())

    by_id = {e.event_id: e for e in result.events}
    for original in src:
        out = by_id[original.event_id]
        assert out.quads_per_frame == original.quads_per_frame
        assert out.quad_median == original.quad_median
        assert out.member_frame_indices == original.member_frame_indices
        assert out.raw_ocr_texts == original.raw_ocr_texts
        assert out.raw_ocr_confidences == original.raw_ocr_confidences
        assert out.fansub_frame_start == original.fansub_frame_start
        assert out.fansub_frame_end == original.fansub_frame_end


def test_passthrough_stats(mock_globals: PipelineGlobals) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    result = AnimationStage().run(mock_globals, AnimationConfig())

    assert result.stats == {
        "static": 3,
        "linear_move": 0,
        "flagged_nonlinear": 0,
        "fade_in_only": 0,
        "fade_out_only": 0,
        "full_fade": 0,
    }


def test_passthrough_writes_animation_json(mock_globals: PipelineGlobals) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    AnimationStage().run(mock_globals, AnimationConfig())

    out_path = mock_globals.workdir / "08_animation" / "animation.json"
    assert out_path.exists()
    persisted = AnimationAnalysisResult.model_validate_json(out_path.read_text())
    assert len(persisted.events) == 3
    assert all(ev.motion is None for ev in persisted.events)


def test_passthrough_writes_sidecar(mock_globals: PipelineGlobals) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    AnimationStage().run(mock_globals, AnimationConfig())

    sidecar = mock_globals.workdir / "08_animation" / "animation.meta.json"
    assert sidecar.exists()
    meta = BaseMeta.model_validate_json(sidecar.read_text())
    assert meta.stage_name == "08_animation"
    assert meta.stage_version == 1


def test_passthrough_write_is_atomic_leaves_no_tmp_file(
    mock_globals: PipelineGlobals,
) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    AnimationStage().run(mock_globals, AnimationConfig())

    leftovers = list((mock_globals.workdir / "08_animation").glob("*.tmp"))
    assert leftovers == []


def test_resume_skips_when_sidecar_matches(mock_globals: PipelineGlobals) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    stage = AnimationStage()
    first = stage.run(mock_globals, AnimationConfig())

    out_path = mock_globals.workdir / "08_animation" / "animation.json"
    # Mutate the cached output: if the stage skips, the mutated content is what
    # we get back. If the stage recomputes, our mutation is overwritten.
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
            stats={"static": 1, "linear_move": 0, "flagged_nonlinear": 0,
                   "fade_in_only": 0, "fade_out_only": 0, "full_fade": 0},
        ).model_dump_json()
    )

    second = stage.run(mock_globals, AnimationConfig())
    assert second.events[0].event_id == 999
    assert len(second.events) == 1
    # And first result was the real 3-event output
    assert len(first.events) == 3


def test_resume_recomputes_when_config_changes(mock_globals: PipelineGlobals) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    stage = AnimationStage()
    stage.run(mock_globals, AnimationConfig())

    out_path = mock_globals.workdir / "08_animation" / "animation.json"
    out_path.write_text(
        AnimationAnalysisResult(events=[], stats={
            "static": 0, "linear_move": 0, "flagged_nonlinear": 0,
            "fade_in_only": 0, "fade_out_only": 0, "full_fade": 0,
        }).model_dump_json()
    )

    # Change a cache-invalidating field
    changed = AnimationConfig(min_move_displacement_px=99)
    second = stage.run(mock_globals, changed)

    assert len(second.events) == 3


def test_animation_json_key_types_round_trip(mock_globals: PipelineGlobals) -> None:
    _write_group_result(mock_globals.workdir, _three_events())

    AnimationStage().run(mock_globals, AnimationConfig())

    out_path = mock_globals.workdir / "08_animation" / "animation.json"
    # Pydantic v2 emits int-keyed dicts as string keys in JSON; reading raw and
    # re-validating must produce the original int keys on AnimatedEvent.
    raw = json.loads(out_path.read_text())
    first_event_quads = raw["events"][0]["quads_per_frame"]
    assert all(isinstance(k, str) for k in first_event_quads.keys())

    validated = AnimationAnalysisResult.model_validate(raw)
    assert 10 in validated.events[0].quads_per_frame
