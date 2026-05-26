"""JSONL stage resume + corruption tests (OCR and EventCleanup).

Per ADR-0004 §5.4 each JSONL-driven stage uses `JsonlWriter`:
  - resume_index() counts valid lines at the start
  - mid-file corruption raises `CacheCorruptionError`
  - a partial trailing line is tolerated (truncated on reopen)

These tests exercise the contract end-to-end against tmp_workdir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from subtitles_ocr.config import (
    AnimationConfig,
    ConformConfig,
    EventCleanupConfig,
    FrameProcessingConfig,
    GroupConfig,
    OcrConfig,
    PipelineGlobals,
)
from subtitles_ocr.exceptions import CacheCorruptionError
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.pipeline.alignment.stage import (
    AlignmentConfig,
    AlignmentStage,
)
from subtitles_ocr.pipeline.animation import (
    AnimatedEvent,
    AnimationAnalysisResult,
    AnimationStage,
)
from subtitles_ocr.pipeline.conform import ConformStage
from subtitles_ocr.pipeline.event_cleanup import (
    CleanedEvent,
    EventCleanupItem,
    EventCleanupStage,
)
from subtitles_ocr.pipeline.frame_processing.iterator import ComposedFrame
from subtitles_ocr.pipeline.group import GroupStage
from subtitles_ocr.pipeline.ocr import FrameOcrResult, OcrStage

from tests.integration.conftest import (
    FakeFfmpeg,
    FakeFrameSource,
    FakeLlm,
    FakeOcrEngine,
    make_video_metadata,
)


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------


def _build_fake_ffmpeg(globals_: PipelineGlobals) -> FakeFfmpeg:
    meta = make_video_metadata(globals_.fansub_width, globals_.fansub_height)
    return FakeFfmpeg(
        probe_returns={
            globals_.hardsub_path: meta,
            globals_.raw_path: meta,
        }
    )


def _composed_frames(n: int) -> list[ComposedFrame]:
    return [
        ComposedFrame(fansub_frame_idx=i, image=np.full((8, 8, 3), i, dtype=np.uint8))
        for i in range(n)
    ]


def _prepare_alignment_only(globals_: PipelineGlobals) -> None:
    """Run only Conform + Alignment, leaving downstream stages untouched."""
    ConformStage(ffmpeg=_build_fake_ffmpeg(globals_)).run(globals_, ConformConfig())
    AlignmentStage(
        frame_source=FakeFrameSource(),
        raw_total_frames=globals_.fansub_total_frames,
    ).run(globals_, AlignmentConfig())


# ---------------------------------------------------------------------------
# OCR resume tests
# ---------------------------------------------------------------------------


def test_ocr_partial_then_resume_continues_from_last_index(
    integration_globals: PipelineGlobals,
) -> None:
    """Pre-seed 3 frames in results.jsonl as if a previous run was killed.
    Re-running OcrStage with a 5-frame composed iterator must invoke the engine
    only for frames 3 and 4 (resume_index = 3).
    """
    _prepare_alignment_only(integration_globals)

    jsonl = integration_globals.workdir / "06_ocr" / "results.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Pre-seed 3 already-OCRed frames as the previous-run remnant.
    with JsonlWriter(jsonl, FrameOcrResult) as w:
        for i in range(3):
            w.append(FrameOcrResult(fansub_frame_idx=i, detections=[]))

    engine = FakeOcrEngine()
    OcrStage(ocr_engine=engine).run(
        integration_globals,
        OcrConfig(frame_processing=FrameProcessingConfig()),
        composed_frames=iter(_composed_frames(5)),
    )

    # Engine called exactly for the missing tail (indices 3 and 4)
    assert engine.call_count == 2

    persisted = list(JsonlWriter(jsonl, FrameOcrResult).iter_persisted())
    assert [p.fansub_frame_idx for p in persisted] == [0, 1, 2, 3, 4]


def test_ocr_resume_with_trailing_partial_line_truncates_and_continues(
    integration_globals: PipelineGlobals,
) -> None:
    """JsonlWriter tolerates a truncated last line on reopen: the partial line
    is silently dropped, resume_index reflects valid lines only.
    """
    _prepare_alignment_only(integration_globals)

    jsonl = integration_globals.workdir / "06_ocr" / "results.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 2 complete lines + a truncated trailing line (no closing newline).
    with jsonl.open("wb") as f:
        for i in range(2):
            f.write(FrameOcrResult(fansub_frame_idx=i, detections=[]).model_dump_json().encode())
            f.write(b"\n")
        f.write(b'{"fansub_frame_idx": 2, "detections": []')  # truncated

    engine = FakeOcrEngine()
    OcrStage(ocr_engine=engine).run(
        integration_globals,
        OcrConfig(frame_processing=FrameProcessingConfig()),
        composed_frames=iter(_composed_frames(5)),
    )

    # Engine should be called for frames 2, 3, 4 (after the 2 valid lines).
    assert engine.call_count == 3
    persisted = list(JsonlWriter(jsonl, FrameOcrResult).iter_persisted())
    assert [p.fansub_frame_idx for p in persisted] == [0, 1, 2, 3, 4]


def test_ocr_midfile_corruption_raises_cache_corruption_error(
    integration_globals: PipelineGlobals,
) -> None:
    """A corrupted line in the *middle* of the JSONL (not the trailing one) is
    a hard failure surfaced as CacheCorruptionError.
    """
    _prepare_alignment_only(integration_globals)

    jsonl = integration_globals.workdir / "06_ocr" / "results.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Valid line 0, garbage line 1 (complete with \n so it's not tolerated as
    # a trailing-partial), valid line 2.
    with jsonl.open("wb") as f:
        f.write(FrameOcrResult(fansub_frame_idx=0, detections=[]).model_dump_json().encode())
        f.write(b"\n")
        f.write(b"{not json at all}\n")
        f.write(FrameOcrResult(fansub_frame_idx=2, detections=[]).model_dump_json().encode())
        f.write(b"\n")

    engine = FakeOcrEngine()
    with pytest.raises(CacheCorruptionError):
        OcrStage(ocr_engine=engine).run(
            integration_globals,
            OcrConfig(frame_processing=FrameProcessingConfig()),
            composed_frames=iter(_composed_frames(5)),
        )


# ---------------------------------------------------------------------------
# EventCleanup resume tests
# ---------------------------------------------------------------------------


def _make_animated_event(event_id: int, texts: list[str]) -> AnimatedEvent:
    quad = [(0, 0), (10, 0), (10, 5), (0, 5)]
    return AnimatedEvent(
        event_id=event_id,
        fansub_frame_start=event_id * 10,
        fansub_frame_end=event_id * 10 + 5,
        raw_ocr_texts=texts,
        raw_ocr_confidences=[0.9] * len(texts),
        quads_per_frame={event_id * 10: quad},
        quad_median=quad,
        member_frame_indices=[event_id * 10],
        motion=None,
        fade_in_ms=0,
        fade_out_ms=0,
    )


def _write_animation_json(workdir: Path, events: list[AnimatedEvent]) -> None:
    out = workdir / "08_animation" / "animation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(AnimationAnalysisResult(events=events, stats={}).model_dump_json())


def test_event_cleanup_partial_then_resume_skips_done_events(
    integration_globals: PipelineGlobals,
) -> None:
    """Pre-seed cleaned.jsonl with 3 of 5 event results, re-run, only the last
    2 events hit the LLM.
    """
    events = [_make_animated_event(i, [f"t{i}-a", f"t{i}-b"]) for i in range(5)]
    _write_animation_json(integration_globals.workdir, events)

    out_dir = integration_globals.workdir / "10_event_cleanup"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "cleaned.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(
                EventCleanupItem(
                    event_id=i, cleaned_text=f"pre{i}", skipped_llm=False
                ).model_dump_json()
                + "\n"
            )

    llm = FakeLlm(response_factory=lambda schema, prompt: CleanedEvent(text="new"))
    EventCleanupStage(llm=llm).run(
        integration_globals,
        EventCleanupConfig(model="m"),
    )

    # Only events 3 and 4 hit the LLM
    assert llm.call_count == 2
    persisted = list(JsonlWriter(jsonl, EventCleanupItem).iter_persisted())
    assert [p.event_id for p in persisted] == [0, 1, 2, 3, 4]
    assert persisted[0].cleaned_text == "pre0"
    assert persisted[4].cleaned_text == "new"


def test_event_cleanup_midfile_corruption_raises_cache_corruption_error(
    integration_globals: PipelineGlobals,
) -> None:
    """Corrupted middle line in cleaned.jsonl → CacheCorruptionError on the
    next EventCleanupStage.run().
    """
    events = [_make_animated_event(i, [f"t{i}-a", f"t{i}-b"]) for i in range(3)]
    _write_animation_json(integration_globals.workdir, events)

    out_dir = integration_globals.workdir / "10_event_cleanup"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "cleaned.jsonl"
    with jsonl.open("wb") as f:
        f.write(
            EventCleanupItem(event_id=0, cleaned_text="ok", skipped_llm=False)
            .model_dump_json()
            .encode()
        )
        f.write(b"\n")
        f.write(b"{ this is not valid json at all }\n")
        f.write(
            EventCleanupItem(event_id=2, cleaned_text="ok", skipped_llm=False)
            .model_dump_json()
            .encode()
        )
        f.write(b"\n")

    llm = FakeLlm(response_factory=lambda schema, prompt: CleanedEvent(text="x"))
    with pytest.raises(CacheCorruptionError):
        EventCleanupStage(llm=llm).run(
            integration_globals,
            EventCleanupConfig(model="m"),
        )
