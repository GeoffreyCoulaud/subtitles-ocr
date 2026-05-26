"""End-to-end smoke test: 9-stage pipeline with fakes → parsable .ass.

This is the canonical "does the whole thing wire together" test. It does NOT
verify subtitle correctness or visual fidelity; it verifies that:

  - every stage runs without raising,
  - every intermediate artefact exists in the workdir,
  - the final .ass file is parsable by pysubs2 (the very same library the
    Export stage uses internally — but parsing on the consumer side is a
    proxy for "the file is well-formed").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pysubs2

from subtitles_ocr.config import (
    AnimationConfig,
    ColorConfig,
    ConformConfig,
    DocCleanupConfig,
    EventCleanupConfig,
    ExportConfig,
    FrameProcessingConfig,
    GroupConfig,
    OcrConfig,
    PipelineGlobals,
)
from subtitles_ocr.pipeline.alignment.stage import AlignmentConfig, AlignmentStage
from subtitles_ocr.pipeline.animation import AnimationStage
from subtitles_ocr.pipeline.color import ColorStage
from subtitles_ocr.pipeline.conform import ConformStage
from subtitles_ocr.pipeline.doc_cleanup import (
    DocCleanupResult,
    DocCleanupStage,
    FinalEvent,
)
from subtitles_ocr.pipeline.event_cleanup import CleanedEvent, EventCleanupStage
from subtitles_ocr.pipeline.export import ExportStage
from subtitles_ocr.pipeline.frame_processing.iterator import ComposedFrame
from subtitles_ocr.pipeline.group import GroupResult, GroupStage
from subtitles_ocr.pipeline.ocr import OcrStage

from tests.integration.conftest import (
    FakeFfmpeg,
    FakeFrameReader,
    FakeFrameSource,
    FakeLlm,
    FakeOcrEngine,
    make_video_metadata,
)


def _composed_for(globals_: PipelineGlobals) -> list[ComposedFrame]:
    """Synthetic composed frames: 5 fansub frames + 5 raw frames are encoded
    here as 5 ComposedFrame objects (one per aligned fansub idx); the diff /
    mask / compose stages are bypassed via the `composed_frames` injection
    point, exactly as integration tests must do (those stages need a real
    video reader).
    """
    return [
        ComposedFrame(
            fansub_frame_idx=i,
            image=np.full((32, 32, 3), 100 + i, dtype=np.uint8),
        )
        for i in range(globals_.fansub_total_frames)
    ]


def test_full_pipeline_produces_pysubs2_parsable_ass(
    integration_globals: PipelineGlobals,
) -> None:
    g = integration_globals
    fake_meta = make_video_metadata(g.fansub_width, g.fansub_height)
    fake_ffmpeg = FakeFfmpeg(
        probe_returns={g.hardsub_path: fake_meta, g.raw_path: fake_meta}
    )

    # Stage 1
    ConformStage(ffmpeg=fake_ffmpeg).run(g, ConformConfig())

    # Stage 2 — phash-only branch via FakeFrameSource
    AlignmentStage(
        frame_source=FakeFrameSource(),
        raw_total_frames=g.fansub_total_frames,
    ).run(g, AlignmentConfig())

    # Stage 6 — composed frames injected directly
    OcrStage(ocr_engine=FakeOcrEngine()).run(
        g,
        OcrConfig(),
        frame_processing_config=FrameProcessingConfig(),
        composed_frames=iter(_composed_for(g)),
    )

    # Stage 7
    GroupStage().run(g, GroupConfig())

    # Stage 8 (passthrough MVP)
    AnimationStage().run(g, AnimationConfig())

    # Stage 9 — fake frame reader feeding synthetic white-on-dark glyphs
    ColorStage(
        frame_reader=FakeFrameReader(width=g.fansub_width, height=g.fansub_height)
    ).run(g, ColorConfig())

    # Stage 10 — consensus text → no LLM call needed
    EventCleanupStage(
        llm=FakeLlm(response_factory=lambda s, p: CleanedEvent(text="X"))
    ).run(g, EventCleanupConfig(model="m"))

    # Stage 11 — must echo the exact event_ids in order
    events_count = len(
        GroupResult.model_validate_json(
            (g.workdir / "07_group" / "events.json").read_text()
        ).events
    )
    DocCleanupStage(
        llm=FakeLlm(
            response_factory=lambda s, p: DocCleanupResult(
                events=[
                    FinalEvent(event_id=i, cleaned_text=f"final {i}")
                    for i in range(events_count)
                ]
            )
        )
    ).run(g, DocCleanupConfig(model="m"))

    # Stage 12
    ExportStage().run(g, ExportConfig())

    # All intermediate artefacts exist
    expected = [
        g.workdir / "01_conform" / "raw.mkv",
        g.workdir / "02_alignment" / "alignment.json",
        g.workdir / "06_ocr" / "results.jsonl",
        g.workdir / "07_group" / "events.json",
        g.workdir / "08_animation" / "animation.json",
        g.workdir / "09_color" / "colors.json",
        g.workdir / "10_event_cleanup" / "cleaned.jsonl",
        g.workdir / "11_doc_cleanup" / "cleaned_final.json",
        g.out_path,
    ]
    for p in expected:
        assert p.exists(), f"missing artefact: {p}"

    # The .ass file is parsable by pysubs2 and contains at least one event
    subs = pysubs2.load(str(g.out_path), encoding="utf-8")
    assert len(subs.events) >= 1
    # First event has the expected cleaned text (pysubs2 unescapes \N in text)
    assert "final 0" in subs.events[0].text
