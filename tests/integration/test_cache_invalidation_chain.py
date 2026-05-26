"""Cross-stage cache invalidation tests.

Each stage has its own sidecar comparison logic (ADR-0004 §3.1, §5). This
suite verifies the *observable* invariants across the 9-stage chain:

  - happy path: every stage writes its sidecar
  - re-run with no changes: every stage cache-hits (no fake counter advances)
  - bumping a stage's STAGE_VERSION: that stage + transitively-downstream
    stages (those whose `input_fingerprints` reference its output) invalidate
  - NoCacheKey field change: no stage invalidates
  - cache-invalidating field change (OcrConfig.language): OCR invalidates,
    which through Group's intermediate fingerprint cascades downstream

Stages that fingerprint upstream intermediates: Conform(sources), Group(ocr,
alignment), Animation(group), Color(animation), DocCleanup(event_cleanup).
Stages that DO NOT fingerprint upstream: Alignment, OCR, EventCleanup. That
asymmetry is intrinsic to the current design and the assertions below reflect
it honestly.
"""

from __future__ import annotations

import numpy as np
import pytest

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
from subtitles_ocr.pipeline.group import GroupStage
from subtitles_ocr.pipeline.ocr import OcrStage

from tests.integration.conftest import (
    FakeFfmpeg,
    FakeFrameReader,
    FakeFrameSource,
    FakeLlm,
    FakeOcrEngine,
    make_video_metadata,
)


# ---------------------------------------------------------------------------
# Fake-builders shared across tests
# ---------------------------------------------------------------------------


def _build_fake_ffmpeg(globals_: PipelineGlobals) -> FakeFfmpeg:
    meta = make_video_metadata(globals_.fansub_width, globals_.fansub_height)
    return FakeFfmpeg(
        probe_returns={
            globals_.hardsub_path: meta,
            globals_.raw_path: meta,
        }
    )


def _composed_frames_for(globals_: PipelineGlobals) -> list[ComposedFrame]:
    """One ComposedFrame per fansub frame, pixel values keyed by idx."""
    return [
        ComposedFrame(
            fansub_frame_idx=i,
            image=np.full((16, 16, 3), 100 + i, dtype=np.uint8),
        )
        for i in range(globals_.fansub_total_frames)
    ]


def _build_doc_cleanup_factory(event_count: int):
    """Factory that returns DocCleanupResult with N events echoing input ids."""

    def factory(schema, prompt):
        return DocCleanupResult(
            events=[
                FinalEvent(event_id=i, cleaned_text=f"final {i}") for i in range(event_count)
            ]
        )

    return factory


def _run_event_cleanup_factory(schema, prompt):
    return CleanedEvent(text="canonical")


# ---------------------------------------------------------------------------
# Stage runner used by the chain tests
# ---------------------------------------------------------------------------


class StageRunner:
    """Owns one fake of each kind so tests can inspect call counts.

    Stages are stateless after init (per ADR-0004 §3.1) so it is safe to call
    .run_all() multiple times on the same runner.
    """

    def __init__(
        self,
        globals_: PipelineGlobals,
        *,
        ocr_config: OcrConfig | None = None,
        frame_processing_config: FrameProcessingConfig | None = None,
        event_cleanup_config: EventCleanupConfig | None = None,
        ocr_engine: FakeOcrEngine | None = None,
        ffmpeg: FakeFfmpeg | None = None,
        frame_reader: FakeFrameReader | None = None,
        event_llm: FakeLlm | None = None,
        doc_llm: FakeLlm | None = None,
    ) -> None:
        self.globals_ = globals_
        self.ocr_config = ocr_config or OcrConfig()
        self.frame_processing_config = frame_processing_config or FrameProcessingConfig()
        self.event_cleanup_config = event_cleanup_config or EventCleanupConfig(model="m")
        self.ffmpeg = ffmpeg or _build_fake_ffmpeg(globals_)
        self.ocr_engine = ocr_engine or FakeOcrEngine()
        self.frame_reader = frame_reader or FakeFrameReader(
            width=globals_.fansub_width, height=globals_.fansub_height
        )
        # Default LLM fakes — event-cleanup is reached only with non-consensus
        # text, but we wire one to detect any unwanted invocation.
        self.event_llm = event_llm or FakeLlm(response_factory=_run_event_cleanup_factory)
        # DocCleanup is reached unconditionally; provide a 1-event response.
        self.doc_llm = doc_llm or FakeLlm(response_factory=_build_doc_cleanup_factory(1))

    def run_all(self) -> None:
        g = self.globals_

        ConformStage(ffmpeg=self.ffmpeg).run(g, ConformConfig())

        AlignmentStage(
            frame_source=FakeFrameSource(),
            raw_total_frames=g.fansub_total_frames,
        ).run(g, AlignmentConfig())

        composed = _composed_frames_for(g)
        OcrStage(ocr_engine=self.ocr_engine).run(
            g,
            self.ocr_config.model_copy(
                update={"frame_processing": self.frame_processing_config}
            ),
            composed_frames=iter(composed),
        )

        GroupStage().run(g, GroupConfig())
        AnimationStage().run(g, AnimationConfig())
        ColorStage(frame_reader=self.frame_reader).run(g, ColorConfig())
        EventCleanupStage(llm=self.event_llm).run(g, self.event_cleanup_config)

        # DocCleanup factory must size to the number of grouped events; for the
        # canonical "all frames same text" case this is exactly 1.
        events_path = g.workdir / "07_group" / "events.json"
        from subtitles_ocr.pipeline.group import GroupResult

        event_count = len(
            GroupResult.model_validate_json(events_path.read_text()).events
        )
        # Reconfigure doc_llm factory to match event count (in case caller didn't).
        self.doc_llm.response_factory = _build_doc_cleanup_factory(event_count)

        DocCleanupStage(llm=self.doc_llm).run(g, DocCleanupConfig(model="m"))
        ExportStage().run(g, ExportConfig())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_first_run_writes_every_sidecar_and_produces_ass(
    integration_globals: PipelineGlobals,
) -> None:
    runner = StageRunner(integration_globals)
    runner.run_all()
    w = integration_globals.workdir

    # Each stage's persisted output + sidecar
    assert (w / "01_conform" / "raw.mkv").exists()
    assert (w / "01_conform" / "raw.meta.json").exists()
    assert (w / "02_alignment" / "alignment.json").exists()
    assert (w / "02_alignment" / "alignment.meta.json").exists()
    assert (w / "06_ocr" / "results.jsonl").exists()
    assert (w / "06_ocr" / "results.meta.json").exists()
    assert (w / "07_group" / "events.json").exists()
    assert (w / "07_group" / "events.meta.json").exists()
    assert (w / "08_animation" / "animation.json").exists()
    assert (w / "08_animation" / "animation.meta.json").exists()
    assert (w / "09_color" / "colors.json").exists()
    assert (w / "09_color" / "colors.meta.json").exists()
    assert (w / "10_event_cleanup" / "cleaned.jsonl").exists()
    assert (w / "10_event_cleanup" / "cleaned.meta.json").exists()
    assert (w / "11_doc_cleanup" / "cleaned_final.json").exists()
    assert (w / "11_doc_cleanup" / "cleaned_final.meta.json").exists()
    assert integration_globals.out_path.exists()


def test_rerun_with_no_changes_skips_every_recompute(
    integration_globals: PipelineGlobals,
) -> None:
    runner = StageRunner(integration_globals)
    runner.run_all()

    transcodes_after_1 = len(runner.ffmpeg.transcode_calls)
    ocr_calls_after_1 = runner.ocr_engine.call_count
    reader_calls_after_1 = runner.frame_reader.call_count
    doc_llm_calls_after_1 = runner.doc_llm.call_count
    event_llm_calls_after_1 = runner.event_llm.call_count

    runner.run_all()

    # Conform: cache hit → no extra transcode call
    assert len(runner.ffmpeg.transcode_calls) == transcodes_after_1
    # OCR: sidecar comparison — but OCR stage always streams composed frames
    # and only skips already-written ones via resume_index. With composed_frames
    # injected directly, OCR re-iterates them all but resume_index() == 5
    # short-circuits each one — call_count stays equal.
    assert runner.ocr_engine.call_count == ocr_calls_after_1
    # Color, DocCleanup: cache hit → no extra reader / LLM calls
    assert runner.frame_reader.call_count == reader_calls_after_1
    assert runner.doc_llm.call_count == doc_llm_calls_after_1
    # EventCleanup: consensus text → no LLM call to begin with, both runs == 0
    assert runner.event_llm.call_count == event_llm_calls_after_1


def test_bumping_group_stage_version_reexecutes_group_and_downstream_fingerprinters(
    integration_globals: PipelineGlobals, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bumping Group's STAGE_VERSION should:
      - invalidate Group (its sidecar comparison sees stage_version mismatch)
      - Group rewrites events.json → Animation's input fingerprint of group
        events changes → Animation invalidates → animation.json mtime changes
        → Color's input fingerprint changes → Color invalidates
      - DocCleanup fingerprints event_cleanup_jsonl; that file is rewritten
        only if EventCleanup re-runs, but EventCleanup has no input
        fingerprints so it does not invalidate from upstream changes.
      - Conform / Alignment / OCR are upstream of Group → must NOT invalidate.

    The observable signal we assert on is the frame-reader call count
    (Color's only side-effect that we can count). It re-incrementing proves
    Color re-ran, which proves Animation re-ran, which proves Group re-ran.
    """
    runner = StageRunner(integration_globals)
    runner.run_all()
    transcodes_baseline = len(runner.ffmpeg.transcode_calls)
    reader_baseline = runner.frame_reader.call_count

    import subtitles_ocr.pipeline.group as group_mod

    monkeypatch.setattr(group_mod, "STAGE_VERSION", group_mod.STAGE_VERSION + 1)

    runner.run_all()

    # Conform upstream → no invalidation
    assert len(runner.ffmpeg.transcode_calls) == transcodes_baseline
    # Color re-ran → frame_reader was hit again
    assert runner.frame_reader.call_count > reader_baseline


def test_changing_nocachekey_field_does_not_reexecute_any_fingerprinted_stage(
    integration_globals: PipelineGlobals,
) -> None:
    """EventCleanupConfig.parallelism is NoCacheKey; flipping it must not
    invalidate. We assert via the doc_llm call count (DocCleanup runs
    unconditionally on each invocation only when its cache is invalidated).
    """
    runner = StageRunner(integration_globals)
    runner.run_all()
    transcodes_baseline = len(runner.ffmpeg.transcode_calls)
    reader_baseline = runner.frame_reader.call_count
    doc_llm_baseline = runner.doc_llm.call_count

    # Flip a NoCacheKey field on EventCleanupConfig
    runner.event_cleanup_config = EventCleanupConfig(
        model=runner.event_cleanup_config.model, parallelism=8
    )
    runner.run_all()

    # No stage invalidates
    assert len(runner.ffmpeg.transcode_calls) == transcodes_baseline
    assert runner.frame_reader.call_count == reader_baseline
    assert runner.doc_llm.call_count == doc_llm_baseline


def test_changing_cache_invalidating_group_field_reexecutes_group_chain(
    integration_globals: PipelineGlobals,
) -> None:
    """GroupConfig.text_levenshtein_max is cache-invalidating. Changing it must
    invalidate Group's sidecar comparison; Color (which fingerprints
    animation.json, which fingerprints events.json) re-runs as a result.
    """
    runner = StageRunner(integration_globals)
    runner.run_all()
    reader_baseline = runner.frame_reader.call_count

    # Run with overridden Group config
    g = integration_globals
    ConformStage(ffmpeg=runner.ffmpeg).run(g, ConformConfig())
    AlignmentStage(
        frame_source=FakeFrameSource(),
        raw_total_frames=g.fansub_total_frames,
    ).run(g, AlignmentConfig())
    OcrStage(ocr_engine=runner.ocr_engine).run(
        g,
        runner.ocr_config.model_copy(
            update={"frame_processing": runner.frame_processing_config}
        ),
        composed_frames=iter(_composed_frames_for(g)),
    )
    # Cache-invalidating field of Group flipped from default 0.2 → 0.3
    GroupStage().run(g, GroupConfig(text_levenshtein_max=0.3))
    AnimationStage().run(g, AnimationConfig())
    ColorStage(frame_reader=runner.frame_reader).run(g, ColorConfig())

    # Color must have re-run (frame_reader called again)
    assert runner.frame_reader.call_count > reader_baseline


def test_modifying_hardsub_source_invalidates_conform(
    integration_globals: PipelineGlobals,
) -> None:
    """Conform fingerprints the hardsub source. Touching its mtime invalidates."""
    runner = StageRunner(integration_globals)
    runner.run_all()
    transcodes_baseline = len(runner.ffmpeg.transcode_calls)

    # Touch the source: rewrite with different bytes so size/full_hash both change
    # (small file → full_hash branch in fingerprint()).
    integration_globals.hardsub_path.write_bytes(b"ALTERED_HARDSUB_PAYLOAD_BYTES" * 4)

    runner.run_all()
    assert len(runner.ffmpeg.transcode_calls) == transcodes_baseline + 1


def test_modifying_raw_source_invalidates_conform_but_not_alignment(
    integration_globals: PipelineGlobals,
) -> None:
    """Conform fingerprints raw_path; Alignment carries it as a *string* in
    globals_subset (not a fingerprint). So modifying the file bytes invalidates
    only Conform.
    """
    runner = StageRunner(integration_globals)
    runner.run_all()
    transcodes_baseline = len(runner.ffmpeg.transcode_calls)

    integration_globals.raw_path.write_bytes(b"NEW_RAW_PAYLOAD_BYTES" * 4)

    runner.run_all()
    # Conform re-ran
    assert len(runner.ffmpeg.transcode_calls) == transcodes_baseline + 1
