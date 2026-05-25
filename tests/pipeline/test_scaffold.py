"""Scaffold tests: every pipeline stage module imports, has STAGE_VERSION = 1,
exposes its Pydantic schemas with JSON round-trip, and its Stage class raises
NotImplementedError from `run()`. Plus a representative check on PipelineConfig
sub-config defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args, get_origin

import pytest
from pydantic import BaseModel

from subtitles_ocr.config import (
    AlignmentConfig,
    AnimationConfig,
    ColorConfig,
    ConformConfig,
    DocCleanupConfig,
    EventCleanupConfig,
    ExportConfig,
    FrameProcessingConfig,
    GroupConfig,
    OcrConfig,
    PipelineConfig,
)
from subtitles_ocr.meta import NoCacheKey


# -------------------- conform --------------------

def test_conform_module_imports() -> None:
    from subtitles_ocr.pipeline import conform  # noqa: F401

    assert conform.STAGE_VERSION == 1


def test_conform_result_round_trip() -> None:
    from subtitles_ocr.pipeline.conform import ConformResult

    r = ConformResult(
        raw_conformed_path=Path("/tmp/raw.mkv"),
        target_width=1440,
        target_height=1080,
        pix_fmt="yuv420p",
    )
    restored = ConformResult.model_validate_json(r.model_dump_json())
    assert restored.target_width == 1440
    assert restored.target_height == 1080
    assert restored.pix_fmt == "yuv420p"


def test_conform_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.conform import ConformStage

    stage = ConformStage()
    assert stage.CONFIG_FIELD == "conform"
    assert isinstance(stage.GLOBALS_USED, tuple)
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, ConformConfig())


# -------------------- alignment --------------------

def test_alignment_module_imports() -> None:
    from subtitles_ocr.pipeline.alignment import stage as alignment_stage_mod

    assert alignment_stage_mod.STAGE_VERSION == 1


def test_alignment_segment_round_trip() -> None:
    from subtitles_ocr.pipeline.alignment.stage import AlignmentSegment

    seg = AlignmentSegment(
        fansub_frame_start=0,
        fansub_frame_end=100,
        raw_frame_start=10,
        raw_frame_end=110,
        offset_frames=10,
        status="ALIGNED",
        confidence_avg=0.9,
    )
    restored = AlignmentSegment.model_validate_json(seg.model_dump_json())
    assert restored.status == "ALIGNED"
    assert restored.offset_frames == 10


def test_alignment_result_round_trip() -> None:
    from subtitles_ocr.pipeline.alignment.stage import (
        AlignmentResult,
        AlignmentSegment,
    )

    r = AlignmentResult(
        fansub_total_frames=100,
        raw_total_frames=120,
        method_used="phash_only",
        aligned_ratio=0.95,
        orphan_ratio=0.05,
        user_skipped_ratio=0.0,
        segments=[
            AlignmentSegment(
                fansub_frame_start=0,
                fansub_frame_end=100,
                raw_frame_start=10,
                raw_frame_end=110,
                offset_frames=10,
                status="ALIGNED",
                confidence_avg=0.9,
            )
        ],
        warnings=[],
    )
    restored = AlignmentResult.model_validate_json(r.model_dump_json())
    assert restored.method_used == "phash_only"
    assert restored.aligned_ratio == 0.95
    assert len(restored.segments) == 1


def test_alignment_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.alignment import AlignmentStage

    stage = AlignmentStage()
    assert stage.CONFIG_FIELD == "alignment"
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, AlignmentConfig())


# -------------------- frame_processing --------------------

def test_frame_processing_module_imports() -> None:
    from subtitles_ocr.pipeline.frame_processing import iterator as it_mod

    assert it_mod.STAGE_VERSION == 1


def test_composed_frame_is_frozen_dataclass() -> None:
    import dataclasses

    import numpy as np

    from subtitles_ocr.pipeline.frame_processing import ComposedFrame

    cf = ComposedFrame(fansub_frame_idx=42, image=np.zeros((4, 4, 3), dtype=np.uint8))
    assert dataclasses.is_dataclass(cf)
    assert cf.fansub_frame_idx == 42
    with pytest.raises(dataclasses.FrozenInstanceError):
        cf.fansub_frame_idx = 0  # type: ignore[misc]


def test_iter_composed_frames_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.alignment.stage import AlignmentResult
    from subtitles_ocr.pipeline.frame_processing import iter_composed_frames

    alignment = AlignmentResult(
        fansub_total_frames=10,
        raw_total_frames=10,
        method_used="phash_only",
        aligned_ratio=1.0,
        orphan_ratio=0.0,
        user_skipped_ratio=0.0,
        segments=[],
        warnings=[],
    )
    with pytest.raises(NotImplementedError):
        # Iterator must be consumed to trigger the body
        list(iter_composed_frames(mock_globals, alignment, FrameProcessingConfig()))


# -------------------- ocr --------------------

def test_ocr_module_imports() -> None:
    from subtitles_ocr.pipeline import ocr

    assert ocr.STAGE_VERSION == 1


def test_ocr_detection_round_trip() -> None:
    from subtitles_ocr.pipeline.ocr import OcrDetection

    d = OcrDetection(
        text="hello",
        confidence=0.95,
        quad=[(0, 0), (10, 0), (10, 5), (0, 5)],
    )
    restored = OcrDetection.model_validate_json(d.model_dump_json())
    assert restored.text == "hello"
    assert restored.quad == [(0, 0), (10, 0), (10, 5), (0, 5)]


def test_frame_ocr_result_round_trip() -> None:
    from subtitles_ocr.pipeline.ocr import FrameOcrResult, OcrDetection

    r = FrameOcrResult(
        fansub_frame_idx=12,
        detections=[OcrDetection(text="a", confidence=0.5, quad=[(0, 0), (1, 0), (1, 1), (0, 1)])],
    )
    restored = FrameOcrResult.model_validate_json(r.model_dump_json())
    assert restored.fansub_frame_idx == 12
    assert restored.detections[0].text == "a"


def test_ocr_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.ocr import OcrStage

    stage = OcrStage(ocr_engine=object())  # avoid instantiating real paddle engine
    assert stage.CONFIG_FIELD == "ocr"
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, OcrConfig())


# -------------------- group --------------------

def test_group_module_imports() -> None:
    from subtitles_ocr.pipeline import group

    assert group.STAGE_VERSION == 1


def test_subtitle_event_round_trip() -> None:
    from subtitles_ocr.pipeline.group import SubtitleEvent

    ev = SubtitleEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=50,
        raw_ocr_texts=["hi", "hi"],
        raw_ocr_confidences=[0.9, 0.91],
        quads_per_frame={
            0: [(0, 0), (10, 0), (10, 5), (0, 5)],
            1: [(0, 0), (10, 0), (10, 5), (0, 5)],
        },
        quad_median=[(0, 0), (10, 0), (10, 5), (0, 5)],
        member_frame_indices=[0, 1],
    )
    restored = SubtitleEvent.model_validate_json(ev.model_dump_json())
    assert restored.event_id == 0
    assert restored.quads_per_frame[0] == [(0, 0), (10, 0), (10, 5), (0, 5)]
    assert restored.quad_median == [(0, 0), (10, 0), (10, 5), (0, 5)]


def test_group_result_round_trip() -> None:
    from subtitles_ocr.pipeline.group import GroupResult, SubtitleEvent

    r = GroupResult(
        fansub_total_frames=100,
        events=[
            SubtitleEvent(
                event_id=0,
                fansub_frame_start=0,
                fansub_frame_end=2,
                raw_ocr_texts=["x"],
                raw_ocr_confidences=[0.5],
                quads_per_frame={0: [(0, 0), (1, 0), (1, 1), (0, 1)]},
                quad_median=[(0, 0), (1, 0), (1, 1), (0, 1)],
                member_frame_indices=[0],
            )
        ],
        stats={"events": 1},
    )
    restored = GroupResult.model_validate_json(r.model_dump_json())
    assert restored.fansub_total_frames == 100
    assert len(restored.events) == 1


def test_group_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.group import GroupStage

    stage = GroupStage()
    assert stage.CONFIG_FIELD == "group"
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, GroupConfig())


# -------------------- animation --------------------

def test_animation_module_imports() -> None:
    from subtitles_ocr.pipeline import animation

    assert animation.STAGE_VERSION == 1


def test_animated_event_round_trip() -> None:
    from subtitles_ocr.pipeline.animation import AnimatedEvent

    ev = AnimatedEvent(
        event_id=0,
        fansub_frame_start=0,
        fansub_frame_end=10,
        raw_ocr_texts=["t"],
        raw_ocr_confidences=[0.9],
        quads_per_frame={0: [(0, 0), (1, 0), (1, 1), (0, 1)]},
        quad_median=[(0, 0), (1, 0), (1, 1), (0, 1)],
        member_frame_indices=[0],
        motion={"type": "linear", "start": (0, 0), "end": (100, 0)},
        fade_in_ms=200,
        fade_out_ms=0,
    )
    restored = AnimatedEvent.model_validate_json(ev.model_dump_json())
    assert restored.fade_in_ms == 200
    assert restored.motion is not None and restored.motion["type"] == "linear"


def test_animation_result_round_trip() -> None:
    from subtitles_ocr.pipeline.animation import (
        AnimatedEvent,
        AnimationAnalysisResult,
    )

    r = AnimationAnalysisResult(
        events=[
            AnimatedEvent(
                event_id=0,
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
        stats={"static": 1},
    )
    restored = AnimationAnalysisResult.model_validate_json(r.model_dump_json())
    assert len(restored.events) == 1


def test_animation_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.animation import AnimationStage

    stage = AnimationStage()
    assert stage.CONFIG_FIELD == "animation"
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, AnimationConfig())


# -------------------- color --------------------

def test_color_module_imports() -> None:
    from subtitles_ocr.pipeline import color

    assert color.STAGE_VERSION == 1


def test_event_colors_round_trip() -> None:
    from subtitles_ocr.pipeline.color import EventColors

    c = EventColors(
        event_id=3,
        fill_color=(255, 255, 255),
        outline_color=(0, 0, 0),
        style_supported=True,
        stroke_width_px=2.5,
    )
    restored = EventColors.model_validate_json(c.model_dump_json())
    assert restored.fill_color == (255, 255, 255)
    assert restored.outline_color == (0, 0, 0)
    assert restored.style_supported is True


def test_color_extraction_result_round_trip() -> None:
    from subtitles_ocr.pipeline.color import ColorExtractionResult, EventColors

    r = ColorExtractionResult(
        events=[
            EventColors(
                event_id=0,
                fill_color=None,
                outline_color=None,
                style_supported=False,
                stroke_width_px=0.0,
            )
        ],
        stats={"supported": 0},
    )
    restored = ColorExtractionResult.model_validate_json(r.model_dump_json())
    assert restored.events[0].style_supported is False


def test_color_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.color import ColorStage

    stage = ColorStage()
    assert stage.CONFIG_FIELD == "color"
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, ColorConfig())


# -------------------- event_cleanup --------------------

def test_event_cleanup_module_imports() -> None:
    from subtitles_ocr.pipeline import event_cleanup

    assert event_cleanup.STAGE_VERSION == 1


def test_cleaned_event_round_trip() -> None:
    from subtitles_ocr.pipeline.event_cleanup import CleanedEvent

    ce = CleanedEvent(text="hello world")
    restored = CleanedEvent.model_validate_json(ce.model_dump_json())
    assert restored.text == "hello world"


def test_event_cleanup_item_round_trip() -> None:
    from subtitles_ocr.pipeline.event_cleanup import EventCleanupItem

    it = EventCleanupItem(event_id=5, cleaned_text="ok", skipped_llm=True)
    restored = EventCleanupItem.model_validate_json(it.model_dump_json())
    assert restored.event_id == 5
    assert restored.skipped_llm is True


def test_event_cleanup_result_round_trip() -> None:
    from subtitles_ocr.pipeline.event_cleanup import (
        EventCleanupItem,
        EventCleanupResult,
    )

    r = EventCleanupResult(items=[EventCleanupItem(event_id=0, cleaned_text="x", skipped_llm=False)])
    restored = EventCleanupResult.model_validate_json(r.model_dump_json())
    assert len(restored.items) == 1


def test_event_cleanup_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.event_cleanup import EventCleanupStage

    stage = EventCleanupStage(llm=object())
    assert stage.CONFIG_FIELD == "event_cleanup"
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, EventCleanupConfig())


# -------------------- doc_cleanup --------------------

def test_doc_cleanup_module_imports() -> None:
    from subtitles_ocr.pipeline import doc_cleanup

    assert doc_cleanup.STAGE_VERSION == 1


def test_final_event_round_trip() -> None:
    from subtitles_ocr.pipeline.doc_cleanup import FinalEvent

    fe = FinalEvent(event_id=7, cleaned_text="hi")
    restored = FinalEvent.model_validate_json(fe.model_dump_json())
    assert restored.event_id == 7
    assert restored.cleaned_text == "hi"


def test_doc_cleanup_result_round_trip() -> None:
    from subtitles_ocr.pipeline.doc_cleanup import DocCleanupResult, FinalEvent

    r = DocCleanupResult(events=[FinalEvent(event_id=0, cleaned_text="x")])
    restored = DocCleanupResult.model_validate_json(r.model_dump_json())
    assert len(restored.events) == 1


def test_doc_cleanup_stage_construction() -> None:
    # Scaffold sentinel retired: DocCleanupStage is implemented and covered by
    # tests/pipeline/test_doc_cleanup.py. We keep a lightweight construction
    # check here to preserve the scaffold's "every stage's CONFIG_FIELD is
    # wired" invariant.
    from subtitles_ocr.pipeline.doc_cleanup import DocCleanupStage

    stage = DocCleanupStage(llm=object())
    assert stage.CONFIG_FIELD == "doc_cleanup"


# -------------------- export --------------------

def test_export_module_imports() -> None:
    from subtitles_ocr.pipeline import export

    assert export.STAGE_VERSION == 1


def test_export_stage_run_raises_not_implemented(mock_globals) -> None:
    from subtitles_ocr.pipeline.export import ExportStage

    stage = ExportStage()
    assert stage.CONFIG_FIELD == "export"
    with pytest.raises(NotImplementedError):
        stage.run(mock_globals, ExportConfig())


# -------------------- PipelineConfig sub-config representative defaults --------------------

def test_pipeline_config_ocr_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.ocr.language == "latin"
    assert cfg.ocr.device == "auto"
    assert cfg.ocr.parallelism == 1
    assert cfg.ocr.chunk_size == 500


def test_pipeline_config_group_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.group.text_levenshtein_max == 0.2
    assert cfg.group.quad_iou_min == 0.5


def test_pipeline_config_animation_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.animation.min_move_displacement_px == 8
    assert cfg.animation.move_gap_tolerance_ms == 200
    assert cfg.animation.move_r2_threshold == 0.95
    assert cfg.animation.move_text_levenshtein_max == 0.2
    assert cfg.animation.fade_search_window_ms == 1250
    assert cfg.animation.min_fade_duration_ms == 125
    assert cfg.animation.fade_duration_cap_ms == 1000
    assert cfg.animation.fade_score_fit_range == (0.05, 0.95)
    assert cfg.animation.fade_fit_r2_threshold == 0.7


def test_pipeline_config_color_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.color.interior_min_pixels == 100
    assert cfg.color.pool_ratio_min == 0.1
    assert cfg.color.pool_ratio_max == 10.0
    assert cfg.color.crop_padding_pct == 0.10
    assert cfg.color.erosion_factor == 0.4
    assert cfg.color.hsv_bins == 16


def test_pipeline_config_event_cleanup_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.event_cleanup.model is None
    assert cfg.event_cleanup.parallelism == 4
    assert cfg.event_cleanup.chunk_size == 100


def test_pipeline_config_doc_cleanup_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.doc_cleanup.model is None
    assert cfg.doc_cleanup.parallelism == 1


def test_pipeline_config_export_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.export.default_font == "Arial"
    assert cfg.export.default_font_size == 60


def test_pipeline_config_frame_processing_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.frame_processing.mask_dilation_iter == 1


def test_pipeline_config_alignment_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.alignment.thresh_agree == 10
    assert cfg.alignment.threshold_disagree == 0.30
    assert cfg.alignment.orphan_ratio_max == 0.30


# -------------------- NoCacheKey on device/parallelism --------------------

def _has_no_cache_key_metadata(field) -> bool:
    return any(m is NoCacheKey or isinstance(m, NoCacheKey) for m in field.metadata)


def test_ocr_device_and_parallelism_are_no_cache_key() -> None:
    fields = OcrConfig.model_fields
    assert _has_no_cache_key_metadata(fields["device"])
    assert _has_no_cache_key_metadata(fields["parallelism"])


def test_event_cleanup_parallelism_is_no_cache_key() -> None:
    fields = EventCleanupConfig.model_fields
    assert _has_no_cache_key_metadata(fields["parallelism"])


def test_doc_cleanup_parallelism_is_no_cache_key() -> None:
    fields = DocCleanupConfig.model_fields
    assert _has_no_cache_key_metadata(fields["parallelism"])
