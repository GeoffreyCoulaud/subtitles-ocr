from fractions import Fraction
from pathlib import Path

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
    NoCacheKey,
    OcrConfig,
    PipelineConfig,
    PipelineGlobals,
    section_for,
)


def _make_globals(tmp_path: Path, fps: Fraction = Fraction(24000, 1001)) -> PipelineGlobals:
    return PipelineGlobals(
        workdir=tmp_path,
        hardsub_path=tmp_path / "hardsub.mkv",
        raw_path=tmp_path / "raw.mkv",
        out_path=tmp_path / "out.ass",
        fps=fps,
        fansub_width=1920,
        fansub_height=1080,
        fansub_total_frames=35000,
    )


def test_pipeline_globals_instantiable(tmp_path: Path) -> None:
    g = _make_globals(tmp_path)

    assert g.workdir == tmp_path
    assert g.fansub_width == 1920
    assert g.fansub_height == 1080
    assert g.fansub_total_frames == 35000
    assert g.debug_images is False
    assert g.fps == Fraction(24000, 1001)


def test_pipeline_globals_fps_accepts_int(tmp_path: Path) -> None:
    g = _make_globals(tmp_path, fps=24)  # type: ignore[arg-type]

    assert g.fps == Fraction(24, 1)
    assert isinstance(g.fps, Fraction)


def test_pipeline_globals_json_round_trip_preserves_fraction(tmp_path: Path) -> None:
    g = _make_globals(tmp_path, fps=Fraction(24000, 1001))

    payload = g.model_dump_json()
    restored = PipelineGlobals.model_validate_json(payload)

    assert restored.fps == Fraction(24000, 1001)
    assert isinstance(restored.fps, Fraction)
    assert restored.fps.numerator == 24000
    assert restored.fps.denominator == 1001


def test_pipeline_globals_json_round_trip_preserves_other_fractions(tmp_path: Path) -> None:
    g = _make_globals(tmp_path, fps=Fraction(30000, 1001))

    restored = PipelineGlobals.model_validate_json(g.model_dump_json())

    assert restored.fps == Fraction(30000, 1001)


def test_pipeline_config_instantiable_with_defaults() -> None:
    cfg = PipelineConfig()

    # ar_strategy / audio_tracks / skip_ranges no longer live at the root —
    # they belong to ConformConfig and AlignmentConfig respectively (issues 3 & 4).
    assert cfg.conform.ar_strategy == "error"
    assert cfg.alignment.hardsub_audio_track is None
    assert cfg.alignment.raw_audio_track is None
    assert cfg.alignment.hardsub_skip_ranges == []
    assert cfg.alignment.raw_skip_ranges == []
    assert cfg.export.color_cluster_threshold == 10.0
    assert cfg.doc_cleanup.synopsis_path is None
    assert isinstance(cfg.conform, ConformConfig)
    assert isinstance(cfg.alignment, AlignmentConfig)
    # FrameProcessingConfig is now owned by OcrConfig (issue 1 cleanup).
    assert isinstance(cfg.ocr.frame_processing, FrameProcessingConfig)
    assert isinstance(cfg.ocr, OcrConfig)
    assert isinstance(cfg.group, GroupConfig)
    assert isinstance(cfg.animation, AnimationConfig)
    assert isinstance(cfg.color, ColorConfig)
    assert isinstance(cfg.event_cleanup, EventCleanupConfig)
    assert isinstance(cfg.doc_cleanup, DocCleanupConfig)
    assert isinstance(cfg.export, ExportConfig)


def test_pipeline_config_has_no_root_ar_strategy() -> None:
    """Issue 3: ar_strategy must live only in ConformConfig, not on the root."""
    cfg = PipelineConfig()
    assert not hasattr(cfg, "ar_strategy")


def test_pipeline_config_has_no_root_audio_or_skip_ranges() -> None:
    """Issue 4: audio tracks and skip ranges live on AlignmentConfig only."""
    cfg = PipelineConfig()
    for name in (
        "hardsub_audio_track",
        "raw_audio_track",
        "hardsub_skip_ranges",
        "raw_skip_ranges",
    ):
        assert not hasattr(cfg, name), f"{name} must not be on PipelineConfig root"


def test_pipeline_config_has_no_root_frame_processing() -> None:
    """frame_processing is owned by OcrConfig now, not the root config."""
    cfg = PipelineConfig()
    assert not hasattr(cfg, "frame_processing")


def test_ocr_config_embeds_frame_processing() -> None:
    cfg = OcrConfig()
    assert isinstance(cfg.frame_processing, FrameProcessingConfig)


def test_alignment_config_audio_tracks_and_skip_ranges_defaults() -> None:
    cfg = AlignmentConfig()
    assert cfg.hardsub_audio_track is None
    assert cfg.raw_audio_track is None
    assert cfg.hardsub_skip_ranges == []
    assert cfg.raw_skip_ranges == []


def test_section_for_returns_correct_subconfig() -> None:
    cfg = PipelineConfig()

    class FakeOcrStage:
        CONFIG_FIELD = "ocr"

    class FakeGroupStage:
        CONFIG_FIELD = "group"

    assert section_for(cfg, FakeOcrStage()) is cfg.ocr
    assert section_for(cfg, FakeGroupStage()) is cfg.group


def test_no_cache_key_reexported_from_config() -> None:
    from subtitles_ocr.meta import NoCacheKey as MetaNoCacheKey

    assert NoCacheKey is MetaNoCacheKey


def test_subconfigs_are_pydantic_basemodels() -> None:
    for cls in (
        ConformConfig,
        AlignmentConfig,
        FrameProcessingConfig,
        OcrConfig,
        GroupConfig,
        AnimationConfig,
        ColorConfig,
        EventCleanupConfig,
        DocCleanupConfig,
        ExportConfig,
    ):
        assert issubclass(cls, BaseModel)
        # Empty skeletons — instantiable without arguments
        cls()
