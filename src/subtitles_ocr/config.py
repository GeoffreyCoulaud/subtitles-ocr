"""Pipeline configuration root + per-stage sub-configs.

Design note (import cycle avoidance):
    Per ADR-0004 §12 each stage exposes its Result schema in its own module.
    The corresponding per-stage `XConfig` model could in principle live alongside
    the stage code, but `config.py` is imported by `cli.py` and the stage modules
    themselves import `PipelineGlobals` from here for type hints. Defining the
    sub-configs in stage modules would force `config.py` to import them back to
    assemble `PipelineConfig`, creating a cycle. We therefore keep every
    sub-config in `config.py` and let stage modules import their config class
    from here. The schemas (Result, sub-types) still live in the stage modules,
    per ADR-0004 §12.
"""

from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from subtitles_ocr.meta import NoCacheKey

__all__ = [
    "AlignmentConfig",
    "AnimationConfig",
    "ColorConfig",
    "ConformConfig",
    "DocCleanupConfig",
    "EventCleanupConfig",
    "ExportConfig",
    "FrameProcessingConfig",
    "GroupConfig",
    "NoCacheKey",
    "OcrConfig",
    "PipelineConfig",
    "PipelineGlobals",
    "section_for",
]


class PipelineGlobals(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workdir: Path
    hardsub_path: Path
    raw_path: Path
    out_path: Path
    fps: Fraction
    fansub_width: int
    fansub_height: int
    fansub_total_frames: int
    debug_images: bool = False

    @field_serializer("fps")
    def _serialize_fps(self, value: Fraction) -> str:
        return f"{value.numerator}/{value.denominator}"

    @field_validator("fps", mode="before")
    @classmethod
    def _validate_fps(cls, value: object) -> Fraction:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, str):
            return Fraction(value)
        if isinstance(value, int):
            return Fraction(value)
        raise TypeError(f"fps must be Fraction, str, or int; got {type(value).__name__}")


class ConformConfig(BaseModel):
    # ADR-0002 §3 Stage 1. ar_strategy governs the AR-mismatch policy between
    # fansub and raw; "error" (default) is the conservative choice to surface
    # mismatched pairs explicitly before they corrupt downstream alignment.
    ar_strategy: Literal["error", "letterbox", "crop"] = "error"
    output_codec: str = "ffv1"
    output_container: str = "mkv"


class AlignmentConfig(BaseModel):
    # Sub-stage 2a — audio coarse alignment (baselines, à tuner)
    audio_thresh_low: float = 0.20  # baseline, à tuner
    audio_thresh_high: float = 0.50  # baseline, à tuner
    audio_thresh_snr: float = 3.0  # baseline, à tuner
    min_match_s: float = 1.0  # baseline, à tuner
    offset_tolerance_frames: int = 2  # baseline, à tuner
    # Sub-stage 2b — phash refinement (defaults from ADR-0002 §9)
    thresh_agree: int = 10
    threshold_disagree: float = 0.30
    # Sub-stage 2c — phash fallback (per ADR-0001 §17)
    w_initial: int = 4  # baseline, à tuner
    w_min: int = 2  # baseline, à tuner
    w_max: int = 16  # baseline, à tuner
    grow_step: int = 2  # baseline, à tuner
    shrink_step: int = 1  # baseline, à tuner
    thresh_match: int = 12  # baseline, à tuner
    # Failure policy (ADR-0002 §3 Stage 2)
    orphan_ratio_max: float = 0.30


class FrameProcessingConfig(BaseModel):
    # Stage 3 — diff (ADR-0002 §9, baselines à tuner)
    lcn_sigma: float = 1.5  # baseline, à tuner
    std_floor: float = 1e-3  # baseline, à tuner
    # Stage 4 — mask formation
    mask_smoothing_sigma: float = 1.0  # baseline, à tuner
    mask_t_high: float = 0.30  # baseline, à tuner
    mask_t_low: float = 0.10  # baseline, à tuner
    mask_area_min: int = 20  # baseline, à tuner
    mask_area_max: int = 500_000  # baseline, à tuner
    mask_dilation_iter: int = 1


class OcrConfig(BaseModel):
    language: str = "latin"
    device: Annotated[Literal["auto", "cuda", "rocm", "cpu"], NoCacheKey] = "auto"
    parallelism: Annotated[int, NoCacheKey] = 1
    chunk_size: int = 500


class GroupConfig(BaseModel):
    text_levenshtein_max: float = 0.2
    quad_iou_min: float = 0.5


class AnimationConfig(BaseModel):
    # All defaults are baselines from ADR-0003 §8 (placeholder, to tune).
    min_move_displacement_px: int = 8
    move_gap_tolerance_ms: int = 200
    move_r2_threshold: float = 0.95
    move_text_levenshtein_max: float = 0.2
    fade_search_window_ms: int = 1250
    min_fade_duration_ms: int = 125
    fade_duration_cap_ms: int = 1000
    fade_score_fit_range: tuple[float, float] = (0.05, 0.95)
    fade_fit_r2_threshold: float = 0.7


class ColorConfig(BaseModel):
    interior_hue_var_max: float = 0.05  # baseline, à tuner
    outline_hue_var_max: float = 0.05  # baseline, à tuner
    interior_min_pixels: int = 100
    pool_ratio_min: float = 0.1
    pool_ratio_max: float = 10.0
    crop_padding_pct: float = 0.10
    erosion_factor: float = 0.4
    hsv_bins: int = 16


class EventCleanupConfig(BaseModel):
    model: str | None = None
    parallelism: Annotated[int, NoCacheKey] = 4
    chunk_size: int = 100


class DocCleanupConfig(BaseModel):
    model: str | None = None
    parallelism: Annotated[int, NoCacheKey] = 1
    synopsis_path: Path | None = None


class ExportConfig(BaseModel):
    default_font: str = "Arial"
    default_font_size: int = 60


class PipelineConfig(BaseModel):
    ar_strategy: Literal["error", "letterbox", "crop"] = "error"
    synopsis_path: Path | None = None
    color_cluster_threshold: float = 10.0
    hardsub_audio_track: int | None = None
    raw_audio_track: int | None = None
    hardsub_skip_ranges: list[str] = Field(default_factory=list)
    raw_skip_ranges: list[str] = Field(default_factory=list)

    conform: ConformConfig = Field(default_factory=ConformConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    frame_processing: FrameProcessingConfig = Field(default_factory=FrameProcessingConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    group: GroupConfig = Field(default_factory=GroupConfig)
    animation: AnimationConfig = Field(default_factory=AnimationConfig)
    color: ColorConfig = Field(default_factory=ColorConfig)
    event_cleanup: EventCleanupConfig = Field(default_factory=EventCleanupConfig)
    doc_cleanup: DocCleanupConfig = Field(default_factory=DocCleanupConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


def section_for(config: PipelineConfig, stage: object) -> BaseModel:
    field_name = stage.CONFIG_FIELD
    return getattr(config, field_name)
