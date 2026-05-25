from fractions import Fraction
from pathlib import Path
from typing import Literal

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
    pass


class AlignmentConfig(BaseModel):
    pass


class FrameProcessingConfig(BaseModel):
    pass


class OcrConfig(BaseModel):
    pass


class GroupConfig(BaseModel):
    pass


class AnimationConfig(BaseModel):
    pass


class ColorConfig(BaseModel):
    pass


class EventCleanupConfig(BaseModel):
    pass


class DocCleanupConfig(BaseModel):
    pass


class ExportConfig(BaseModel):
    pass


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
