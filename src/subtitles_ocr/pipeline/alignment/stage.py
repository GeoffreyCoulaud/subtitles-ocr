"""Stage 2 — adaptive alignment (ADR-0002 §3 Stage 2)."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel

from subtitles_ocr.config import AlignmentConfig, PipelineGlobals
from subtitles_ocr.ffmpeg.protocol import FfmpegRunner

STAGE_VERSION: int = 1


class AlignmentSegment(BaseModel):
    fansub_frame_start: int
    fansub_frame_end: int
    raw_frame_start: int | None
    raw_frame_end: int | None
    offset_frames: int | None
    status: Literal["ALIGNED", "ORPHAN", "USER_SKIPPED"]
    confidence_avg: float | None


class AlignmentResult(BaseModel):
    fansub_total_frames: int
    raw_total_frames: int
    method_used: Literal["audio+phash_refinement", "phash_only"]
    aligned_ratio: float
    orphan_ratio: float
    user_skipped_ratio: float
    segments: list[AlignmentSegment]
    warnings: list[str]


class AlignmentStage:
    CONFIG_FIELD: ClassVar[str] = "alignment"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "hardsub_path",
        "raw_path",
        "fps",
        "fansub_total_frames",
    )

    def __init__(self, ffmpeg: FfmpegRunner | None = None) -> None:
        self.ffmpeg = ffmpeg

    def run(self, globals: PipelineGlobals, config: AlignmentConfig) -> AlignmentResult:
        raise NotImplementedError
