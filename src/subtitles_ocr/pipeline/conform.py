"""Stage 1 — spatial conform (ADR-0002 §3 Stage 1)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import ConformConfig, PipelineGlobals
from subtitles_ocr.ffmpeg.protocol import FfmpegRunner

STAGE_VERSION: int = 1


class ConformResult(BaseModel):
    raw_conformed_path: Path
    target_width: int
    target_height: int
    pix_fmt: str


class ConformStage:
    CONFIG_FIELD: ClassVar[str] = "conform"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "hardsub_path",
        "raw_path",
        "fansub_width",
        "fansub_height",
    )

    def __init__(self, ffmpeg: FfmpegRunner | None = None) -> None:
        self.ffmpeg = ffmpeg

    def run(self, globals: PipelineGlobals, config: ConformConfig) -> ConformResult:
        raise NotImplementedError
