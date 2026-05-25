"""Stage 12 — ASS export (ADR-0002 §3 Stage 11, ADR-0003 §4.4)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import ExportConfig, PipelineGlobals

STAGE_VERSION: int = 1


class ExportResult(BaseModel):
    out_path_written: str
    event_count: int


class ExportStage:
    CONFIG_FIELD: ClassVar[str] = "export"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "out_path",
        "fps",
        "fansub_width",
        "fansub_height",
    )

    def __init__(self) -> None:
        pass

    def run(self, globals: PipelineGlobals, config: ExportConfig) -> ExportResult:
        raise NotImplementedError
