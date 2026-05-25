"""Stage 9 — per-event color extraction (ADR-0002 §3 Stage 8, ADR-0003 §4.3)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import ColorConfig, PipelineGlobals

STAGE_VERSION: int = 1


class EventColors(BaseModel):
    event_id: int
    fill_color: tuple[int, int, int] | None
    outline_color: tuple[int, int, int] | None
    style_supported: bool
    stroke_width_px: float


class ColorExtractionResult(BaseModel):
    events: list[EventColors]
    stats: dict


class ColorStage:
    CONFIG_FIELD: ClassVar[str] = "color"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir", "fps")

    def __init__(self) -> None:
        pass

    def run(self, globals: PipelineGlobals, config: ColorConfig) -> ColorExtractionResult:
        raise NotImplementedError
