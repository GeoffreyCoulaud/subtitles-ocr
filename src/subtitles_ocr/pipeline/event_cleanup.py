"""Stage 10 — per-event LLM cleanup with OCR reconciliation (ADR-0002 §3 Stage 9)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import EventCleanupConfig, PipelineGlobals
from subtitles_ocr.llm.protocol import LlmClient

STAGE_VERSION: int = 1


class CleanedEvent(BaseModel):
    """Strict LLM response schema for one event."""

    text: str


class EventCleanupItem(BaseModel):
    event_id: int
    cleaned_text: str
    skipped_llm: bool


class EventCleanupResult(BaseModel):
    items: list[EventCleanupItem]


class EventCleanupStage:
    CONFIG_FIELD: ClassVar[str] = "event_cleanup"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir",)

    def __init__(self, llm: LlmClient | None = None) -> None:
        self.llm = llm

    def run(self, globals: PipelineGlobals, config: EventCleanupConfig) -> EventCleanupResult:
        raise NotImplementedError
