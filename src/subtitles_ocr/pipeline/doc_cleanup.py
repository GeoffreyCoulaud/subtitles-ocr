"""Stage 11 — whole-document LLM cleanup (ADR-0002 §3 Stage 10)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import DocCleanupConfig, PipelineGlobals
from subtitles_ocr.llm.protocol import LlmClient

STAGE_VERSION: int = 1


class FinalEvent(BaseModel):
    event_id: int
    cleaned_text: str


class DocCleanupResult(BaseModel):
    events: list[FinalEvent]


class DocCleanupStage:
    CONFIG_FIELD: ClassVar[str] = "doc_cleanup"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir",)

    def __init__(self, llm: LlmClient | None = None) -> None:
        self.llm = llm

    def run(self, globals: PipelineGlobals, config: DocCleanupConfig) -> DocCleanupResult:
        raise NotImplementedError
