"""Stage 10 — per-event LLM cleanup with OCR reconciliation (ADR-0002 §3 Stage 9 / ADR-0003 Stage 10)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import EventCleanupConfig, PipelineGlobals
from subtitles_ocr.exceptions import LlmRetryExhausted
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.llm.protocol import LlmCallFailed, LlmClient
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict
from subtitles_ocr.pipeline.animation import AnimatedEvent, AnimationAnalysisResult

logger = logging.getLogger(__name__)

STAGE_VERSION: int = 1

_STAGE_NAME = "10_event_cleanup"
_LLM_HINT = "Check Ollama logs and --event-cleanup-model availability."


class CleanedEvent(BaseModel):
    """Strict LLM response schema for one event."""

    text: str


class EventCleanupItem(BaseModel):
    event_id: int
    cleaned_text: str
    skipped_llm: bool


class EventCleanupResult(BaseModel):
    items: list[EventCleanupItem]


def _build_prompt(event: AnimatedEvent) -> str:
    variants = "\n".join(f"- {t!r}" for t in event.raw_ocr_texts)
    return (
        "You are reconciling OCR variants of one subtitle event into the single "
        "canonical text the original line most likely was.\n"
        "Fix common OCR confusables (rn/m, I/l/1, missing accents).\n"
        "Return JSON {\"text\": \"...\"} only.\n\n"
        f"Variants:\n{variants}\n"
    )


class EventCleanupStage:
    CONFIG_FIELD: ClassVar[str] = "event_cleanup"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir",)

    def __init__(self, llm: LlmClient | None = None) -> None:
        # Lazy default keeps test construction (EventCleanupStage(llm=fake)) clean
        # and avoids importing OllamaLlmClient when a fake is injected.
        self.llm = llm

    def run(self, globals: PipelineGlobals, config: EventCleanupConfig) -> EventCleanupResult:
        if self.llm is None:
            from subtitles_ocr.llm.ollama import OllamaLlmClient

            self.llm = OllamaLlmClient()

        animation_path = globals.workdir / "08_animation" / "animation.json"
        animation = AnimationAnalysisResult.model_validate_json(
            animation_path.read_text(encoding="utf-8")
        )

        out_dir = globals.workdir / _STAGE_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "cleaned.jsonl"

        with JsonlWriter(jsonl_path, EventCleanupItem, fsync_every=config.chunk_size) as writer:
            already_done = writer.resume_index()
            persisted = list(writer.iter_persisted())[:already_done]
            remaining = animation.events[already_done:]

            results: list[EventCleanupItem] = list(persisted)

            if remaining:
                consensus_flags = [
                    len(set(ev.raw_ocr_texts)) == 1 and len(ev.raw_ocr_texts) >= 1
                    for ev in remaining
                ]

                def process(args: tuple[AnimatedEvent, bool]) -> EventCleanupItem:
                    ev, is_consensus = args
                    if is_consensus:
                        return EventCleanupItem(
                            event_id=ev.event_id,
                            cleaned_text=ev.raw_ocr_texts[0],
                            skipped_llm=True,
                        )
                    prompt = _build_prompt(ev)
                    try:
                        resp = self.llm.complete(prompt, CleanedEvent, model=config.model or "")
                    except LlmCallFailed as e:
                        raise LlmRetryExhausted(
                            f"LLM cleanup failed for event_id={ev.event_id}",
                            stage=_STAGE_NAME,
                            hint=_LLM_HINT,
                        ) from e
                    return EventCleanupItem(
                        event_id=ev.event_id,
                        cleaned_text=resp.text,
                        skipped_llm=False,
                    )

                pairs = list(zip(remaining, consensus_flags, strict=True))
                with ThreadPoolExecutor(max_workers=max(1, config.parallelism)) as ex:
                    for item in ex.map(process, pairs):
                        writer.append(item)
                        results.append(item)

        meta = BaseMeta(
            stage_name=_STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset={k: getattr(globals, k) for k in self.GLOBALS_USED if k != "workdir"},
            input_fingerprints={},
            written_at=datetime.now(tz=timezone.utc),
        )
        meta_path = out_dir / "cleaned.meta.json"
        meta_path.write_text(meta.model_dump_json(), encoding="utf-8")

        return EventCleanupResult(items=results)
