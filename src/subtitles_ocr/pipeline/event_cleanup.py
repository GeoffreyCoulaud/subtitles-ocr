"""Stage 10 — per-event LLM cleanup with OCR reconciliation (ADR-0002 §3 Stage 9 / ADR-0003 Stage 10)."""

from __future__ import annotations

import logging
from collections import Counter

from rapidfuzz.distance import Levenshtein
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

STAGE_VERSION: int = 3

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
        "Tu reçois plusieurs variantes OCR d'un même sous-titre français. "
        "Choisis la variante qui te semble la plus correcte et corrige UNIQUEMENT "
        "ces erreurs OCR systématiques :\n"
        "  - accents perdus : etre→être, deja→déjà, etudiant→étudiant, "
        "ca→ça, ou→où, la→là, a→à, meme→même, tres→très\n"
        "  - apostrophes perdues : Cest→C'est, quil→qu'il, Jai→J'ai, Tai→J'ai (T mal lu)\n"
        "  - confusables : I/l/1, rn/m, 0/O\n"
        "RÈGLES STRICTES :\n"
        "  - N'INVENTE PAS de mots. Le sens et la longueur doivent rester proches "
        "des variantes fournies.\n"
        "  - Si toutes les variantes sont identiques après normalisation, garde la modale.\n"
        "  - Réponds en JSON {\"text\": \"...\"} uniquement, sans commentaire.\n\n"
        f"Variantes OCR :\n{variants}\n"
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
                def _modal_consensus(texts: list[str]) -> tuple[str, bool]:
                    """Return (modal_text, is_consensus) — consensus is True when
                    the most common variant covers ≥ threshold of all variants."""
                    if not texts:
                        return "", False
                    mode_text, mode_count = Counter(texts).most_common(1)[0]
                    return mode_text, (mode_count / len(texts)) >= config.modal_consensus_threshold

                consensus_decisions = [_modal_consensus(ev.raw_ocr_texts) for ev in remaining]

                def process(args: tuple[AnimatedEvent, tuple[str, bool]]) -> EventCleanupItem:
                    ev, (modal_text, is_consensus) = args
                    if is_consensus:
                        return EventCleanupItem(
                            event_id=ev.event_id,
                            cleaned_text=modal_text,
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
                    # Hallucination guard: if the LLM diverges too far from any
                    # OCR variant, it likely invented text (we observed this
                    # with small models). Fall back to the modal text in that
                    # case so the LLM can only help, not hurt.
                    llm_text = resp.text
                    max_len = max(len(llm_text), max(len(v) for v in ev.raw_ocr_texts))
                    if max_len > 0:
                        best_sim = max(
                            1.0 - Levenshtein.distance(llm_text, v) / max(len(llm_text), len(v), 1)
                            for v in ev.raw_ocr_texts
                        )
                        if best_sim < 0.5:
                            return EventCleanupItem(
                                event_id=ev.event_id,
                                cleaned_text=modal_text,
                                skipped_llm=True,
                            )
                    return EventCleanupItem(
                        event_id=ev.event_id,
                        cleaned_text=llm_text,
                        skipped_llm=False,
                    )

                pairs = list(zip(remaining, consensus_decisions, strict=True))
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
