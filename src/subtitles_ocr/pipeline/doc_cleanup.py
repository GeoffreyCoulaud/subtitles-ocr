"""Stage 11 — whole-document LLM cleanup (ADR-0002 §3 Stage 10).

One LLM call over all events of the episode. Optional synopsis (free Markdown)
injected into the prompt as a JSON string. Strict response schema validation:
event count AND ordered event_id sequence must match the input. Context window
overflow is detected from the underlying LlmCallFailed message and surfaced as
LlmPromptTooLarge (no chunked fallback, per ADR-0002 §3 Stage 10).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import DocCleanupConfig, PipelineGlobals
from subtitles_ocr.exceptions import (
    LlmPromptTooLarge,
    LlmResponseSchemaError,
    LlmRetryExhausted,
)
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.llm.protocol import LlmCallFailed, LlmClient
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict, fingerprint
from subtitles_ocr.pipeline.event_cleanup import EventCleanupItem

logger = logging.getLogger(__name__)

STAGE_VERSION: int = 1

_STAGE_NAME = "11_doc_cleanup"
_OUT_DIR = "11_doc_cleanup"
_OUT_FILE = "cleaned_final.json"
_META_FILE = "cleaned_final.meta.json"
_IN_PATH = ("10_event_cleanup", "cleaned.jsonl")


# Ollama context-overflow signals. Match conservatively on substrings present
# in the error message; misclassification only changes which PipelineError
# subclass is raised, not whether the pipeline fails.
_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context window",
    "too many tokens",
    "max tokens",
    "exceeds context",
    "prompt is too long",
)


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
        workdir = globals.workdir
        out_dir = workdir / _OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / _OUT_FILE
        meta_path = out_dir / _META_FILE
        in_path = workdir / _IN_PATH[0] / _IN_PATH[1]

        items = list(self._read_input(in_path))
        candidate_meta = self._build_meta(globals, config, in_path)

        if self._cache_hit(out_path, meta_path, candidate_meta):
            logger.info("doc cleanup cache hit; loading %s", out_path)
            return DocCleanupResult.model_validate_json(out_path.read_text(encoding="utf-8"))

        if self.llm is None:
            raise RuntimeError("DocCleanupStage requires an LlmClient (none injected)")

        prompt = self._build_prompt(items, config.synopsis_path)
        try:
            response = self.llm.complete(prompt, DocCleanupResult, model=config.model or "")
        except LlmCallFailed as exc:
            if self._is_context_overflow(exc):
                raise LlmPromptTooLarge(
                    "LLM context window exceeded for document cleanup",
                    stage=_STAGE_NAME,
                    hint=(
                        "Use a model with a larger context window or reduce the number "
                        "of events (e.g. split the episode)."
                    ),
                ) from exc
            raise LlmRetryExhausted(
                "LLM document-cleanup call exhausted retries",
                stage=_STAGE_NAME,
                hint="Check Ollama logs and --doc-cleanup-model availability.",
            ) from exc

        self._validate_response(response, items)
        self._atomic_write(out_path, response.model_dump_json())
        self._atomic_write(meta_path, candidate_meta.model_dump_json())
        return response

    # -------------------- helpers --------------------

    @staticmethod
    def _read_input(path: str | os.PathLike) -> list[EventCleanupItem]:
        from pathlib import Path

        p = Path(path)
        reader = JsonlWriter(p, EventCleanupItem)
        return list(reader.iter_persisted())

    @staticmethod
    def _build_prompt(items: list[EventCleanupItem], synopsis_path) -> str:
        # User payload structured as ADR-0002 §3 Stage 10 dictates:
        #   {"synopsis": <text or null>, "events": [{"id": int, "text": str}, ...]}
        # Serialised with json (not Pydantic) so `null` appears literally and so
        # that the synopsis string is JSON-encoded (escaping handled by stdlib).
        import json
        from pathlib import Path

        synopsis_text: str | None = None
        if synopsis_path is not None:
            synopsis_text = Path(synopsis_path).read_text(encoding="utf-8")

        payload = {
            "synopsis": synopsis_text,
            "events": [{"id": it.event_id, "text": it.cleaned_text} for it in items],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _validate_response(
        response: DocCleanupResult, inputs: list[EventCleanupItem]
    ) -> None:
        if len(response.events) != len(inputs):
            raise LlmResponseSchemaError(
                f"LLM returned {len(response.events)} events, expected {len(inputs)}",
                stage=_STAGE_NAME,
                hint="The model must echo every input event id in order.",
            )
        expected_ids = [it.event_id for it in inputs]
        got_ids = [ev.event_id for ev in response.events]
        if got_ids != expected_ids:
            raise LlmResponseSchemaError(
                "LLM response event_ids do not match input ids in order",
                stage=_STAGE_NAME,
                hint="The model must preserve the input event_id sequence.",
            )

    @staticmethod
    def _is_context_overflow(exc: BaseException) -> bool:
        msg = str(exc).lower()
        # Walk the cause chain too: the OllamaLlmClient wraps the underlying
        # HTTP/server error inside LlmCallFailed, but the overflow signal is
        # most informative on the original message.
        cur: BaseException | None = exc
        while cur is not None:
            if any(marker in str(cur).lower() for marker in _CONTEXT_OVERFLOW_MARKERS):
                return True
            cur = cur.__cause__
        return any(marker in msg for marker in _CONTEXT_OVERFLOW_MARKERS)

    def _build_meta(
        self,
        globals_: PipelineGlobals,
        config: DocCleanupConfig,
        in_path,
    ) -> BaseMeta:
        from pathlib import Path

        input_fps = {
            "event_cleanup_jsonl": fingerprint(Path(in_path), treat_as_intermediate=True),
        }
        if config.synopsis_path is not None:
            input_fps["synopsis"] = fingerprint(Path(config.synopsis_path))

        return BaseMeta(
            stage_name=_STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset={k: str(getattr(globals_, k)) for k in self.GLOBALS_USED},
            input_fingerprints=input_fps,
            written_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _cache_hit(out_path, meta_path, candidate: BaseMeta) -> bool:
        if not out_path.exists() or not meta_path.exists():
            return False
        try:
            persisted = BaseMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            return False
        return persisted.matches(candidate)

    @staticmethod
    def _atomic_write(path, payload: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
