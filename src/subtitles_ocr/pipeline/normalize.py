"""Stage 11 - deterministic text normalization (ADR-0005).

Supersedes the LLM-based whole-document cleanup of ADR-0002 section 3
Stage 10. Pure functions on event text, no external dependencies, no LLM
client.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import NormalizeConfig, PipelineGlobals
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict, fingerprint
from subtitles_ocr.pipeline.event_cleanup import EventCleanupItem

logger = logging.getLogger(__name__)

STAGE_VERSION: int = 1

_STAGE_NAME = "11_normalize"
_OUT_DIR = "11_normalize"
_OUT_FILE = "normalized.json"
_META_FILE = "normalized.meta.json"
_IN_PATH = ("09_event_cleanup", "cleaned.jsonl")

# Invisible characters to strip after NFKC normalization. NFKC already folds
# NBSP (U+00A0) into a regular space, so it is absent here. ZWJ (U+200D) is
# kept (potentially legitimate in combined glyphs / emoji sequences).
_INVISIBLE_CHARS: tuple[str, ...] = (
    chr(0x200B),  # ZERO WIDTH SPACE
    chr(0x200C),  # ZERO WIDTH NON-JOINER
    chr(0xFEFF),  # BYTE ORDER MARK / ZWNBSP
    chr(0x2060),  # WORD JOINER
)

# Collapse runs of horizontal whitespace (spaces, tabs) into a single space.
# Newlines are explicitly NOT collapsed: they are meaningful (multi-line
# subtitles, exported as \N).
_HSPACE_RUN = re.compile(r"[ \t]+")

# Trim horizontal whitespace around newlines so " \n " becomes "\n".
_NEWLINE_WITH_SPACES = re.compile(r" *\n *")


class NormalizedEvent(BaseModel):
    event_id: int
    cleaned_text: str


class NormalizeResult(BaseModel):
    events: list[NormalizedEvent]


def normalize_text(text: str) -> str:
    """Apply NFKC + invisible-char strip + whitespace normalization."""
    text = unicodedata.normalize("NFKC", text)
    for c in _INVISIBLE_CHARS:
        text = text.replace(c, "")
    text = _HSPACE_RUN.sub(" ", text)
    text = _NEWLINE_WITH_SPACES.sub("\n", text)
    return text.strip()


def is_noise(text: str) -> bool:
    """Return True if `text` is OCR noise that should be pruned.

    Criterion: shorter than 2 characters, or zero alphabetic characters.
    """
    if len(text) < 2:
        return True
    if not any(c.isalpha() for c in text):
        return True
    return False


class NormalizeStage:
    CONFIG_FIELD: ClassVar[str] = "normalize"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir",)

    def run(self, globals: PipelineGlobals, config: NormalizeConfig) -> NormalizeResult:
        workdir = globals.workdir
        out_dir = workdir / _OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / _OUT_FILE
        meta_path = out_dir / _META_FILE
        in_path = workdir / _IN_PATH[0] / _IN_PATH[1]

        candidate_meta = self._build_meta(globals, config, in_path)
        if self._cache_hit(out_path, meta_path, candidate_meta):
            logger.info("normalize cache hit; loading %s", out_path)
            return NormalizeResult.model_validate_json(out_path.read_text(encoding="utf-8"))

        items = list(JsonlWriter(in_path, EventCleanupItem).iter_persisted())
        events: list[NormalizedEvent] = []
        for item in items:
            normalized = normalize_text(item.cleaned_text)
            if is_noise(normalized):
                continue
            events.append(NormalizedEvent(event_id=item.event_id, cleaned_text=normalized))
        result = NormalizeResult(events=events)

        self._atomic_write(out_path, result.model_dump_json())
        self._atomic_write(meta_path, candidate_meta.model_dump_json())
        return result

    def _build_meta(
        self,
        globals_: PipelineGlobals,
        config: NormalizeConfig,
        in_path: Path,
    ) -> BaseMeta:
        return BaseMeta(
            stage_name=_STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset={k: str(getattr(globals_, k)) for k in self.GLOBALS_USED},
            input_fingerprints={
                "event_cleanup_jsonl": fingerprint(in_path, treat_as_intermediate=True),
            },
            written_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _cache_hit(out_path: Path, meta_path: Path, candidate: BaseMeta) -> bool:
        if not out_path.exists() or not meta_path.exists():
            return False
        try:
            persisted = BaseMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            return False
        return persisted.matches(candidate)

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)


