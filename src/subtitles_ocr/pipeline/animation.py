"""Stage 8 — animation analysis (`\\move` + `\\fad`), ADR-0003 §4.2.

MVP scope (ADR-0003 §3): this stage runs as a passthrough — every Stage 7 event
becomes an AnimatedEvent with motion=None, fade_in_ms=0, fade_out_ms=0.
Disabling Stage 8 (or running it in passthrough form) yields the ADR-0002
static-only pipeline behavior exactly. Full \\move + \\fad detection is
deferred to Phase 6.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import AnimationConfig, PipelineGlobals
from subtitles_ocr.meta import (
    BaseMeta,
    FileFingerprint,
    cache_invalidating_dict,
    fingerprint,
)
from subtitles_ocr.pipeline.group import GroupResult, SubtitleEvent

logger = logging.getLogger(__name__)

STAGE_VERSION: int = 1

STAGE_NAME = "08_animation"


class AnimatedEvent(BaseModel):
    event_id: int
    fansub_frame_start: int
    fansub_frame_end: int
    raw_ocr_texts: list[str]
    raw_ocr_confidences: list[float]
    quads_per_frame: dict[int, list[tuple[int, int]]]
    quad_median: list[tuple[int, int]]
    member_frame_indices: list[int]
    # motion = {"type": "linear", "start": (x, y), "end": (x, y)}
    #        | {"type": "nonlinear_flagged"}
    #        | None  (static)
    motion: dict | None
    fade_in_ms: int
    fade_out_ms: int


class AnimationAnalysisResult(BaseModel):
    events: list[AnimatedEvent]
    stats: dict


_EMPTY_STATS: dict[str, int] = {
    "static": 0,
    "linear_move": 0,
    "flagged_nonlinear": 0,
    "fade_in_only": 0,
    "fade_out_only": 0,
    "full_fade": 0,
}


class AnimationStage:
    CONFIG_FIELD: ClassVar[str] = "animation"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir", "fps", "fansub_total_frames")

    def __init__(self) -> None:
        pass

    def run(self, globals: PipelineGlobals, config: AnimationConfig) -> AnimationAnalysisResult:
        out_dir = globals.workdir / STAGE_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "animation.json"
        meta_path = out_dir / "animation.meta.json"

        group_path = globals.workdir / "07_group" / "events.json"

        expected_meta = self._build_meta(globals, config, group_path)

        if out_path.exists() and meta_path.exists():
            try:
                cached_meta = BaseMeta.model_validate_json(meta_path.read_text())
            except ValueError:
                cached_meta = None
            # Round-trip the candidate meta through JSON before comparing so that
            # tuple/list and Path/str equivalences match the persisted form.
            expected_for_compare = BaseMeta.model_validate_json(expected_meta.model_dump_json())
            if cached_meta is not None and cached_meta.matches(expected_for_compare):
                logger.info("animation stage: cache hit, skipping recompute")
                return AnimationAnalysisResult.model_validate_json(out_path.read_text())

        group_result = GroupResult.model_validate_json(group_path.read_text())
        result = _passthrough(group_result.events)

        _atomic_write_text(out_path, result.model_dump_json())
        _atomic_write_text(meta_path, expected_meta.model_dump_json())
        logger.info(
            "animation stage: wrote %d events (stats=%s)",
            len(result.events),
            result.stats,
        )
        return result

    def _build_meta(
        self,
        globals: PipelineGlobals,
        config: AnimationConfig,
        group_path: Path,
    ) -> BaseMeta:
        globals_subset = {
            "fps": f"{globals.fps.numerator}/{globals.fps.denominator}",
            "fansub_total_frames": globals.fansub_total_frames,
            "workdir": str(globals.workdir),
        }
        inputs: dict[str, FileFingerprint] = {}
        if group_path.exists():
            inputs["group_events"] = fingerprint(group_path, treat_as_intermediate=True)
        return BaseMeta(
            stage_name=STAGE_NAME,
            stage_version=STAGE_VERSION,
            config=cache_invalidating_dict(config),
            globals_subset=globals_subset,
            input_fingerprints=inputs,
            written_at=datetime.now(timezone.utc),
        )


def _passthrough(events: list[SubtitleEvent]) -> AnimationAnalysisResult:
    animated = [
        AnimatedEvent(
            event_id=ev.event_id,
            fansub_frame_start=ev.fansub_frame_start,
            fansub_frame_end=ev.fansub_frame_end,
            raw_ocr_texts=list(ev.raw_ocr_texts),
            raw_ocr_confidences=list(ev.raw_ocr_confidences),
            quads_per_frame=dict(ev.quads_per_frame),
            quad_median=list(ev.quad_median),
            member_frame_indices=list(ev.member_frame_indices),
            motion=None,
            fade_in_ms=0,
            fade_out_ms=0,
        )
        for ev in events
    ]
    stats = dict(_EMPTY_STATS)
    stats["static"] = len(animated)
    return AnimationAnalysisResult(events=animated, stats=stats)


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
