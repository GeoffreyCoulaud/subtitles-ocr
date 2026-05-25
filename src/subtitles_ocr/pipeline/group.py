"""Stage 7 — per-quad trajectory grouping (ADR-0002 §3 Stage 7, ADR-0003 §4.1)."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import GroupConfig, PipelineGlobals

STAGE_VERSION: int = 1


class SubtitleEvent(BaseModel):
    event_id: int
    fansub_frame_start: int
    fansub_frame_end: int
    raw_ocr_texts: list[str]
    raw_ocr_confidences: list[float]
    # ADR-0003 §4.1: per-frame quads required for animation analysis.
    # Pydantic v2 serializes int keys as strings in JSON and converts back at validation time.
    quads_per_frame: dict[int, list[tuple[int, int]]]
    quad_median: list[tuple[int, int]]
    member_frame_indices: list[int]


class GroupResult(BaseModel):
    fansub_total_frames: int
    events: list[SubtitleEvent]
    stats: dict


class GroupStage:
    CONFIG_FIELD: ClassVar[str] = "group"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir", "fansub_total_frames")

    def __init__(self) -> None:
        pass

    def run(self, globals: PipelineGlobals, config: GroupConfig) -> GroupResult:
        raise NotImplementedError
