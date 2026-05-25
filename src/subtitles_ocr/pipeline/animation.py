"""Stage 8 — animation analysis (`\\move` + `\\fad`), ADR-0003 §4.2."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from subtitles_ocr.config import AnimationConfig, PipelineGlobals

STAGE_VERSION: int = 1


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


class AnimationStage:
    CONFIG_FIELD: ClassVar[str] = "animation"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("workdir", "fps", "fansub_total_frames")

    def __init__(self) -> None:
        pass

    def run(self, globals: PipelineGlobals, config: AnimationConfig) -> AnimationAnalysisResult:
        raise NotImplementedError
