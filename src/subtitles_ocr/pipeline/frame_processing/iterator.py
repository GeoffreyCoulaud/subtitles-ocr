"""Stages 3-5 — diff + mask + compose, exposed as a streaming iterator.

`ComposedFrame` is a runtime-only frozen dataclass (never persisted), per
ADR-0004 §3.2. Pydantic is reserved for persisted boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from subtitles_ocr.config import FrameProcessingConfig, PipelineGlobals
from subtitles_ocr.pipeline.alignment.stage import AlignmentResult

STAGE_VERSION: int = 1


@dataclass(frozen=True)
class ComposedFrame:
    fansub_frame_idx: int
    image: np.ndarray  # RGB uint8, H×W×3


def iter_composed_frames(
    globals: PipelineGlobals,
    alignment_result: AlignmentResult,
    config: FrameProcessingConfig,
    start_at_fansub_idx: int = 0,
) -> Iterator[ComposedFrame]:
    raise NotImplementedError
    yield  # pragma: no cover  -- makes the function a generator for type-checkers
