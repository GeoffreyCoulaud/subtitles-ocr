"""Stage 6 — OCR via PaddleOCR PP-OCRv5 (ADR-0002 §3 Stage 6)."""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel

from subtitles_ocr.config import OcrConfig, PipelineGlobals
from subtitles_ocr.meta import NoCacheKey  # noqa: F401  -- re-exported by config; kept for future direct imports
from subtitles_ocr.ocr_engine.protocol import OcrEngine

# Stage 6's sidecar must also include FrameProcessingConfig (it consumes the
# diff→mask→compose stream), per ADR-0004 §3.2. The wiring happens in run() at
# implementation time; this scaffold only declares the contract.

STAGE_VERSION: int = 1


class OcrDetection(BaseModel):
    text: str
    confidence: float
    quad: list[tuple[int, int]]  # 4 points (TL, TR, BR, BL), oriented


class FrameOcrResult(BaseModel):
    fansub_frame_idx: int
    detections: list[OcrDetection]


class OcrStage:
    CONFIG_FIELD: ClassVar[str] = "ocr"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "fps",
        "fansub_total_frames",
        "debug_images",
    )

    def __init__(self, ocr_engine: OcrEngine | None = None) -> None:
        self.ocr_engine = ocr_engine

    def run(self, globals: PipelineGlobals, config: OcrConfig) -> "OcrResult":
        raise NotImplementedError


class OcrResult(BaseModel):
    """Summary of the OCR stage; per-frame detections persisted to JSONL."""

    fansub_total_frames: int
    frames_with_detections: int
