"""Stage 6 — OCR via PaddleOCR PP-OCRv5 (ADR-0002 §3 Stage 6, ADR-0004 §3.2).

OcrStage consumes ``iter_composed_frames`` (the diff→mask→compose streaming
library) and is the architectural owner of the diff/mask/compose/ocr pipe;
its sidecar therefore includes both ``OcrConfig`` and ``FrameProcessingConfig``
in the cache-key payload.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Iterable

from pydantic import BaseModel

from subtitles_ocr.config import FrameProcessingConfig, OcrConfig, PipelineGlobals
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.meta import BaseMeta, cache_invalidating_dict
from subtitles_ocr.ocr_engine.protocol import OcrEngine
from subtitles_ocr.pipeline.alignment.stage import AlignmentResult
from subtitles_ocr.pipeline.frame_processing import (
    ComposedFrame,
    iter_composed_frames,
)

STAGE_VERSION: int = 1


class OcrDetection(BaseModel):
    text: str
    confidence: float
    quad: list[tuple[int, int]]  # 4 points (TL, TR, BR, BL), oriented


class FrameOcrResult(BaseModel):
    fansub_frame_idx: int
    detections: list[OcrDetection]


class OcrResult(BaseModel):
    """Summary of the OCR stage; per-frame detections persisted to JSONL."""

    fansub_total_frames: int
    frames_with_detections: int


class OcrStage:
    CONFIG_FIELD: ClassVar[str] = "ocr"
    GLOBALS_USED: ClassVar[tuple[str, ...]] = (
        "workdir",
        "fps",
        "fansub_total_frames",
        "debug_images",
    )

    def __init__(self, ocr_engine: OcrEngine | None = None) -> None:
        # Default deferred to keep imports light when only the class is needed.
        self.ocr_engine = ocr_engine

    def run(
        self,
        globals: PipelineGlobals,
        config: OcrConfig,
        *,
        alignment_result: AlignmentResult | None = None,
        frame_processing_config: FrameProcessingConfig | None = None,
        composed_frames: Iterable[ComposedFrame] | None = None,
    ) -> OcrResult:
        if self.ocr_engine is None:
            from subtitles_ocr.ocr_engine.paddle import PaddleOcrEngine

            self.ocr_engine = PaddleOcrEngine(lang=config.language, device=config.device)

        if frame_processing_config is None:
            frame_processing_config = FrameProcessingConfig()

        out_dir = globals.workdir / "06_ocr"
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "results.jsonl"

        frames_with_detections = 0
        with JsonlWriter(jsonl_path, FrameOcrResult) as writer:
            skip = writer.resume_index()

            # Count already-persisted frames that had detections so the summary
            # statistic survives across resume.
            if skip > 0:
                for prev in writer.iter_persisted():
                    if prev.detections:
                        frames_with_detections += 1

            if composed_frames is None:
                if alignment_result is None:
                    raise ValueError(
                        "OcrStage.run requires either composed_frames or alignment_result"
                    )
                composed_frames = iter_composed_frames(
                    globals,
                    alignment_result,
                    frame_processing_config,
                    start_at_fansub_idx=0,
                )

            seen = 0
            for cf in composed_frames:
                if seen < skip:
                    seen += 1
                    continue
                detections = self.ocr_engine.detect(cf.image)
                writer.append(
                    FrameOcrResult(
                        fansub_frame_idx=cf.fansub_frame_idx, detections=detections
                    )
                )
                if detections:
                    frames_with_detections += 1
                seen += 1

        self._write_sidecar(
            globals=globals,
            ocr_config=config,
            frame_processing_config=frame_processing_config,
        )
        return OcrResult(
            fansub_total_frames=globals.fansub_total_frames,
            frames_with_detections=frames_with_detections,
        )

    def _write_sidecar(
        self,
        *,
        globals: PipelineGlobals,
        ocr_config: OcrConfig,
        frame_processing_config: FrameProcessingConfig,
    ) -> None:
        config_payload: dict = {
            "ocr": cache_invalidating_dict(ocr_config),
            "frame_processing": cache_invalidating_dict(frame_processing_config),
            "stage_version": STAGE_VERSION,
        }
        globals_subset = {
            name: getattr(globals, name) for name in self.GLOBALS_USED if name != "workdir"
        }
        # Pydantic v2 round-trips Fraction via the model's serializer; here we
        # use the JSON-serializable form to avoid type errors in `dict[str, Any]`.
        if "fps" in globals_subset:
            fps = globals_subset["fps"]
            globals_subset["fps"] = f"{fps.numerator}/{fps.denominator}"
        meta = BaseMeta(
            stage_name="06_ocr",
            stage_version=STAGE_VERSION,
            config=config_payload,
            globals_subset=globals_subset,
            input_fingerprints={},
            written_at=datetime.now(tz=timezone.utc),
        )
        sidecar_path = globals.workdir / "06_ocr" / "results.meta.json"
        sidecar_path.write_text(meta.model_dump_json(indent=2))
