"""Tests for stage 6 — OcrStage.run + PaddleOcrEngine device handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from subtitles_ocr.config import FrameProcessingConfig, OcrConfig
from subtitles_ocr.exceptions import OcrDeviceInitError
from subtitles_ocr.io import JsonlWriter
from subtitles_ocr.pipeline.frame_processing import ComposedFrame
from subtitles_ocr.pipeline.ocr import FrameOcrResult, OcrDetection, OcrStage


# ---------------------------------------------------------------------------
# Fake OCR engine implementing the OcrEngine Protocol
# ---------------------------------------------------------------------------


class FakeOcrEngine:
    def __init__(self, responses: list[list[OcrDetection]]) -> None:
        self._responses = list(responses)
        self.calls: list[np.ndarray] = []

    def detect(self, image: np.ndarray) -> list[OcrDetection]:
        self.calls.append(image)
        if not self._responses:
            return []
        return self._responses.pop(0)


def _composed_frames(n: int) -> list[ComposedFrame]:
    return [
        ComposedFrame(fansub_frame_idx=i, image=np.full((8, 8, 3), i, dtype=np.uint8))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# OcrStage.run happy path
# ---------------------------------------------------------------------------


def _quad() -> list[tuple[int, int]]:
    return [(0, 0), (10, 0), (10, 5), (0, 5)]


def test_ocr_stage_writes_one_jsonl_line_per_composed_frame(
    tmp_workdir: Path, mock_globals
) -> None:
    composed = _composed_frames(5)
    responses = [
        [OcrDetection(text=f"frame{i}", confidence=0.9, quad=_quad())] for i in range(5)
    ]
    engine = FakeOcrEngine(responses=responses)

    stage = OcrStage(ocr_engine=engine)
    result = stage.run(
        mock_globals,
        OcrConfig(frame_processing=FrameProcessingConfig()),
        composed_frames=iter(composed),
    )

    assert result.fansub_total_frames == mock_globals.fansub_total_frames
    assert result.frames_with_detections == 5
    assert len(engine.calls) == 5

    jsonl = tmp_workdir / "06_ocr" / "results.jsonl"
    assert jsonl.exists()
    with JsonlWriter(jsonl, FrameOcrResult) as r:
        persisted = list(r.iter_persisted())
    assert [p.fansub_frame_idx for p in persisted] == [0, 1, 2, 3, 4]
    assert persisted[2].detections[0].text == "frame2"


def test_ocr_stage_writes_sidecar_including_frame_processing_config(
    tmp_workdir: Path, mock_globals
) -> None:
    composed = _composed_frames(2)
    engine = FakeOcrEngine(responses=[[], []])
    stage = OcrStage(ocr_engine=engine)
    stage.run(
        mock_globals,
        OcrConfig(frame_processing=FrameProcessingConfig(mask_t_high=0.42)),
        composed_frames=iter(composed),
    )
    sidecar = tmp_workdir / "06_ocr" / "results.meta.json"
    assert sidecar.exists()
    import json

    data = json.loads(sidecar.read_text())
    assert data["stage_name"] == "06_ocr"
    # ocr config (cache-invalidating fields only) and frame_processing config both
    # contribute to the cache key.
    assert "ocr" in data["config"]
    assert "frame_processing" in data["config"]
    # device + parallelism are NoCacheKey → excluded.
    assert "device" not in data["config"]["ocr"]
    assert "parallelism" not in data["config"]["ocr"]
    # FrameProcessingConfig overrides reflected.
    assert data["config"]["frame_processing"]["mask_t_high"] == 0.42


def test_ocr_stage_reads_alignment_json_from_workdir_when_composed_not_given(
    tmp_workdir: Path, mock_globals
) -> None:
    """Issue 1: when composed_frames is not injected, OcrStage must load
    ``02_alignment/alignment.json`` from the workdir (no orchestrator chaining).
    Missing file → PipelineError.
    """
    from subtitles_ocr.exceptions import PipelineError

    engine = FakeOcrEngine(responses=[])
    stage = OcrStage(ocr_engine=engine)
    # No alignment.json on disk and no composed_frames → must raise PipelineError
    with pytest.raises(PipelineError):
        stage.run(mock_globals, OcrConfig())


def test_ocr_stage_raises_pipeline_error_on_invalid_alignment_json(
    tmp_workdir: Path, mock_globals
) -> None:
    """Corrupted alignment.json on disk → PipelineError."""
    from subtitles_ocr.exceptions import PipelineError

    align_dir = tmp_workdir / "02_alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    (align_dir / "alignment.json").write_text("{not valid json")

    engine = FakeOcrEngine(responses=[])
    stage = OcrStage(ocr_engine=engine)
    with pytest.raises(PipelineError):
        stage.run(mock_globals, OcrConfig())


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_ocr_stage_resumes_after_partial_jsonl(tmp_workdir: Path, mock_globals) -> None:
    # Pre-seed 3 lines in 06_ocr/results.jsonl as if 3 frames already done.
    jsonl = tmp_workdir / "06_ocr" / "results.jsonl"
    with JsonlWriter(jsonl, FrameOcrResult) as w:
        for i in range(3):
            w.append(FrameOcrResult(fansub_frame_idx=i, detections=[]))

    composed = _composed_frames(5)
    # Engine should only be called for frames 3 and 4.
    engine = FakeOcrEngine(
        responses=[
            [OcrDetection(text="frame3", confidence=0.8, quad=_quad())],
            [OcrDetection(text="frame4", confidence=0.8, quad=_quad())],
        ]
    )
    stage = OcrStage(ocr_engine=engine)
    stage.run(
        mock_globals,
        OcrConfig(frame_processing=FrameProcessingConfig()),
        composed_frames=iter(composed),
    )

    assert len(engine.calls) == 2

    with JsonlWriter(jsonl, FrameOcrResult) as r:
        persisted = list(r.iter_persisted())
    assert [p.fansub_frame_idx for p in persisted] == [0, 1, 2, 3, 4]
    assert persisted[3].detections[0].text == "frame3"
    assert persisted[4].detections[0].text == "frame4"


# ---------------------------------------------------------------------------
# PaddleOcrEngine device handling
# ---------------------------------------------------------------------------


class _AlwaysFailFactory:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, *, lang: str, device: str) -> object:
        self.calls.append(device)
        raise RuntimeError(f"simulated paddle init failure on device={device}")


class _CpuOnlyFactory:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, *, lang: str, device: str) -> object:
        self.calls.append(device)
        if device.startswith("gpu") or device == "cuda" or device == "rocm":
            raise RuntimeError(f"GPU init failed on device={device}")
        return object()  # opaque CPU handle


def test_paddle_ocr_engine_explicit_cuda_hard_fails(monkeypatch) -> None:
    from subtitles_ocr.ocr_engine.paddle import PaddleOcrEngine

    factory = _AlwaysFailFactory()
    with pytest.raises(OcrDeviceInitError):
        PaddleOcrEngine(lang="latin", device="cuda", _engine_factory=factory)
    # Hard fail: no silent fallback attempt to CPU.
    assert "gpu" in factory.calls[0] or factory.calls[0] == "cuda"
    assert all("cpu" not in c for c in factory.calls)


def test_paddle_ocr_engine_explicit_rocm_hard_fails() -> None:
    from subtitles_ocr.ocr_engine.paddle import PaddleOcrEngine

    factory = _AlwaysFailFactory()
    with pytest.raises(OcrDeviceInitError):
        PaddleOcrEngine(lang="latin", device="rocm", _engine_factory=factory)


def test_paddle_ocr_engine_auto_falls_back_to_cpu_with_warning(caplog) -> None:
    import logging

    from subtitles_ocr.ocr_engine.paddle import PaddleOcrEngine

    factory = _CpuOnlyFactory()
    caplog.set_level(logging.WARNING, logger="subtitles_ocr.ocr_engine.paddle")
    engine = PaddleOcrEngine(lang="latin", device="auto", _engine_factory=factory)
    # First a GPU attempt, then a CPU fallback.
    assert factory.calls[0] in ("gpu", "gpu:0", "cuda")
    assert "cpu" in factory.calls[-1]
    assert engine.device_used == "cpu"
    warning_texts = [rec.getMessage().lower() for rec in caplog.records if rec.levelname == "WARNING"]
    assert any("fall" in msg and "cpu" in msg for msg in warning_texts)


def test_paddle_ocr_engine_cpu_baseline_initializes() -> None:
    from subtitles_ocr.ocr_engine.paddle import PaddleOcrEngine

    factory = _CpuOnlyFactory()
    engine = PaddleOcrEngine(lang="latin", device="cpu", _engine_factory=factory)
    assert engine.device_used == "cpu"
    assert factory.calls == ["cpu"]
