"""PaddleOCR-backed implementation of the OcrEngine protocol (ADR-0002 §3 Stage 6).

CPU is the baseline that must always work. GPU support is experimental:
PaddlePaddle has solid CUDA support, partial/experimental ROCm support, and no
MPS support. On `--ocr-device=auto`, init failures fall back to CPU with a
warning; on explicit `cuda`/`rocm`, init failures hard-fail via
``OcrDeviceInitError``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from subtitles_ocr.exceptions import OcrDeviceInitError

if TYPE_CHECKING:
    from subtitles_ocr.pipeline.ocr import OcrDetection

logger = logging.getLogger(__name__)


# A factory takes (lang, device) kwargs and returns a PaddleOCR-like instance.
EngineFactory = Callable[..., object]


def _default_paddle_factory(*, lang: str, device: str) -> object:
    # Import lazily so test environments without paddleocr installed can still
    # import this module (e.g., to exercise device-handling logic with a fake
    # factory).
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    # paddleocr 3.x with PP-OCRv5 mobile models — significantly better
    # recognition quality than v4 on small-resolution material while staying
    # within an interactive CPU latency budget (the server-det variant is
    # ~4× slower). MKL-DNN is disabled because the bundled paddlepaddle 3.x
    # runtime has incomplete oneDNN PIR support and segfaults on some Conv
    # kernels otherwise.
    return PaddleOCR(
        lang=lang,
        ocr_version="PP-OCRv5",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )


def _normalize_device(requested: str) -> str:
    # paddleocr 3.x PaddleOCR accepts "cpu", "gpu", "gpu:0", etc. via the
    # `device` arg; we map the project's CLI vocabulary onto that.
    if requested in ("cuda", "rocm"):
        return "gpu"
    return requested


class PaddleOcrEngine:
    def __init__(
        self,
        *,
        lang: str = "latin",
        device: str = "auto",
        _engine_factory: EngineFactory | None = None,
    ) -> None:
        self.lang = lang
        self.requested_device = device
        factory = _engine_factory or _default_paddle_factory

        if device == "auto":
            self._engine, self.device_used = self._init_with_fallback(factory, lang)
        elif device in ("cuda", "rocm"):
            try:
                paddle_device = _normalize_device(device)
                self._engine = factory(lang=lang, device=paddle_device)
                self.device_used = device
            except Exception as exc:  # noqa: BLE001 -- any paddle init failure
                raise OcrDeviceInitError(
                    f"PaddleOCR init failed on device={device}: {exc}",
                    stage="06_ocr",
                    hint=(
                        "Verify GPU drivers and PaddlePaddle GPU build, or "
                        "rerun with --ocr-device=cpu (or auto)."
                    ),
                ) from exc
        elif device == "cpu":
            try:
                self._engine = factory(lang=lang, device="cpu")
                self.device_used = "cpu"
            except Exception as exc:  # noqa: BLE001
                raise OcrDeviceInitError(
                    f"PaddleOCR CPU init failed: {exc}",
                    stage="06_ocr",
                    hint="CPU init failure is unexpected; check the paddleocr install.",
                ) from exc
        else:
            raise ValueError(
                f"Unsupported device={device!r}; expected one of "
                "{'auto', 'cuda', 'rocm', 'cpu'}"
            )

    @staticmethod
    def _init_with_fallback(
        factory: EngineFactory, lang: str
    ) -> tuple[object, str]:
        try:
            engine = factory(lang=lang, device="gpu")
            return engine, "gpu"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GPU init failed (%s: %s) — falling back to CPU.",
                type(exc).__name__,
                exc,
            )
            engine = factory(lang=lang, device="cpu")
            return engine, "cpu"

    def detect(self, image: np.ndarray) -> list["OcrDetection"]:
        from subtitles_ocr.pipeline.ocr import OcrDetection

        # paddleocr 3.x exposes `.predict(image)` and returns one dict per
        # image with `rec_texts`, `rec_scores`, and `rec_polys`. The legacy
        # `.ocr(image, cls=False)` path is kept as a fallback when a test
        # injects a 2.x-style engine via `_engine_factory`.
        if hasattr(self._engine, "predict"):
            results = self._engine.predict(image)  # type: ignore[attr-defined]
            out: list[OcrDetection] = []
            for entry in results:
                texts = entry.get("rec_texts") or []
                scores = entry.get("rec_scores") or []
                polys = entry.get("rec_polys") or []
                for text, score, poly in zip(texts, scores, polys, strict=False):
                    pts = [
                        (int(round(float(x))), int(round(float(y))))
                        for x, y in poly
                    ]
                    if len(pts) != 4:
                        continue
                    out.append(
                        OcrDetection(
                            text=str(text),
                            confidence=float(score),
                            quad=pts,
                        )
                    )
            return out

        # Legacy paddleocr 2.x path.
        results = self._engine.ocr(image, cls=False)  # type: ignore[attr-defined]
        out_legacy: list[OcrDetection] = []
        if not results:
            return out_legacy
        for image_result in results:
            if not image_result:
                continue
            for entry in image_result:
                if len(entry) != 2:
                    continue
                poly, text_and_score = entry
                if not isinstance(text_and_score, (list, tuple)) or len(text_and_score) < 2:
                    continue
                text = str(text_and_score[0])
                score = float(text_and_score[1])
                pts = [(int(round(float(x))), int(round(float(y)))) for x, y in poly]
                if len(pts) != 4:
                    continue
                out_legacy.append(OcrDetection(text=text, confidence=score, quad=pts))
        return out_legacy
