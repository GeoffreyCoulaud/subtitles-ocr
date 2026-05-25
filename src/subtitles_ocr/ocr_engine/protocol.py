from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from subtitles_ocr.pipeline.ocr import OcrDetection


class OcrEngine(Protocol):
    def detect(self, image: np.ndarray) -> list["OcrDetection"]: ...
