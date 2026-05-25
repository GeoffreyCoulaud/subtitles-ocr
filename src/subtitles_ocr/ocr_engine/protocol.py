from __future__ import annotations

from typing import Protocol

import numpy as np

# OcrDetection: défini dans pipeline/ocr.py (P2)


class OcrEngine(Protocol):
    def detect(self, image: np.ndarray) -> list[OcrDetection]: ...
