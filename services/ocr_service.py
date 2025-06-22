import pytesseract
from queue import Queue
from pathlib import Path
from dataclasses import dataclass
from PIL.Image import Image
from typing import Any

from services.service import Service


@dataclass
class OcrResult:
    timestamp: float
    text: str


class OCRService(Service[Path, OcrResult]):

    __lang: str

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        lang: str = "fra",
    ):
        super().__init__(input_queue, output_queue)
        self.__lang = lang

    def process_item(self, item):
        stem = item.stem.split("_")[1]
        timestamp = float(stem)
        text = pytesseract.image_to_string(
            item,
            lang=self.__lang,
            config="--psm 6",
        ).strip()
        return OcrResult(timestamp, text)
