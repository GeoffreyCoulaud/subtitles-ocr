import pytesseract  # type: ignore

from src.main.models.FramesWorkerOutput import FramesWorkerOutput
from src.main.models.OcrWorkerOutput import OcrWorkerOutput
from src.main.workers.Worker import Worker


class OcrWorker(Worker[FramesWorkerOutput, OcrWorkerOutput]):
    """Service that performs OCR on images and extracts text"""

    name = "OCR"
    input_queue_name = "Frames"
    output_queue_name = "Timed texts"

    __lang: str

    def __init__(self, lang: str = "fra"):
        self.__lang = lang

    def process_item(self, item: FramesWorkerOutput) -> list[OcrWorkerOutput]:
        text = pytesseract.image_to_string(
            image=str(item.path),
            lang=self.__lang,
            config="--psm 6",  # Assuming a single block of text, no layout analysis
        ).strip()
        return [
            OcrWorkerOutput(
                text=text,
                # Relay the input metadata
                timestamp=item.timestamp,
                index=item.index,
                total=item.total,
            )
        ]
