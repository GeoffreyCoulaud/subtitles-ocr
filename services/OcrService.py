import pytesseract

from models.OcrServiceOutput import OcrServiceOutput
from models.ImageExtractionServiceOutput import ImageExtractionServiceOutput
from services.Service import Service


class OcrService(Service[ImageExtractionServiceOutput, OcrServiceOutput]):
    """Service that performs OCR on images and extracts text"""

    __lang: str

    def __init__(
        self,
        input_queue,
        output_queue,
        lang: str = "fra",
    ):
        super().__init__(input_queue, output_queue)
        self.__lang = lang

    def process_item(
        self, item: ImageExtractionServiceOutput
    ) -> list[OcrServiceOutput]:
        text = pytesseract.image_to_string(
            image=str(item.path),
            lang=self.__lang,
            config="--psm 6",  # Assuming a single block of text, no layout analysis
        ).strip()
        return [
            OcrServiceOutput(
                text=text,
                # Relay the input metadata
                timestamp=item.timestamp,
                index=item.index,
                total=item.total,
                source_video=item.source_video,
            )
        ]
