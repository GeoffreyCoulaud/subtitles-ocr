from queue import Queue
from dataclasses import dataclass
from typing import Literal


from services.ocr_service import OcrResult
from services.service import Service


@dataclass
class SubtitleEntry:
    start: float
    end: float
    text: str


class SubtitleService(Service[OcrResult, None | list[SubtitleEntry]]):

    __buffer: list[OcrResult]

    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__(input_queue, output_queue)
        self.__buffer = []

    def process_item(self, item: OcrResult) -> None:
        self.__buffer.append(item)
        return None

    def handle_end_of_input(self) -> tuple[Literal[True], list[SubtitleEntry]]:
        # Take all the OCR results in the buffer and create subtitles
        # We want to create the minimum number of subtitles
        # - If there are consecurtive OCR results with the same text, we merge them
        # - There is empty text, skip it

        subtitles = list[SubtitleEntry]()
        current_subtitle = None

        for result in self.__buffer:

            stripped_text = result.text.strip()

            # Found empty text, time to flush the current subtitle if it exists
            if not stripped_text and current_subtitle is not None:
                subtitles.append(current_subtitle)
                current_subtitle = None
                continue

            # Found text, check if we need to create a new subtitle
            if current_subtitle is None:
                # No current subtitle, create a new one
                current_subtitle = SubtitleEntry(
                    start=result.timestamp,
                    end=result.timestamp,
                    text=stripped_text,
                )
            elif current_subtitle.text == stripped_text:
                # Extend the current subtitle
                current_subtitle.end = result.timestamp
            else:
                # Create a new subtitle
                subtitles.append(current_subtitle)
                current_subtitle = SubtitleEntry(
                    start=result.timestamp,
                    end=result.timestamp,
                    text=stripped_text,
                )

        # If we have a current subtitle, add it to the list
        if current_subtitle is not None:
            subtitles.append(current_subtitle)

        return True, subtitles
