from dataclasses import dataclass
from pathlib import Path

from src.main.models.OcrWorkerOutput import OcrWorkerOutput
from src.main.models.SubtitleEntriesWorkerOutput import SubtitleEntriesWorkerOutput
from src.main.workers.Worker import Worker


@dataclass
class _TimedText:
    """
    Intermediate representation of a subtitle with a timestamp and text.
    Will be stored in a buffer until we can determine the start and end times of the
    subtitle composed by consecutive OCR results.
    """

    timestamp: float
    text: str
    index: int


@dataclass
class _TimedTextBuffer:
    size: int
    dict: dict[int, _TimedText]


class SubtitleEntriesWorker(Worker[OcrWorkerOutput, SubtitleEntriesWorkerOutput]):
    """Service that processes OCR results to create subtitle entries."""

    name = "Subtitles"
    input_queue_name = "Timed texts"
    output_queue_name = "Subtitle entries"

    def process_item(self, item: OcrWorkerOutput) -> list[SubtitleEntriesWorkerOutput]:
        return [
            SubtitleEntriesWorkerOutput(
                start=item.timestamp,
                end=item.timestamp,
                text=item.text,
                frame_size=1,
            )
        ]
