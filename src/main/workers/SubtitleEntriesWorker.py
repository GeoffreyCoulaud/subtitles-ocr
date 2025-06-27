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

    __buffer: None | _TimedTextBuffer

    def process_item(self, item: OcrWorkerOutput) -> list[SubtitleEntriesWorkerOutput]:

        # Initialize the buffer if it is not already initialized
        if self.__buffer is None:
            self._send_message(
                "Initializing subtitle buffer, size %d" % item.total,
                level="DEBUG",
            )
            self.__buffer = _TimedTextBuffer(size=item.total, dict={})

        # Add the item to the buffer
        self._send_message(
            "[%d/%d] %f Adding item to buffer"
            % (item.index, item.total, item.timestamp),
            level="DEBUG",
        )
        current_timed_text = _TimedText(
            timestamp=item.timestamp,
            text=item.text,
            index=item.index,
        )
        self.__buffer.dict[item.index] = current_timed_text

        # --- Shortcut
        # If the item is empty text and has empty text neighbors or void,
        # skip finding the boundaries and outputting a subtitle.
        if current_timed_text.text == "":
            before = self.__buffer.dict.get(current_timed_text.index - 1, None)
            after = self.__buffer.dict.get(current_timed_text.index + 1, None)
            if (before is None or before.text == "") and (
                after is None or after.text == ""
            ):
                self._send_message(
                    "[%d/%d] Empty text item, skipping" % (item.index, item.total),
                    level="DEBUG",
                )
                return []

        # Find the boundaries of the subtitle group
        # Boundaries are defined by the first and last items with the same text in the buffer.
        # Boundaries are inclusive, meaning they include the first and last items with the same text.
        # Boundaries cannot have gaps
        past_boundary = self._find_past_boundary(self.__buffer, current_timed_text)
        future_boundary = self._find_future_boundary(self.__buffer, current_timed_text)
        if past_boundary is None or future_boundary is None:
            # We cannot determine the boundaries, we need more items in the buffer
            return []

        self._send_message(
            "[%d/%d] Found boundaries: %d -> %d"
            % (item.index, item.total, past_boundary.index, future_boundary.index),
            level="DEBUG",
        )

        # Output a subtitle from the past and future boundaries
        frame_count = future_boundary.index - past_boundary.index + 1
        return [
            SubtitleEntriesWorkerOutput(
                start=past_boundary.timestamp,
                end=future_boundary.timestamp,
                text=item.text,
                frame_size=frame_count,
                frame_total=item.total,
            )
        ]

    def _find_past_boundary(
        self,
        buffer: _TimedTextBuffer,
        item: _TimedText,
    ) -> _TimedText | None:
        """
        The past boundary may be:
        - the first item in the buffer (may be the item itself)
        - the item after the first item that has a different text
        """

        result = None

        for i in range(item.index, -1, -1):
            if i not in buffer.dict:
                # There is a gap in the buffer, we cannot determine the past boundary
                break
            elif i == 0:
                # We reached the beginning of the buffer
                result = buffer.dict[i]
                break
            elif buffer.dict[i].text != item.text:
                # We found a different text, we just went past the past boundary
                result = buffer.dict[i + 1]
                break
            else:
                # Should not happen, but just in case
                raise RuntimeError("Unexpected state: past boundary not found")

        return result

    def _find_future_boundary(
        self,
        buffer: _TimedTextBuffer,
        item: _TimedText,
    ) -> _TimedText | None:
        """
        The future boundary may be :
        - the last item in the buffer (may be this item itself)
        - the item before the first item that has a different text
        """

        result = None

        for i in range(item.index, buffer.size):
            if i not in buffer.dict:
                # There is a gap in the buffer, we cannot determine the future boundary
                break
            elif i == buffer.size - 1:
                # We reached the end of the buffer
                result = buffer.dict[i]
                break
            elif buffer.dict[i].text != item.text:
                # We found a different text, we just went past the future boundary
                result = buffer.dict[i - 1]
                break
            else:
                # Should not happen, but just in case
                raise RuntimeError("Unexpected state: future boundary not found")

        return result
