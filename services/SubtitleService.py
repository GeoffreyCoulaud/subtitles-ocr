from dataclasses import dataclass
from pathlib import Path


from models.OcrServiceOutput import OcrServiceOutput
from models.SubtitleServiceOutput import SubtitleServiceOutput
from services.service import Service


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


class SubtitleService(Service[OcrServiceOutput, SubtitleServiceOutput]):
    """Service that processes OCR results to create subtitle entries."""

    __buffer: dict[Path, _TimedTextBuffer]

    def __init__(self, input_queue, output_queue):
        super().__init__(input_queue, output_queue)
        self.__buffer = {}

    def process_item(self, item: OcrServiceOutput) -> list[SubtitleServiceOutput]:

        # Create the source video buffer if it does not exist
        if item.source_video not in self.__buffer:
            self.__buffer[item.source_video] = _TimedTextBuffer(
                size=item.total,
                dict={},
            )

        buffer = self.__buffer[item.source_video]

        # Add the item to the buffer
        current_timed_text = _TimedText(
            timestamp=item.timestamp,
            text=item.text,
            index=item.index,
        )
        buffer.dict[item.index] = current_timed_text

        # --- Shortcut
        # If the item is empty text and has empty text neighbors or void,
        # skip finding the boundaries and outputting a subtitle.
        if current_timed_text.text == "":
            before = buffer.dict.get(current_timed_text.index - 1, None)
            after = buffer.dict.get(current_timed_text.index + 1, None)
            if (before is None or before.text == "") and (
                after is None or after.text == ""
            ):
                return []

        # Find the boundaries of the subtitle group
        # Boundaries are defined by the first and last items with the same text in the buffer.
        # Boundaries are inclusive, meaning they include the first and last items with the same text.
        # Boundaries cannot have gaps
        past_boundary = self._find_past_boundary(buffer, current_timed_text)
        future_boundary = self._find_future_boundary(buffer, current_timed_text)
        if past_boundary is None or future_boundary is None:
            # We cannot determine the boundaries, we need more items in the buffer
            return []

        # Output a subtitle from the past and future boundaries
        frame_count = future_boundary.index - past_boundary.index + 1
        return [
            SubtitleServiceOutput(
                start=past_boundary.timestamp,
                end=future_boundary.timestamp,
                text=item.text,
                source_video=item.source_video,
                frame_count=frame_count,
                total=item.total,
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
        for i in range(item.index, -1, -1):
            if i not in buffer.dict:
                # There is a gap in the buffer, we cannot determine the past boundary
                return None
            elif i == 0:
                # We reached the beginning of the buffer
                return buffer.dict[i]
            elif buffer.dict[i].text != item.text:
                # We found a different text, we just went past the past boundary
                return buffer.dict[i + 1]
            else:
                # Should not happen, but just in case
                raise RuntimeError("Unexpected state: past boundary not found")

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
        for i in range(item.index, buffer.size):
            if i not in buffer.dict:
                # There is a gap in the buffer, we cannot determine the future boundary
                return None
            elif i == buffer.size - 1:
                # We reached the end of the buffer
                return buffer.dict[i]
            elif buffer.dict[i].text != item.text:
                # We found a different text, we just went past the future boundary
                return buffer.dict[i - 1]
            else:
                # Should not happen, but just in case
                raise RuntimeError("Unexpected state: future boundary not found")
