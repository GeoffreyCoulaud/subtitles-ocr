from pathlib import Path

from src.main.models.SubtitleEntriesWorkerOutput import SubtitleEntriesWorkerOutput
from src.main.workers.Worker import Worker


class SubtitleFileWorker(Worker[SubtitleEntriesWorkerOutput, Path]):
    """Service that processes subtitle entries to create a subtitle file."""

    name = "Subtitle file"
    input_queue_name = "Subtitle entries"
    output_queue_name = "Subtitle file path"

    __output_path: Path
    __entries: list[SubtitleEntriesWorkerOutput]

    def __init__(self, subtitle_path: Path):
        self.__output_path = subtitle_path
        self.__entries = []

    def process_item(self, item):
        # Do nothing until the entire input queue has been collected
        self.__entries.append(item)
        return []

    def process_no_more_items(self) -> list[Path]:
        self._send_message(
            "No more items in the input queue, writing subtitles to file",
            level="INFO",
        )
        # Write the entries to the output file
        self.__write_subtitles(self.__entries, self.__output_path)
        # Return the output path as the result
        return [self.__output_path]

    def __write_subtitles(
        self, subtitles: list[SubtitleEntriesWorkerOutput], output_file: Path
    ) -> None:
        """
        Take the subtitles list and write them to an output file in SRT format.
        The output file will be placed in the output directory,
        named after the video file, with a .srt extension.
        """

        # Sort the subtitles by start time
        subtitles.sort(key=lambda x: x.start)

        # Write the subtitles to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            for i, subtitle in enumerate(subtitles):
                # Convert start and end times to SRT format
                start = self.__seconds_to_srt_time(subtitle.start)
                end = self.__seconds_to_srt_time(subtitle.end)
                # Write the subtitle in SRT format
                f.write(f"{i + 1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{subtitle.text}\n\n")

    def __seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format

        The timecode format used is hours:minutes:seconds,milliseconds
        with time units fixed to two zero-padded digits
        and fractions fixed to three zero-padded digits (00:00:00,000).
        The comma (,) is used for fractional separator.
        """

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
