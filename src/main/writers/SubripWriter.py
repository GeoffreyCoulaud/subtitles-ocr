from pathlib import Path

from src.main.models.SubtitleServiceOutput import SubtitleServiceOutput


class SubripWriter:

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

    def write_subtitles(
        self,
        video_path: Path,
        subtitles: list[SubtitleServiceOutput],
        output_dir: Path,
    ) -> None:
        """
        Take the subtitles list and write them to an output file in SRT format.
        The output file will be placed in the output directory,
        named after the video file, with a .srt extension.
        """

        # Sort the subtitles by start time
        subtitles.sort(key=lambda x: x.start)

        # Write the subtitles to the output file
        output_file = output_dir / f"{video_path.stem}.srt"

        print(f"Writing subtitles to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            for i, subtitle in enumerate(subtitles):
                # Convert start and end times to SRT format
                start = self.__seconds_to_srt_time(subtitle.start)
                end = self.__seconds_to_srt_time(subtitle.end)
                # Write the subtitle in SRT format
                f.write(f"{i + 1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{subtitle.text}\n\n")
