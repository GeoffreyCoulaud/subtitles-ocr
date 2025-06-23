import argparse

from pathlib import Path
from queue import Empty, Queue, ShutDown
from models.OcrServiceOutput import OcrServiceOutput
from models.SubtitleServiceOutput import SubtitleServiceOutput
from models.ImageExtractionServiceOutput import ImageExtractionServiceOutput
from services.ImageExtractionService import ImageExtractionService
from services.OcrService import OCRService
from services.SubtitleService import SubtitleService


class Arguments(argparse.Namespace):
    # Required parameters
    input: Path
    output: Path
    # Optional parameters
    crop_height: int
    y_pos: int
    fps: float
    lang: str


def seconds_to_srt_time(seconds: float) -> str:
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


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crop-height", type=int, default=100)
    parser.add_argument("--y-pos", type=int, default=380)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--lang", type=str, default="fra")
    arguments = parser.parse_args(namespace=Arguments())

    # Create queues for inter-service communication
    image_extraction_input_queue = Queue[Path]()
    ocr_input_queue = Queue[ImageExtractionServiceOutput]()
    subtitle_input_queue = Queue[OcrServiceOutput]()
    subtitle_output_queue = Queue[SubtitleServiceOutput]()

    # Create the output srt buffer
    # This will be used to collect the final subtitles to be written to the output file.
    output_srt_buffer = dict[Path, list[SubtitleServiceOutput]]()

    # Create the services
    image_extraction_service = ImageExtractionService(
        input_queue=image_extraction_input_queue,
        output_queue=ocr_input_queue,
        output_dir=arguments.output,
        fps=arguments.fps,
        crop_height=arguments.crop_height,
        y_position=arguments.y_pos,
    )
    ocr_service = OCRService(
        input_queue=ocr_input_queue,
        output_queue=subtitle_input_queue,
        lang=arguments.lang,
    )
    subtitle_service = SubtitleService(
        input_queue=subtitle_input_queue,
        output_queue=subtitle_output_queue,
    )

    # Input the video file into the image extraction service
    image_extraction_input_queue.put(arguments.input)

    # TODO start the services
    # For now, we will run them sequentially
    # Later, we can use threading or multiprocessing to run them in parallel

    # Wait for the services to finish processing
    QUEUE_GET_TIMEOUT = 1 / 1000  # seconds
    while True:
        try:
            item = subtitle_output_queue.get(timeout=QUEUE_GET_TIMEOUT)
        except Empty:
            pass
        except ShutDown:
            print("Subtitle processing is done")
            break
        else:
            if item.source_video not in output_srt_buffer:
                output_srt_buffer[item.source_video] = []
            output_srt_buffer[item.source_video].append(item)

    # After all services are finished, we can write the sorted subtitles to the output file
    for video_path, subtitles in output_srt_buffer.items():

        # Sort the subtitles by start time
        subtitles.sort(key=lambda x: x.start)

        # Write the subtitles to the output file
        output_file = arguments.output / f"{video_path.stem}.srt"
        with open(output_file, "w", encoding="utf-8") as f:
            for i, subtitle in enumerate(subtitles):
                # Convert start and end times to SRT format
                start = seconds_to_srt_time(subtitle.start)
                end = seconds_to_srt_time(subtitle.end)
                # Write the subtitle in SRT format
                f.write(f"{i + 1}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{subtitle.text}\n\n")


if __name__ == "__main__":
    main()
