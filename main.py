import argparse

from mimetypes import guess_file_type
from multiprocessing import JoinableQueue
from pathlib import Path
from queue import Empty
from writers.SubripWriter import SubripWriter
from models.OcrServiceOutput import OcrServiceOutput
from models.SubtitleServiceOutput import SubtitleServiceOutput
from models.ImageExtractionServiceOutput import ImageExtractionServiceOutput
from services.ImageExtractionService import ImageExtractionService
from services.OcrService import OcrService
from services.ServiceRunner import ServiceRunner
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

    # Detect the videos at the input path
    raw_input_paths = []
    if arguments.input.is_file():
        raw_input_paths.append(arguments.input)
    elif arguments.input.is_dir():
        for dirent in arguments.input.iterdir():
            raw_input_paths.append(dirent)
    input_video_file_paths = []
    for dirent in raw_input_paths:
        if not dirent.is_file():
            continue
        (mime, _encoding) = guess_file_type(dirent)
        if mime is None or not mime.startswith("video/"):
            continue
        input_video_file_paths.append(dirent)

    # Check the output path
    if not arguments.output.exists():
        arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.output.is_dir():
        raise ValueError(f"Output path {arguments.output} is not a directory")

    # Create queues for inter-service communication
    videos_queue = JoinableQueue[Path]()
    frames_queue = JoinableQueue[ImageExtractionServiceOutput]()
    timed_text_queue = JoinableQueue[OcrServiceOutput]()
    subtitles_queue = JoinableQueue[SubtitleServiceOutput]()

    # Create the services
    image_extraction_service = ImageExtractionService(
        input_queue=videos_queue,
        output_queue=frames_queue,
        output_dir=arguments.output,
        fps=arguments.fps,
        crop_height=arguments.crop_height,
        y_position=arguments.y_pos,
    )
    ocr_service = OcrService(
        input_queue=frames_queue,
        output_queue=timed_text_queue,
        lang=arguments.lang,
    )
    subtitle_service = SubtitleService(
        input_queue=timed_text_queue,
        output_queue=subtitles_queue,
    )

    # Create the service runners
    service_runners: list[ServiceRunner] = [
        ServiceRunner(service=image_extraction_service),
        ServiceRunner(service=ocr_service),
        ServiceRunner(service=subtitle_service),
    ]

    # Input the videos to the queue
    for video_path in input_video_file_paths:
        videos_queue.put(video_path)

    # Start the service runners
    print("Starting service runners...")
    for runner in service_runners:
        runner.start()

    # Collect the output from the subtitle service
    print("Collecting subtitles...")
    print("\n" * 4)  # Leave some space for the output texts
    blocking_seconds = 15 / 1000
    output_srt_buffer: dict[Path, list[SubtitleServiceOutput]] = {}
    while True:

        # Clear the 4 previous lines
        print("\033[F\033[K" * 4, end="")

        # Print the current status of the queues
        print(f"Videos to process      : {videos_queue.qsize()}")
        print(f"Images to process      : {frames_queue.qsize()}")
        print(f"Timed texts to process : {timed_text_queue.qsize()}")
        print(f"Subtitles to process   : {subtitles_queue.qsize()}")

        # Collect an item from the subtitles queue
        try:
            item = subtitles_queue.get(timeout=blocking_seconds)
        except Empty:
            pass
        except ValueError:
            print("Subtitles creation done")
            subtitles_queue.close()
            break
        else:
            subtitles_queue.task_done()
            if item.source_video not in output_srt_buffer:
                output_srt_buffer[item.source_video] = []
            output_srt_buffer[item.source_video].append(item)

    # Join everything to ensure all services are finished
    print("Waiting for all services to finish...")
    for runner in service_runners:
        runner.join()
    for queue in (
        videos_queue,
        frames_queue,
        timed_text_queue,
        subtitles_queue,
    ):
        queue.join()

    # Create the writer
    subrip_writer = SubripWriter()

    # Write the subtitles to the files
    print("Writing subtitles to output files...")
    for video_path, subtitles in output_srt_buffer.items():
        subrip_writer.write_subtitles(
            video_path=video_path,
            subtitles=subtitles,
            output_dir=arguments.output,
        )

    print("All subtitles written successfully.")
    print("Bye :)")


if __name__ == "__main__":
    main()
