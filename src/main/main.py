#!/usr/bin/env python3

import argparse
from mimetypes import guess_file_type
from multiprocessing import Queue
from pathlib import Path
from queue import Empty

from src.main.models.ImageExtractionServiceOutput import ImageExtractionServiceOutput
from src.main.models.OcrServiceOutput import OcrServiceOutput
from src.main.models.SubtitleServiceOutput import SubtitleServiceOutput
from src.main.runners.ProcessServiceRunner import ProcessServiceRunner
from src.main.runners.ServiceRunner import ServiceRunner
from src.main.services.ImageExtractionService import ImageExtractionService
from src.main.services.OcrService import OcrService
from src.main.services.SubtitleService import SubtitleService
from src.main.writers.SubripWriter import SubripWriter


class Arguments(argparse.Namespace):
    # Required parameters
    input: Path
    output: Path
    # Optional parameters
    crop_height: int
    y_pos: int
    fps: float
    lang: str
    no_verify: bool


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crop-height", type=int, default=100)
    parser.add_argument("--y-pos", type=int, default=380)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--lang", type=str, default="fra")
    parser.add_argument("--no-verify", action="store_true")
    arguments = parser.parse_args(namespace=Arguments())

    # Detect the videos at the input path
    if not arguments.input.exists():
        raise ValueError(f"Input path {arguments.input} does not exist")
    raw_input_paths = []
    if arguments.input.is_file():
        raw_input_paths.append(arguments.input)
    elif arguments.input.is_dir():
        for dirent in arguments.input.iterdir():
            if not dirent.is_file():
                continue
            raw_input_paths.append(dirent)
    input_video_file_paths = []
    for dirent in raw_input_paths:
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
    videos_queue: "Queue[Path]" = Queue()
    frames_queue: "Queue[ImageExtractionServiceOutput]" = Queue()
    timed_text_queue: "Queue[OcrServiceOutput]" = Queue()
    subtitles_queue: "Queue[SubtitleServiceOutput]" = Queue()

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
    # Note: We run the OCR service with a pool size of 4 to allow parallel processing
    runners: list[ServiceRunner] = [
        ProcessServiceRunner(service=image_extraction_service),
        # HACK - this is a workaround, building the processes in a runner doesn't work
        *[
            # Pool of OCR services
            ProcessServiceRunner(service=ocr_service)
            for _ in range(6)
        ],
        ProcessServiceRunner(service=subtitle_service),
    ]

    # Input the videos to the queue
    print(f"Inputting {len(input_video_file_paths)} videos to the queue...")
    for video_path in input_video_file_paths:
        print(f"- {video_path}")
        videos_queue.put(video_path)

    # Check with the user before proceeding
    if not arguments.no_verify:
        answer = input("Last check, are you sure you want to proceed? (y/n): ")
        if answer.lower().strip() not in ("y", "yes"):
            print("Aborting...")
            exit(1)

    # Start the service runners
    print("Starting service runners...")
    for runner in runners:
        runner.run()

    # Collect the output from the subtitle service
    print("Collecting subtitles...")
    print("\n" * 4)  # Leave some space for the output texts
    blocking_seconds = 15 / 1000
    output_srt_buffer: dict[Path, list[SubtitleServiceOutput]] = {}
    while True:

        # Clear the 5 previous lines
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
            if item.source_video not in output_srt_buffer:
                output_srt_buffer[item.source_video] = []
            output_srt_buffer[item.source_video].append(item)

    # Join all the service runners
    print("Joining service runners...")
    for runner in runners:
        runner.join()

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
