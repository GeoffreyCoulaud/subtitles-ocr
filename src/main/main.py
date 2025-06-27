#!/usr/bin/env python3

import argparse
from mimetypes import guess_file_type
from pathlib import Path
from tempfile import TemporaryDirectory

from src.main.orchestrator.Orchestrator import Orchestrator
from src.main.workers.FramesWorker import FramesWorker
from src.main.workers.OcrWorker import OcrWorker
from src.main.workers.SubtitleEntriesWorker import SubtitleEntriesWorker
from src.main.workers.SubtitleFileWorker import SubtitleFileWorker


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

    # Check the input path
    if not arguments.input.is_file():
        raise ValueError(f"Input is not a file")
    mime = guess_file_type(arguments.input)[0]
    if mime is None or not mime.startswith("video/"):
        raise ValueError("Input is not a video file")

    # Check the output path
    if arguments.output.exists():
        raise ValueError(f"Output already exists")

    # Run the orchestrator
    with TemporaryDirectory() as temp_frames_dir:

        frames_worker = FramesWorker(
            frames_dir=Path(temp_frames_dir),
            fps=arguments.fps,
            crop_height=arguments.crop_height,
            y_position=arguments.y_pos,
        )

        ocr_worker = OcrWorker(lang=arguments.lang)

        subtitle_entries_worker = SubtitleEntriesWorker()

        subtitle_file_worker = SubtitleFileWorker(subtitle_path=arguments.output)

        orchestrator = Orchestrator[Path, Path](
            workers=[
                (frames_worker, 1),
                (ocr_worker, 6),
                (subtitle_entries_worker, 1),
                (subtitle_file_worker, 1),
            ]
        )
        orchestrator.run(input_data=[arguments.input])
        orchestrator.join()

    print("Goodbye :)")


if __name__ == "__main__":
    main()
