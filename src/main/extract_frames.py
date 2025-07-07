"""
Script to extract frames from a video file and rename them with their frame number and timestamp.
This script uses ffmpeg to extract frames and ffprobe to get the frame rate of the video.
The frames are saved in a specified output directory with names formatted as:
<frame_number>_<HH:MM:SS,mmm>.png
"""

import argparse
from argparse import Namespace
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from fractions import Fraction


class Arguments(Namespace):
    video_path: Path
    output_dir: Path


def frame_number_to_timestamp(frame_number: int, fps: float) -> str:
    """Convert a frame number to a timestamp string in the format HH:MM:SS,mmm."""
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(namespace=Arguments())

    # Validate arguments
    if not args.video_path.is_file():
        raise ValueError(f"Video path {args.video_path} is not a valid file.")
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    if not args.output_dir.is_dir():
        raise ValueError(f"Output {args.output_dir} is not a directory.")

    # Use ffmpeg to extract frames to a temporary directory first
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Execute the command
        # ffmpeg -vsync 0 -i video.avi -f image2 -frame_pts 1 %d.png
        command = [
            "ffmpeg",
            "-vsync",
            "0",
            "-i",
            str(args.video_path),
            "-f",
            "image2",
            "-frame_pts",
            "1",
            str(temp_path / "%d.png"),
        ]
        subprocess.run(command, check=True)

        # Get the video's frame rate
        # ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 video.avi
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(args.video_path),
        ]
        raw_fps = subprocess.run(command, capture_output=True, text=True, check=True)

        # Parse the frame rate
        fps = float(Fraction(raw_fps.stdout.strip()))

        # Rename frames to include the frame number and timestamp
        # In the format: frame_<frame_number>_00:00:00,000.png
        # Move frames to the output directory
        for frame in temp_path.iterdir():
            frame_number = int(frame.stem)
            frame_timestamp = frame_number_to_timestamp(frame_number, fps)
            new_name = f"{frame_number}_{frame_timestamp}.png"
            frame.rename(args.output_dir / new_name)


if __name__ == "__main__":
    main()
