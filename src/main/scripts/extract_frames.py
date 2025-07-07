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
from shutil import move

from src.main.lib.print_banner import print_banner
from src.main.lib.handle_keyboard_interrupt import handle_keyboard_interrupt
from src.main.lib.frame_number_to_timestamp import frame_number_to_timestamp


class Arguments(Namespace):
    video_path: Path
    output_dir: Path


@handle_keyboard_interrupt
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file and rename them with frame number and timestamp."
    )
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

        print_banner(f"Extracting frames from {args.video_path}")

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

        print_banner("Renaming frames")

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
        fps = float(Fraction(raw_fps.stdout.strip()))

        # Rename frames to include the frame number and timestamp
        # In the format: frame_<frame_number>_00:00:00,000.png
        # Move frames to the output directory
        frames = list(temp_path.iterdir())
        for i, frame in enumerate(frames):

            # Print progress (only print the first, the last and every 1% frame)
            if i == 0 or i == len(frames) - 1 or i % (len(frames) // 100) == 0:
                percent = round(i / len(frames) * 100)
                print(f"[{percent: 3}%] Renaming {frame.name}...")

            frame_number = int(frame.stem)
            frame_timestamp = frame_number_to_timestamp(frame_number, fps)
            new_name = f"{frame_number}_{frame_timestamp}.png"

            # We cannot use `frame.rename()` here because it does not work across different filesystems.
            # Tempororary dirs are often created on tmpfs (RAM disk).
            move(src=frame, dst=args.output_dir / new_name)

        print_banner(f"Done! Frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
