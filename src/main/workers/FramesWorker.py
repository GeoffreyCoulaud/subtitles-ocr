from fractions import Fraction
from pathlib import Path
import subprocess
import json

from src.main.models.FramesWorkerOutput import FramesWorkerOutput
from src.main.workers.Worker import Worker


class FramesWorker(Worker[Path, FramesWorkerOutput]):
    """Service that extracts frames from video files at a specified frame rate (fps)"""

    name = "Frames extraction"
    input_queue_name = "Videos"
    output_queue_name = "Frames"

    __output_dir: Path
    __fps: float
    __crop_height: int
    __y_position: int
    __brightness_threshold: float

    def __init__(
        self,
        frames_dir: Path,
        fps: float,
        crop_height: int,
        y_position: int,
        brightness_threshold: float,
    ):
        self.__output_dir = frames_dir
        self.__fps = fps
        self.__crop_height = crop_height
        self.__y_position = y_position
        self.__brightness_threshold = brightness_threshold

    def _extract_time_base(self, video_path: Path) -> float:
        """
        Extract the time base of the video file using ffprobe
        The time base is the fraction of seconds per frame
        This is used to calculate the timestamps of the extracted frames.
        """
        # Build ffprobe command to get video stream info as JSON
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-show_streams",
            str(video_path),
        ]

        # Run ffprobe and capture output
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Parse the JSON output
        stdout = json.loads(process.stdout)
        time_base_string = stdout["streams"][0]["time_base"].strip()
        return float(Fraction(time_base_string))

    def _extract_frames(self, path: Path, frame_format: str) -> None:
        filters = [
            f"fps={self.__fps}:round=up",  # Set the frame rate
            f"crop=in_w:{self.__crop_height}:0:{self.__y_position}",  # Crop the video
        ]
        cmd = [
            "ffmpeg",
            "-i",
            str(path),
            "-vf",
            ",".join(filters),
            "-frame_pts",
            "1",
            frame_format,
        ]
        subprocess.run(cmd, check=True)

    def process_item(self, item):
        """Process a video file to extract frames"""

        self._send_message("Processing video file: %s" % item.name)

        # Extract frames with PTS in filenames
        self._send_message("Extracting frames from video", level="DEBUG")
        frame_format = str(self.__output_dir / "frame_%010d.png")
        self._extract_frames(item, frame_format)

        # Extract the time base of the video to calculate timestamps
        time_base = self._extract_time_base(item)
        self._send_message(
            "Extracted time base %f from video" % time_base,
            level="DEBUG",
        )

        # Include index and total count
        extracted_frames = sorted(self.__output_dir.glob("frame_*.png"))
        outputs = [
            FramesWorkerOutput(
                timestamp=float(frame_path.stem.split("_")[1]) * time_base,
                index=index,
                total=len(extracted_frames),
                path=frame_path,
            )
            for index, frame_path in enumerate(extracted_frames)
        ]

        # Return the extracted frames with new names
        self._send_message(
            "Extracted %d frames from video %s" % (len(outputs), item.name),
            level="INFO",
        )
        return outputs
