import subprocess
import math
from pathlib import Path

from models.ImageExtractionServiceOutput import ImageExtractionServiceOutput
from services.service import Service


class ImageExtractionService(Service[Path, ImageExtractionServiceOutput]):
    """Service that extracts frames from video files at a specified frame rate (fps)"""

    __output_dir: Path
    __fps: float
    __crop_height: int
    __y_position: int

    def __init__(
        self,
        input_queue,
        output_queue,
        output_dir: Path,
        fps: float = 2.0,
        crop_height: int = 0,
        y_position: int = 0,
    ):
        super().__init__(input_queue, output_queue)
        self.__output_dir = output_dir
        self.__fps = fps
        self.__crop_height = crop_height
        self.__y_position = y_position

    def _get_video_duration(self, video_path: Path) -> float:
        """Get the duration of a video in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def _get_total_frames(self, video_path: Path) -> int:
        """Calculate the total number of frames that will be extracted based on video duration and fps."""
        duration = self._get_video_duration(video_path)
        return math.ceil(duration * self.__fps)

    def process_item(self, item):
        """Process a video file to extract frames"""

        # Create a directory for the frames if it does not exist
        # Its name will be the same as the video file without the extension
        frames_dir = self.__output_dir / item.stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames with timestamps in filenames
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(item),
            "-vf",
            f"fps={self.__fps},crop=iw:{self.__crop_height}:0:{self.__y_position}",
            "-qscale:v",
            "2",
            str(frames_dir / "frame_%06.3f.png"),
        ]
        subprocess.run(cmd, check=True)

        # Include index and total count
        outputs = list[ImageExtractionServiceOutput]()
        extracted_frames = sorted(frames_dir.glob("frame_*.png"))
        for index, frame_path in enumerate(extracted_frames):

            # Extract the timestamp from the original filename
            timestamp = float(frame_path.stem.split("_")[1])

            # Add the new path to the list of renamed frames
            outputs.append(
                ImageExtractionServiceOutput(
                    timestamp=timestamp,
                    index=index,
                    total=len(extracted_frames),
                    path=frame_path,
                    source_video=item,
                )
            )

        # Return the extracted frames with new names
        return outputs
