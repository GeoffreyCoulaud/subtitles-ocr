from fractions import Fraction
from pathlib import Path

import ffmpeg  # type: ignore

from src.main.models.FramesWorkerOutput import FramesWorkerOutput
from src.main.workers.Worker import Worker


class FramesWorker(Worker[Path, FramesWorkerOutput]):
    """Service that extracts frames from video files at a specified frame rate (fps)"""

    name = "Frames Extraction"
    input_queue_name = "Videos"
    output_queue_name = "Frames"

    __output_dir: Path
    __fps: float
    __crop_height: int
    __y_position: int

    def __init__(
        self,
        frames_dir: Path,
        fps: float = 2.0,
        crop_height: int = 0,
        y_position: int = 0,
    ):
        self.__output_dir = frames_dir
        self.__fps = fps
        self.__crop_height = crop_height
        self.__y_position = y_position

    def _extract_time_base(self, video_path: Path) -> float:
        """
        Extract the time base of the video file using ffprobe
        The time base is the fraction of seconds per frame
        This is used to calculate the timestamps of the extracted frames.
        """
        result = ffmpeg.probe(
            filename=str(video_path),
            loglevel="quiet",  # Suppress output
            select_streams="v:0",  # Select the first video stream
        )
        time_base_string = result["streams"][0]["time_base"].strip()
        return float(Fraction(time_base_string))

    def process_item(self, item):
        """Process a video file to extract frames"""

        self._send_message("Processing video file: %s" % item.name)

        # Create a directory for the frames if it does not exist
        # Its name will be the same as the video file without the extension
        frames_dir = self.__output_dir / item.stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames with PTS in filenames
        self._send_message(
            "Extracing frames from video",
            level="DEBUG",
        )
        frame_format = str(frames_dir / "frame_%010d.png")
        (
            ffmpeg.input(filename=str(item))
            .filter("fps", fps=self.__fps, round="up")
            .filter("crop", x=0, y=self.__y_position, w="in_w", h=self.__crop_height)
            .output(filename=frame_format, frame_pts=True)
            .run(**{"-hide_banner": 1, "-loglevel": "info"})
            # .run()
        )

        # Extract the time base of the video to calculate timestamps
        time_base = self._extract_time_base(item)
        self._send_message(
            "Extracted time base %f from video" % time_base,
            level="DEBUG",
        )

        # Include index and total count
        extracted_frames = sorted(frames_dir.glob("frame_*.png"))
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
