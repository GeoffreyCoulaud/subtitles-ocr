from fractions import Fraction
from pathlib import Path

from models.ImageExtractionServiceOutput import ImageExtractionServiceOutput
from services.Service import Service

import ffmpeg

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

        # Create a directory for the frames if it does not exist
        # Its name will be the same as the video file without the extension
        frames_dir = self.__output_dir / item.stem
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames with PTS in filenames

        frame_format = str(frames_dir / "frame_%010d.png")
        (
            ffmpeg.input(filename=str(item))
            .filter("fps", fps=self.__fps, round="up")
            .filter("crop", x=0, y=self.__y_position, w="in_w", h=self.__crop_height)
            .output(filename=frame_format, frame_pts=True)
            .run()
            # .run(**{"-hide_banner": 1, "-loglevel": "info"})
        )

        # Extract the time base of the video to calculate timestamps
        time_base = self._extract_time_base(item)

        # Include index and total count
        outputs = list[ImageExtractionServiceOutput]()
        extracted_frames = sorted(frames_dir.glob("frame_*.png"))
        for index, frame_path in enumerate(extracted_frames):

            # Extract the PTS from the original filename
            pts = float(frame_path.stem.split("_")[1])

            # Add the new path to the list of renamed frames
            outputs.append(
                ImageExtractionServiceOutput(
                    timestamp=pts * time_base,
                    index=index,
                    total=len(extracted_frames),
                    path=frame_path,
                    source_video=item,
                )
            )

        # Return the extracted frames with new names
        return outputs
