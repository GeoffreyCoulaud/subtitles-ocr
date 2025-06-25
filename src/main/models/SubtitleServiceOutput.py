from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleServiceOutput:
    """Represents a subtitle with a start and end time, text, and metadata"""

    start: float
    end: float
    text: str

    frame_size: int  # Number of frames in the subtitle
    frame_total: int  # Total number of frames in the video
    source_video: Path  # Path to the source video file
