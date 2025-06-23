from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleServiceOutput:
    """Represents a subtitle with a start and end time, text, and metadata"""

    start: float
    end: float
    text: str

    frame_count: int  # Number of frames in the subtitle
    total: int  # Total number of frames in the video
    source_video: Path  # Path to the source video file
