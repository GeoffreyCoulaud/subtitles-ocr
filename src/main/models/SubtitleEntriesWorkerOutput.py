from dataclasses import dataclass


@dataclass
class SubtitleEntriesWorkerOutput:
    """Represents a subtitle with a start and end time, text, and metadata"""

    start: float
    end: float
    text: str

    frame_size: int
    """Number of frames in the subtitle"""
