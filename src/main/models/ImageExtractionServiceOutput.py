from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageExtractionServiceOutput:
    timestamp: float
    index: int
    total: int
    path: Path
    source_video: Path