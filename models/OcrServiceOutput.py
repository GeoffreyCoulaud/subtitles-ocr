from dataclasses import dataclass
from pathlib import Path


@dataclass
class OcrServiceOutput:
    timestamp: float
    text: str
    index: int
    total: int
    source_video: Path
