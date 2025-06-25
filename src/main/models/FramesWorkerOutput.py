from dataclasses import dataclass
from pathlib import Path


@dataclass
class FramesWorkerOutput:
    timestamp: float
    index: int
    total: int
    path: Path
