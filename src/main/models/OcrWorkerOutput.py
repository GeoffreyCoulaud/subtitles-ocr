from dataclasses import dataclass


@dataclass
class OcrWorkerOutput:
    timestamp: float
    text: str
    index: int
    total: int
