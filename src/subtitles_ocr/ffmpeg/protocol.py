from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class VideoMetadata(BaseModel):
    width: int
    height: int
    fps_num: int
    fps_den: int
    total_frames: int
    duration_s: float
    pix_fmt: str
    colorspace: str | None


class TranscodeArgs(BaseModel):
    input_path: Path
    output_path: Path
    filter_chain: str
    target_width: int
    target_height: int
    target_pix_fmt: str


class FfmpegRunner(Protocol):
    def probe(self, path: Path) -> VideoMetadata: ...
    def transcode(self, args: TranscodeArgs) -> None: ...
    def extract_audio(self, path: Path, track_index: int, out: Path) -> None: ...
