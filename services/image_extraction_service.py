import subprocess
from pathlib import Path
from queue import Queue
from typing import Any

from services.service import Service


class ImageExtractionService(Service[Path, Path]):

    __fps: float

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        fps: float = 2.0,
    ):
        super().__init__(input_queue, output_queue)
        self.__fps = fps

    def process_item(self, item: Path) -> Path:
        frames_dir = Path(item).parent / f"frames_{item.stem}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(frames_dir / "frame_%06.3f.jpg")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(item),
            "-vf",
            f"fps={self.__fps}",
            "-qscale:v",
            "2",
            pattern,
        ]
        subprocess.run(cmd, check=True)
        return frames_dir
