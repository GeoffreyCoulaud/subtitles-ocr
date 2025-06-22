import subprocess
from pathlib import Path

from services.service import Service


class CropService(Service[Path, Path]):

    __crop_height: int
    __y_position: int

    def __init__(
        self,
        input_queue,
        output_queue,
        crop_height: int,
        y_position: int,
    ):
        super().__init__(input_queue, output_queue)
        self.__crop_height = crop_height
        self.__y_position = y_position

    def process_item(self, item: Path) -> Path:
        output = Path(item).parent / f"cropped_{item.name}"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(item),
            "-vf",
            f"crop=iw:{self.__crop_height}:0:{self.__y_position}",
            "-an",
            str(output),
        ]
        subprocess.run(cmd, check=True)
        return output
