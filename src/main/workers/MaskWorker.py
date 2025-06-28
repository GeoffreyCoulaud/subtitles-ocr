from src.main.models.FramesWorkerOutput import FramesWorkerOutput
from src.main.workers.Worker import Worker


from PIL.Image import open as image_open
from PIL.Image import fromarray as image_fromarray

import numpy as np


class MaskWorker(Worker[FramesWorkerOutput, FramesWorkerOutput]):
    """Service that masks frames to remove unwanted areas"""

    name = "Masking frames"
    input_queue_name = "Frames"
    output_queue_name = "Masked frames"

    __saturation_threshold: int

    def __init__(self, saturation_threshold: int):
        """
        Initialize the MaskWorker with a saturation threshold.
        Pixels with saturation above this threshold will be masked.
        """
        self.__saturation_threshold = saturation_threshold

    def process_item(self, item):
        """Mask a single frame to remove unwanted areas"""

        self._send_message(
            "[%d/%d] Masking frame: %s" % (item.index, item.total, item.path),
            level="DEBUG",
        )
        with image_open(item.path) as original:
            image = original.convert("HSV")
        width, height = image.size

        # Use numpy to exclude pixels based on saturation,
        # setting them to black where over the threshold
        pixels = np.array(image)
        mask = pixels[:, :, 1] > self.__saturation_threshold
        pixels[mask] = (0, 0, 0)

        # Convert the numpy array back to a PIL image and save it
        (
            image_fromarray(pixels, mode="HSV")
            .convert("RGB")
            .save(item.path, format="PNG")
        )

        return [item]
