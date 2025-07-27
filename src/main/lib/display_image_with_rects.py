from pathlib import Path
from cv2 import RotatedRect
import cv2
import numpy as np


def display_image_with_rects(image_path: Path, rectangles: list[RotatedRect]):
    """Display the image with rectangles drawn on it for debugging purposes."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    color = (0, 255, 0)  # Green color for rectangles
    for rect in rectangles:
        pts = rect.points()
        pts = [(int(pt[0]), int(pt[1])) for pt in pts]
        for i in range(4):
            cv2.line(
                img=image,
                pt1=pts[i],
                pt2=pts[(i + 1) % 4],
                color=color,
                thickness=2,
                lineType=0,
                shift=0,
            )

    cv2.imshow("Detected Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
