from typing import TypedDict


class RotatedRectangleDto(TypedDict):
    center: tuple[float, float]
    size: tuple[float, float]
    angle: float
