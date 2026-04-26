from pathlib import Path
from pydantic import BaseModel


class Frame(BaseModel):
    path: Path
    timestamp: float


class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float


class FrameGroup(BaseModel):
    start_time: float
    end_time: float
    frame: Path


class SubtitleElement(BaseModel):
    text: str
    position_x: float
    position_y: float
    font_size_relative: float
    color: str        # "#RRGGBB"
    outline_color: str  # "#RRGGBB"
    bold: bool
    italic: bool
    rotation: float   # degrés
    shear_x: float
    shear_y: float


class FrameAnalysis(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]


class SubtitleEvent(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]
