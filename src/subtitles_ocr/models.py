from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import re

_HEX_COLOR_RE = re.compile(r'^#[0-9A-Fa-f]{6}$')


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
    position_x: float = Field(ge=0.0, le=1.0)
    position_y: float = Field(ge=0.0, le=1.0)
    font_size_relative: float = Field(gt=0.0)
    color: str        # "#RRGGBB"
    outline_color: str  # "#RRGGBB"
    bold: bool
    italic: bool
    rotation: float   # degrés
    shear_x: float
    shear_y: float

    @field_validator("color", "outline_color")
    @classmethod
    def validate_hex_color(cls, v: str) -> str:
        if not _HEX_COLOR_RE.match(v):
            raise ValueError(f"Expected #RRGGBB color, got: {v!r}")
        return v.upper()


class FrameAnalysis(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]


class SubtitleEvent(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]
