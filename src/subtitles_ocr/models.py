from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator

SUBTITLE_PALETTE: dict[str, str] = {
    "white":  "#FFFFFF",
    "yellow": "#FFFF00",
    "cyan":   "#00FFFF",
    "black":  "#000000",
    "gray":   "#808080",
}


class Frame(BaseModel):
    path: Path
    timestamp: float


class VideoInfo(BaseModel):
    width: int
    height: int
    fps: float = Field(gt=0.0)


class FrameGroup(BaseModel):
    start_time: float
    end_time: float
    frame: Path


class SubtitleElement(BaseModel):
    text: str
    style: Literal["regular", "bold", "italic"]
    color: str
    border_color: str
    position: Literal["top", "bottom"]
    alignment: Literal["left", "center", "right"]

    @model_validator(mode="after")
    def resolve_colors(self) -> "SubtitleElement":
        self.color = SUBTITLE_PALETTE.get(self.color, "#FFFFFF")
        self.border_color = SUBTITLE_PALETTE.get(self.border_color, "#000000")
        return self


class FrameAnalysis(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]


class SubtitleEvent(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]
