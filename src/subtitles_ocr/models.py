from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, model_validator

SUBTITLE_PALETTE: dict[str, str] = {
    "white":  "#FFFFFF",
    "yellow": "#FFFF00",
    "cyan":   "#00FFFF",
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
    style: Literal["regular", "italic"] = "regular"
    color: str = "white"
    position: Literal["top", "bottom"] = "bottom"

    @model_validator(mode="after")
    def resolve_color(self) -> "SubtitleElement":
        if not self.color.startswith("#"):
            self.color = SUBTITLE_PALETTE.get(self.color, "#FFFFFF")
        return self


class FrameAnalysis(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]


class SubtitleEvent(BaseModel):
    start_time: float
    end_time: float
    elements: list[SubtitleElement]
