import json
import pytest
from pathlib import Path
from pydantic import ValidationError
from subtitles_ocr.models import (
    Frame, VideoInfo, FrameGroup,
    SubtitleElement, FrameAnalysis, SubtitleEvent,
    SUBTITLE_PALETTE,
)


def test_frame_serialization():
    frame = Frame(path=Path("frames/000001.jpg"), timestamp=0.042)
    data = json.loads(frame.model_dump_json())
    assert data["path"] == "frames/000001.jpg"
    assert data["timestamp"] == 0.042


def test_frame_group_roundtrip():
    group = FrameGroup(start_time=1.0, end_time=2.5, frame=Path("frames/000024.jpg"))
    line = group.model_dump_json()
    restored = FrameGroup.model_validate_json(line)
    assert restored.start_time == 1.0
    assert restored.end_time == 2.5
    assert restored.frame == Path("frames/000024.jpg")


def test_subtitle_element_resolves_color_name_to_hex():
    element = SubtitleElement(
        text="Bonjour",
        style="regular",
        color="white",
        position="bottom",
    )
    assert element.text == "Bonjour"
    assert element.color == "#FFFFFF"


def test_subtitle_element_unknown_color_defaults_to_white():
    element = SubtitleElement(
        text="Test",
        style="regular",
        color="other",
        position="bottom",
    )
    assert element.color == "#FFFFFF"


def test_frame_analysis_empty_elements():
    analysis = FrameAnalysis(start_time=0.0, end_time=1.0, elements=[])
    assert analysis.elements == []


def test_subtitle_event_roundtrip():
    event = SubtitleEvent(
        start_time=1.0,
        end_time=2.5,
        elements=[
            SubtitleElement(
                text="Test",
                style="regular",
                color="yellow",
                position="bottom",
            )
        ],
    )
    data = json.loads(event.model_dump_json())
    restored = SubtitleEvent.model_validate(data)
    assert restored.elements[0].text == "Test"
    assert restored.elements[0].color == "#FFFF00"


def test_video_info_roundtrip():
    info = VideoInfo(width=1920, height=1080, fps=23.976)
    data = json.loads(info.model_dump_json())
    restored = VideoInfo.model_validate(data)
    assert restored.width == 1920
    assert restored.height == 1080
    assert abs(restored.fps - 23.976) < 1e-6


# --- New schema: border_color and alignment removed, bold removed, black/gray removed ---

def test_subtitle_element_valid_with_four_fields():
    element = SubtitleElement(text="Bonjour", style="regular", color="white", position="bottom")
    assert element.text == "Bonjour"
    assert element.color == "#FFFFFF"


def test_subtitle_element_rejects_bold_style():
    with pytest.raises(ValidationError):
        SubtitleElement(text="Test", style="bold", color="white", position="bottom")


def test_subtitle_palette_excludes_black():
    assert "black" not in SUBTITLE_PALETTE


def test_subtitle_palette_excludes_gray():
    assert "gray" not in SUBTITLE_PALETTE


# --- Defaults: text is the only required field ---

def test_subtitle_element_text_only_is_valid():
    element = SubtitleElement(text="Bonjour")
    assert element.text == "Bonjour"


def test_subtitle_element_style_defaults_to_regular():
    assert SubtitleElement(text="Test").style == "regular"


def test_subtitle_element_color_defaults_to_white():
    assert SubtitleElement(text="Test").color == "#FFFFFF"


def test_subtitle_element_position_defaults_to_bottom():
    assert SubtitleElement(text="Test").position == "bottom"
