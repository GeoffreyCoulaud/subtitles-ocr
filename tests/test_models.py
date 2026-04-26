import json
from pathlib import Path
from subtitles_ocr.models import (
    Frame, VideoInfo, FrameGroup,
    SubtitleElement, FrameAnalysis, SubtitleEvent,
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


def test_subtitle_element_defaults():
    element = SubtitleElement(
        text="Bonjour",
        position_x=0.5,
        position_y=0.9,
        font_size_relative=0.05,
        color="#FFFFFF",
        outline_color="#000000",
        bold=False,
        italic=False,
        rotation=0.0,
        shear_x=0.0,
        shear_y=0.0,
    )
    assert element.text == "Bonjour"
    assert element.rotation == 0.0


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
                position_x=0.5, position_y=0.9,
                font_size_relative=0.05,
                color="#FFFFFF", outline_color="#000000",
                bold=False, italic=False,
                rotation=0.0, shear_x=0.0, shear_y=0.0,
            )
        ],
    )
    data = json.loads(event.model_dump_json())
    restored = SubtitleEvent.model_validate(data)
    assert restored.elements[0].text == "Test"
