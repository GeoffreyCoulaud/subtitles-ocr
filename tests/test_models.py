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


def test_subtitle_element_resolves_color_name_to_hex():
    element = SubtitleElement(
        text="Bonjour",
        style="regular",
        color="white",
        border_color="black",
        position="bottom",
        alignment="center",
    )
    assert element.text == "Bonjour"
    assert element.color == "#FFFFFF"
    assert element.border_color == "#000000"


def test_subtitle_element_unknown_color_defaults():
    element = SubtitleElement(
        text="Test",
        style="regular",
        color="other",
        border_color="other",
        position="bottom",
        alignment="center",
    )
    assert element.color == "#FFFFFF"
    assert element.border_color == "#000000"


def test_subtitle_element_colors_default_independently():
    element = SubtitleElement(
        text="Test",
        style="regular",
        color="yellow",
        border_color="other",
        position="bottom",
        alignment="center",
    )
    assert element.color == "#FFFF00"
    assert element.border_color == "#000000"


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
                border_color="cyan",
                position="bottom",
                alignment="center",
            )
        ],
    )
    data = json.loads(event.model_dump_json())
    restored = SubtitleEvent.model_validate(data)
    assert restored.elements[0].text == "Test"
    assert restored.elements[0].color == "#FFFF00"
    assert restored.elements[0].border_color == "#00FFFF"


def test_video_info_roundtrip():
    info = VideoInfo(width=1920, height=1080, fps=23.976)
    data = json.loads(info.model_dump_json())
    restored = VideoInfo.model_validate(data)
    assert restored.width == 1920
    assert restored.height == 1080
    assert abs(restored.fps - 23.976) < 1e-6
