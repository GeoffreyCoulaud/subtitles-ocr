from subtitles_ocr.models import SubtitleElement, SubtitleEvent, VideoInfo
from subtitles_ocr.pipeline.serialize import (
    format_timestamp,
    rgb_to_ass_color,
    element_to_ass_tags,
    event_to_dialogue_lines,
    build_ass_content,
)

VIDEO_INFO = VideoInfo(width=1920, height=1080, fps=24.0)


def _element(**kwargs) -> SubtitleElement:
    defaults = dict(
        text="Test",
        style="regular",
        color="white",
        border_color="black",
        position="bottom",
        alignment="center",
    )
    return SubtitleElement(**{**defaults, **kwargs})


def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "0:00:00.00"


def test_format_timestamp_one_minute():
    assert format_timestamp(61.5) == "0:01:01.50"


def test_format_timestamp_over_one_hour():
    assert format_timestamp(3661.0) == "1:01:01.00"


def test_rgb_to_ass_color_white():
    assert rgb_to_ass_color("#FFFFFF") == "&HFFFFFF&"


def test_rgb_to_ass_color_red():
    assert rgb_to_ass_color("#FF0000") == "&H0000FF&"


def test_rgb_to_ass_color_blue():
    assert rgb_to_ass_color("#0000FF") == "&HFF0000&"


def test_element_to_ass_tags_alignment_bottom_center():
    assert "\\an2" in element_to_ass_tags(_element(position="bottom", alignment="center"))


def test_element_to_ass_tags_alignment_bottom_left():
    assert "\\an1" in element_to_ass_tags(_element(position="bottom", alignment="left"))


def test_element_to_ass_tags_alignment_bottom_right():
    assert "\\an3" in element_to_ass_tags(_element(position="bottom", alignment="right"))


def test_element_to_ass_tags_alignment_top_left():
    assert "\\an7" in element_to_ass_tags(_element(position="top", alignment="left"))


def test_element_to_ass_tags_alignment_top_center():
    assert "\\an8" in element_to_ass_tags(_element(position="top", alignment="center"))


def test_element_to_ass_tags_alignment_top_right():
    assert "\\an9" in element_to_ass_tags(_element(position="top", alignment="right"))


def test_element_to_ass_tags_colors():
    el = _element(color="yellow", border_color="cyan")
    tags = element_to_ass_tags(el)
    assert "\\c&H00FFFF&" in tags    # yellow #FFFF00 → BGR 00FFFF
    assert "\\3c&HFFFF00&" in tags   # cyan #00FFFF → BGR FFFF00


def test_element_to_ass_tags_bold():
    assert "\\b1" in element_to_ass_tags(_element(style="bold"))


def test_element_to_ass_tags_italic():
    assert "\\i1" in element_to_ass_tags(_element(style="italic"))


def test_element_to_ass_tags_regular_has_no_style_tags():
    tags = element_to_ass_tags(_element(style="regular"))
    assert "\\b1" not in tags
    assert "\\i1" not in tags


def test_event_to_dialogue_lines_count():
    event = SubtitleEvent(
        start_time=1.0, end_time=2.5,
        elements=[_element(text="A"), _element(text="B")],
    )
    assert len(event_to_dialogue_lines(event)) == 2


def test_event_to_dialogue_lines_format():
    event = SubtitleEvent(
        start_time=1.0, end_time=2.5,
        elements=[_element(text="Bonjour")],
    )
    lines = event_to_dialogue_lines(event)
    assert lines[0].startswith("Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,")
    assert "Bonjour" in lines[0]


def test_build_ass_content_contains_script_info():
    content = build_ass_content([], VIDEO_INFO)
    assert "[Script Info]" in content
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content


def test_build_ass_content_contains_events_section():
    content = build_ass_content([], VIDEO_INFO)
    assert "[Events]" in content
