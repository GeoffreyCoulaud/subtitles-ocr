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
        position_x=0.5, position_y=0.9,
        font_size_relative=0.05,
        color="#FFFFFF", outline_color="#000000",
        bold=False, italic=False,
        rotation=0.0, shear_x=0.0, shear_y=0.0,
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
    # red #FF0000 → BGR → 0000FF
    assert rgb_to_ass_color("#FF0000") == "&H0000FF&"


def test_rgb_to_ass_color_blue():
    # blue #0000FF → BGR → FF0000
    assert rgb_to_ass_color("#0000FF") == "&HFF0000&"


def test_element_to_ass_tags_position():
    el = _element(position_x=0.5, position_y=0.9)
    tags = element_to_ass_tags(el, VIDEO_INFO)
    assert "\\pos(960,972)" in tags  # 0.5*1920=960, 0.9*1080=972


def test_element_to_ass_tags_font_size():
    el = _element(font_size_relative=0.05)
    tags = element_to_ass_tags(el, VIDEO_INFO)
    assert "\\fs54" in tags  # round(0.05 * 1080) = 54


def test_element_to_ass_tags_colors():
    el = _element(color="#FF0000", outline_color="#0000FF")
    tags = element_to_ass_tags(el, VIDEO_INFO)
    assert "\\c&H0000FF&" in tags   # red in BGR
    assert "\\3c&HFF0000&" in tags  # blue in BGR


def test_element_to_ass_tags_bold_italic():
    el = _element(bold=True, italic=True)
    tags = element_to_ass_tags(el, VIDEO_INFO)
    assert "\\b1" in tags
    assert "\\i1" in tags


def test_element_to_ass_tags_no_bold_italic_when_false():
    el = _element(bold=False, italic=False)
    tags = element_to_ass_tags(el, VIDEO_INFO)
    assert "\\b1" not in tags
    assert "\\i1" not in tags


def test_element_to_ass_tags_rotation():
    el = _element(rotation=45.0)
    tags = element_to_ass_tags(el, VIDEO_INFO)
    assert "\\frz45.00" in tags


def test_element_to_ass_tags_no_rotation_when_zero():
    el = _element(rotation=0.0)
    tags = element_to_ass_tags(el, VIDEO_INFO)
    assert "\\frz" not in tags


def test_event_to_dialogue_lines_count():
    event = SubtitleEvent(
        start_time=1.0, end_time=2.5,
        elements=[_element(text="A"), _element(text="B")],
    )
    lines = event_to_dialogue_lines(event, VIDEO_INFO)
    assert len(lines) == 2


def test_event_to_dialogue_lines_format():
    event = SubtitleEvent(
        start_time=1.0, end_time=2.5,
        elements=[_element(text="Bonjour")],
    )
    lines = event_to_dialogue_lines(event, VIDEO_INFO)
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
