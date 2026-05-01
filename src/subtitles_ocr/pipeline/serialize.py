from subtitles_ocr.models import SubtitleElement, SubtitleEvent, VideoInfo

_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

_AN_CODE: dict[tuple[str, str], int] = {
    ("top", "left"): 7,
    ("top", "center"): 8,
    ("top", "right"): 9,
    ("bottom", "left"): 1,
    ("bottom", "center"): 2,
    ("bottom", "right"): 3,
}


def format_timestamp(seconds: float) -> str:
    cs = round(seconds * 100)
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    cs %= 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def rgb_to_ass_color(hex_color: str) -> str:
    hex_color = hex_color.upper()
    r = hex_color[1:3]
    g = hex_color[3:5]
    b = hex_color[5:7]
    return f"&H{b}{g}{r}&"


def element_to_ass_tags(element: SubtitleElement) -> str:
    an = _AN_CODE[(element.position, element.alignment)]
    tags = [
        f"\\an{an}",
        f"\\c{rgb_to_ass_color(element.color)}",
        f"\\3c{rgb_to_ass_color(element.border_color)}",
    ]
    if element.style == "bold":
        tags.append("\\b1")
    elif element.style == "italic":
        tags.append("\\i1")
    return "{" + "".join(tags) + "}"


def event_to_dialogue_lines(event: SubtitleEvent) -> list[str]:
    start = format_timestamp(event.start_time)
    end = format_timestamp(event.end_time)
    lines = []
    for element in event.elements:
        tags = element_to_ass_tags(element)
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{tags}{element.text}")
    return lines


def build_ass_content(events: list[SubtitleEvent], video_info: VideoInfo) -> str:
    header = _ASS_HEADER.format(width=video_info.width, height=video_info.height)
    dialogue_lines = []
    for event in events:
        dialogue_lines.extend(event_to_dialogue_lines(event))
    return header + "\n".join(dialogue_lines) + ("\n" if dialogue_lines else "")
