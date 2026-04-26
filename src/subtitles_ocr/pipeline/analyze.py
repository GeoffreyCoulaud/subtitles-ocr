import json
from subtitles_ocr.models import FrameGroup, FrameAnalysis, SubtitleElement
from subtitles_ocr.vlm.client import OllamaClient


def parse_elements(raw: str) -> list[SubtitleElement]:
    try:
        data = json.loads(raw)
        return [SubtitleElement.model_validate(item) for item in data]
    except (json.JSONDecodeError, ValueError):
        return []


def analyze_group(
    group: FrameGroup,
    client: OllamaClient,
    prompt: str,
) -> FrameAnalysis:
    raw = client.analyze(group.frame, prompt)
    elements = parse_elements(raw)
    return FrameAnalysis(
        start_time=group.start_time,
        end_time=group.end_time,
        elements=elements,
    )
