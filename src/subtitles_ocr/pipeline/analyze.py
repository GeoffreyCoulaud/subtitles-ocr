import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from subtitles_ocr.models import FrameGroup, FrameAnalysis, SubtitleElement
from subtitles_ocr.vlm.client import OllamaClient

log = logging.getLogger(__name__)


def parse_elements(raw: str) -> list[SubtitleElement]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    result = []
    for item in data:
        try:
            result.append(SubtitleElement.model_validate(item))
        except ValueError:
            pass
    return result


def analyze_group(
    group: FrameGroup,
    client: OllamaClient,
    prompt: str,
) -> FrameAnalysis:
    raw = client.analyze(group.frame, prompt)
    log.debug("analyze [%s] raw → %r", group.frame.name, raw)
    elements = parse_elements(raw)
    if not elements:
        log.info("analyze [%s] (no elements)", group.frame.name)
    for el in elements:
        log.info(
            "analyze [%s] %-7s %-7s %-7s %-7s %-7s | %s",
            group.frame.name,
            el.position, el.alignment, el.color, el.border_color, el.style,
            el.text,
        )
    return FrameAnalysis(
        start_time=group.start_time,
        end_time=group.end_time,
        elements=elements,
    )


def analyze_groups(
    groups: list[FrameGroup],
    filter_results: list[bool],
    client: OllamaClient,
    prompt: str,
    workers: int,
) -> Generator[FrameAnalysis, None, None]:
    def process(group: FrameGroup, has_text: bool) -> FrameAnalysis:
        if not has_text:
            return FrameAnalysis(
                start_time=group.start_time,
                end_time=group.end_time,
                elements=[],
            )
        try:
            return analyze_group(group, client, prompt)
        except RuntimeError as e:
            log.warning("analyze [%s] failed: %s", group.frame.name, e)
            return FrameAnalysis(
                start_time=group.start_time,
                end_time=group.end_time,
                elements=[],
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(process, groups, filter_results):
            yield result
