# src/subtitles_ocr/pipeline/analyze.py
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from subtitles_ocr.models import FrameGroup, FrameAnalysis, SubtitleElement
from subtitles_ocr.vlm.client import OllamaClient
from subtitles_ocr.pipeline.retry import RetryConfig, RetryExhausted, NonRetryable, with_retry

log = logging.getLogger(__name__)


def parse_elements(raw: str) -> list[SubtitleElement]:
    data = json.loads(raw)
    if isinstance(data, dict):
        if not data:
            return []
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array or object, got {type(data).__name__}: {raw!r}")
    result = []
    skipped = 0
    for item in data:
        try:
            result.append(SubtitleElement.model_validate(item))
        except ValueError:
            log.debug("parse_elements: skipping invalid item: %r", item)
            skipped += 1
    if skipped > 0 and not result:
        raise ValueError(f"parse_elements: all {skipped} item(s) failed schema validation")
    return result


def analyze_group(
    group: FrameGroup,
    client: OllamaClient,
    prompt: str,
) -> FrameAnalysis:
    raw = client.analyze(group.frame, prompt, json_mode=True)
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
    retry_config: RetryConfig | None = None,
) -> Generator[FrameAnalysis | None, None, None]:
    if retry_config is None:
        retry_config = RetryConfig()
    def process(group: FrameGroup, has_text: bool) -> FrameAnalysis | None:
        if not has_text:
            return FrameAnalysis(
                start_time=group.start_time,
                end_time=group.end_time,
                elements=[],
            )
        try:
            return with_retry(lambda: analyze_group(group, client, prompt), retry_config, log)
        except NonRetryable as e:
            log.warning("analyze [%s] non-retryable: %s", group.frame.name, e.__cause__)
            return None
        except RetryExhausted:
            log.warning("analyze [%s] retries exhausted", group.frame.name)
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(process, groups, filter_results):
            yield result
