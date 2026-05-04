# src/subtitles_ocr/pipeline/analyze.py
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from subtitles_ocr.models import FrameGroup, FrameAnalysis, SubtitleElement
from subtitles_ocr.vlm.client import OllamaClient
from subtitles_ocr.pipeline.retry import RetryConfig, RetryExhausted, NonRetryable, with_retry

log = logging.getLogger(__name__)


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    for prefix in ("```json", "```"):
        if raw.startswith(prefix) and raw.endswith("```"):
            return raw.removeprefix(prefix).removesuffix("```").strip()
    return raw


def parse_elements(raw: str) -> list[SubtitleElement]:
    raw = _strip_code_fence(raw)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}: {raw!r}")
    if not data:
        return []
    items = data.get("subtitles")
    if items is None:
        raise ValueError(f"parse_elements: missing 'subtitles' key: {raw!r}")
    if not isinstance(items, list):
        raise ValueError(f"parse_elements: 'subtitles' must be a list, got {type(items).__name__}: {raw!r}")
    result = []
    for item in items:
        try:
            result.append(SubtitleElement.model_validate(item))
        except ValueError:
            raise ValueError(f"parse_elements: invalid item in 'subtitles': {item!r}")
    return result


def analyze_group(
    group: FrameGroup,
    client: OllamaClient,
    prompt: str,
) -> FrameAnalysis:
    raw = client.analyze(group.frame, system=prompt)
    log.debug("analyze [%s] raw → %r", group.frame.name, raw)
    elements = parse_elements(raw)
    if not elements:
        log.info("analyze [%s] (no elements)", group.frame.name)
    for el in elements:
        log.info(
            "analyze [%s] %-7s %-7s %-7s | %s",
            group.frame.name,
            el.position, el.color, el.style,
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
