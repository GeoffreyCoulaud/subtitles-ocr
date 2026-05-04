# src/subtitles_ocr/pipeline/reconcile.py
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from subtitles_ocr.models import SubtitleElement, SubtitleEvent
from subtitles_ocr.vlm.client import OllamaClient
from subtitles_ocr.vlm.prompt import RECONCILE_PROMPT
from subtitles_ocr.pipeline.retry import RetryConfig, RetryExhausted, NonRetryable, with_retry

log = logging.getLogger(__name__)


def _majority(values: list[str]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    for v in values:
        if counts[v] == max_count:
            return v
    raise AssertionError("unreachable")


def _reconcile_text(texts: list[str], client: OllamaClient) -> str:
    if len(set(texts)) == 1:
        return texts[0]
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return client.chat(f"Readings:\n{numbered}", system=RECONCILE_PROMPT).strip()


def _reconcile_cluster(cluster: list[SubtitleEvent], client: OllamaClient) -> SubtitleEvent:
    if len(cluster) == 1:
        return cluster[0]

    positions = sorted({el.position for event in cluster for el in event.elements})
    elements: list[SubtitleElement] = []

    for position in positions:
        all_els = [el for event in cluster for el in event.elements if el.position == position]
        elements.append(SubtitleElement(
            text=_reconcile_text([el.text for el in all_els], client),
            style=_majority([el.style for el in all_els]),
            color=_majority([el.color for el in all_els]),
            border_color=_majority([el.border_color for el in all_els]),
            alignment=_majority([el.alignment for el in all_els]),
            position=position,
        ))

    return SubtitleEvent(
        start_time=cluster[0].start_time,
        end_time=cluster[-1].end_time,
        elements=elements,
    )


def reconcile_groups(
    clusters: list[list[SubtitleEvent]],
    client: OllamaClient,
    workers: int,
    retry_config: RetryConfig | None = None,
) -> Iterator[SubtitleEvent | None]:
    if retry_config is None:
        retry_config = RetryConfig()

    def process(cluster: list[SubtitleEvent]) -> SubtitleEvent | None:
        try:
            return with_retry(lambda: _reconcile_cluster(cluster, client), retry_config, log)
        except NonRetryable as e:
            log.warning(
                "reconcile [cluster@%.3f] non-retryable: %s",
                cluster[0].start_time, e.__cause__,
            )
            return None
        except RetryExhausted:
            log.warning("reconcile [cluster@%.3f] retries exhausted", cluster[0].start_time)
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(process, clusters):
            yield result
