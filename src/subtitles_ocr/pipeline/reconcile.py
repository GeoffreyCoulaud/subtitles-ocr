import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Iterator

from subtitles_ocr.models import SubtitleElement, SubtitleEvent
from subtitles_ocr.vlm.client import OllamaClient
from subtitles_ocr.vlm.prompt import RECONCILE_PROMPT

log = logging.getLogger(__name__)


def _majority(values: list[str]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    for v in values:
        if counts[v] == max_count:
            return v
    return values[0]


def _mbr_text(texts: list[str]) -> str:
    best = texts[0]
    best_score = -1.0
    for candidate in texts:
        score = sum(SequenceMatcher(None, candidate, other).ratio() for other in texts) / len(texts)
        if score > best_score:
            best_score = score
            best = candidate
    return best


def _reconcile_text(texts: list[str], client: OllamaClient) -> str:
    if len(set(texts)) == 1:
        return texts[0]
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    try:
        return client.chat(f"Readings:\n{numbered}", system=RECONCILE_PROMPT).strip()
    except RuntimeError:
        log.warning("LLM reconciliation failed, falling back to MBR")
        return _mbr_text(texts)


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
) -> Iterator[SubtitleEvent]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(lambda c: _reconcile_cluster(c, client), clusters):
            yield result
