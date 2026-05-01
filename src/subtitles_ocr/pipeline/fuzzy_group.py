from difflib import SequenceMatcher

from subtitles_ocr.models import SubtitleEvent


def _text_by_position(event: SubtitleEvent) -> dict[str, str]:
    result: dict[str, str] = {}
    for el in event.elements:
        result[el.position] = result.get(el.position, "") + el.text
    return result


def _events_similar(a: SubtitleEvent, b: SubtitleEvent, threshold: float) -> bool:
    pos_a = {el.position for el in a.elements}
    pos_b = {el.position for el in b.elements}
    if pos_a != pos_b:
        return False
    texts_a = _text_by_position(a)
    texts_b = _text_by_position(b)
    return all(
        SequenceMatcher(None, texts_a[pos], texts_b[pos]).ratio() >= threshold
        for pos in pos_a
    )


def fuzzy_group_events(
    events: list[SubtitleEvent],
    similarity_threshold: float,
    gap_tolerance: float,
) -> list[list[SubtitleEvent]]:
    clusters: list[list[SubtitleEvent]] = []
    current: list[SubtitleEvent] = []

    for event in events:
        if not event.elements:
            continue

        if not current:
            current = [event]
            continue

        last = current[-1]
        gap = event.start_time - last.end_time

        if gap <= gap_tolerance and _events_similar(last, event, similarity_threshold):
            current.append(event)
        else:
            clusters.append(current)
            current = [event]

    if current:
        clusters.append(current)

    return clusters
