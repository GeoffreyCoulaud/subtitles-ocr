# src/subtitles_ocr/pipeline/resume.py
import json
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def resume_from_jsonl(
    elements: list[T],
    path: Path,
    element_id: Callable[[T], str],
) -> tuple[list[str], list[T]]:
    """Return (processed_lines_in_element_order, remaining_elements).

    Reads JSONL entries from path; matches each to an element via the "id"
    field. Returns already-processed raw lines sorted to original element
    order, and unprocessed elements in original order.
    """
    if not path.exists():
        return [], list(elements)

    lines_by_id: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        lines_by_id[json.loads(line)["id"]] = line

    processed: list[str] = []
    remaining: list[T] = []
    for element in elements:
        eid = element_id(element)
        if eid in lines_by_id:
            processed.append(lines_by_id[eid])
        else:
            remaining.append(element)

    return processed, remaining
