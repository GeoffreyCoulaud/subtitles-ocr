import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from subtitles_ocr.models import FrameGroup
from subtitles_ocr.vlm.client import OllamaClient


def prefilter_groups(
    groups: list[FrameGroup],
    client: OllamaClient,
    prompt: str,
    workers: int,
) -> Generator[bool, None, None]:
    error_count = 0
    lock = threading.Lock()

    def classify(group: FrameGroup) -> bool:
        nonlocal error_count
        try:
            response = client.analyze(group.frame, prompt)
            low = response.lower()
            if re.search(r"\byes\b", low):
                return True
            if re.search(r"\bno\b", low):
                return False
            return True  # ambiguous → conservative, keep frame
        except RuntimeError:
            with lock:
                error_count += 1
            return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(classify, groups):
            yield result

    if error_count > 0 and error_count == len(groups):
        raise RuntimeError(
            f"Pre-filter model '{client.model}' failed on all {len(groups)} groups — "
            f"is it available? Run: ollama pull {client.model}"
        )
    if error_count > 0:
        print(f"Warning: {error_count}/{len(groups)} pre-filter calls failed, kept as conservative", file=sys.stderr)
