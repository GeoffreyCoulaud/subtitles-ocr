import re
from concurrent.futures import ThreadPoolExecutor
from subtitles_ocr.models import FrameGroup
from subtitles_ocr.vlm.client import OllamaClient


def prefilter_groups(
    groups: list[FrameGroup],
    client: OllamaClient,
    prompt: str,
    workers: int,
) -> list[bool]:
    def classify(group: FrameGroup) -> bool:
        try:
            response = client.analyze(group.frame, prompt)
            low = response.lower()
            if re.search(r"\byes\b", low):
                return True
            if re.search(r"\bno\b", low):
                return False
            return True  # ambiguous → conservative, keep frame
        except RuntimeError:
            return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(classify, groups))
