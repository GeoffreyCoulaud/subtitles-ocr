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
            return "yes" in response.lower()
        except RuntimeError:
            return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(classify, groups))
