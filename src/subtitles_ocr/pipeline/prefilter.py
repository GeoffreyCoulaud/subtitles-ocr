# src/subtitles_ocr/pipeline/prefilter.py
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

from subtitles_ocr.models import FrameGroup
from subtitles_ocr.vlm.client import OllamaClient
from subtitles_ocr.pipeline.retry import RetryConfig, RetryExhausted, NonRetryable, with_retry

log = logging.getLogger(__name__)


def prefilter_groups(
    groups: list[FrameGroup],
    client: OllamaClient,
    prompt: str,
    workers: int,
    retry_config: RetryConfig | None = None,
) -> Generator[bool | None, None, None]:
    if retry_config is None:
        retry_config = RetryConfig()
    def classify(group: FrameGroup) -> bool | None:
        def _attempt() -> bool:
            response = client.analyze(group.frame, prompt, json_mode=True)
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError(f"expected JSON object: {response!r}")
            result = data.get("has_text")
            if isinstance(result, str):
                coerced = {"true": True, "false": False}.get(result.lower())
                if coerced is not None:
                    return coerced
                raise ValueError(f"unrecognized has_text string: {result!r}")
            if isinstance(result, bool):
                return result
            raise ValueError(f"has_text missing or wrong type: {result!r}")

        try:
            return with_retry(_attempt, retry_config, log)
        except NonRetryable as e:
            log.warning("prefilter [%s] non-retryable error: %s", group.frame.name, e.__cause__)
            return None
        except RetryExhausted:
            log.warning("prefilter [%s] retries exhausted", group.frame.name)
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(classify, groups):
            yield result
