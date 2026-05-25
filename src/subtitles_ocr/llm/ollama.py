import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from subtitles_ocr.llm.protocol import LlmCallFailed
from subtitles_ocr.retry import RetryConfig, retry_method

T = TypeVar("T", bound=BaseModel)


DEFAULT_LLM_RETRY = RetryConfig(max_retries=2, backoff_seconds=(1.0, 3.0))


def _is_llm_retryable(e: Exception) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code >= 500
    return isinstance(
        e,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            json.JSONDecodeError,
            ValidationError,
        ),
    )


class OllamaLlmClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        retry: RetryConfig = DEFAULT_LLM_RETRY,
        request_timeout_seconds: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = host
        self.retry = retry
        self.request_timeout_seconds = request_timeout_seconds
        self._http = httpx.Client(
            base_url=host,
            timeout=request_timeout_seconds,
            transport=transport,
        )

    @retry_method(is_retryable=_is_llm_retryable)
    def _complete_once(
        self,
        prompt: str,
        response_schema: type[T],
        *,
        model: str,
    ) -> T:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        response = self._http.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return response_schema.model_validate_json(content)

    def complete(
        self,
        prompt: str,
        response_schema: type[T],
        *,
        model: str,
    ) -> T:
        try:
            return self._complete_once(prompt, response_schema, model=model)
        except Exception as e:
            if _is_llm_retryable(e):
                raise LlmCallFailed(
                    f"LLM call exhausted after {self.retry.max_retries + 1} attempts"
                ) from e
            raise
