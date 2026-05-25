from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LlmClient(Protocol):
    def complete(self, prompt: str, response_schema: type[T], *, model: str) -> T: ...


class LlmCallFailed(Exception):
    """Raised by LlmClient implementations when all retries are exhausted.
    Not a PipelineError; the caller converts to LlmRetryExhausted with stage context."""
