class PipelineError(Exception):
    """Expected pipeline failure. Caught by the orchestrator, formatted cleanly,
    causes a non-zero exit. Exceptions outside this hierarchy are bugs and
    propagate as raw tracebacks."""

    def __init__(self, message: str, *, stage: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.hint = hint


class InputProbeError(PipelineError): ...        # Stage 1 — ffprobe / ffmpeg
class AspectRatioMismatch(PipelineError): ...    # Stage 1 — AR mismatch under --ar-strategy error
class AlignmentRatioTooLow(PipelineError): ...   # Stage 2 — orphan_ratio > 30%
class OcrDeviceInitError(PipelineError): ...     # Stage 6 — explicit device init fails
class LlmRetryExhausted(PipelineError): ...      # Stages 10 / 11
class LlmResponseSchemaError(PipelineError): ... # Stage 11 — event count / id mismatch
class LlmPromptTooLarge(PipelineError): ...      # Stage 11 — exceeds context window
class CacheCorruptionError(PipelineError): ...   # JsonlWriter — mid-file corruption
