# ADR-0004: Shared Infrastructure

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: Designed, not yet implemented. Implementation of this ADR is the prerequisite for parallel work on the pipeline stages of ADR-0002 / ADR-0003.
Origin: Pick-my-brain follow-up session between user and Claude, building on ADR-0001 / 0002 / 0003.
Supersedes: nothing.
Revises: nothing. This ADR is purely additive — it fills the cross-cutting infrastructure gap that ADR-0001 / 0002 / 0003 presumed but did not specify.

This ADR defines the conventions, abstractions, and utilities shared across all pipeline stages. Its purpose is to enable parallel implementation of stages without divergence on cross-cutting concerns (module layout, stage signatures, configuration plumbing, persistence, error handling, logging, retry, testing).

---

## 1. Problem

ADR-0001, ADR-0002, ADR-0003 specify the pipeline stage-by-stage with strong algorithmic detail and well-defined inter-stage Pydantic schemas. They do not specify the surrounding infrastructure that every stage needs: how a stage is invoked, how it receives configuration, how it persists its cache sidecar, what exceptions it raises, how it logs, how its tests are structured. Without this, parallel work on stages would produce divergent conventions and an expensive post-merge refactor.

This ADR fixes that infrastructure up front, in a small synchronous preliminary phase, before any parallel work on stage implementations begins.

## 2. Architectural principle: synchronous preliminary phase

Three implementation strategies were considered:

- **Minimal standardisation** — fix only conventions and inter-stage schemas; let each stage re-implement its sidecar, JSONL handling, config loading, etc. Three similar lines per stage tolerated.
- **Standardisation + a priori shared utilities** — fix conventions *and* implement a small set of shared utilities first (`JsonlWriter`, `BaseMeta`, `PipelineConfig` machinery, exception hierarchy, retry decorator). Synchronous prep phase, then parallel stage work.
- **Standardisation + a posteriori extraction** — fix conventions only; refactor duplicated patterns out after the fact.

Decision: **option 2** (a priori shared utilities). The set of utilities needed is small, universally required, and the cost of producing them up front is contained. Post-hoc refactoring across many parallel branches would be more expensive than the YAGNI cost of writing them now.

The remainder of this ADR specifies the conventions and the utilities. Section 17 lists the concrete artifacts that must exist (with tests) before parallel stage work begins.

## 3. Stage architecture

### 3.1 Stage = class with `__init__(deps)` + `run(globals, config)`

Each cacheable stage is a class. Constructor parameters carry external dependencies (LLM client, OCR engine, ffmpeg runner) with `None` defaults that fall back to the production implementation. `run()` carries the per-execution data and config.

```python
class GroupStage:
    """Stage with no external dependency."""

    def run(self, globals: PipelineGlobals, config: GroupConfig) -> GroupResult: ...


class OcrStage:
    def __init__(self, ocr_engine: OcrEngine | None = None) -> None:
        self.ocr_engine = ocr_engine or PaddleOcrEngine()

    def run(self, globals: PipelineGlobals, config: OcrConfig) -> OcrResult: ...


class EventCleanupStage:
    def __init__(self, llm: LlmClient | None = None) -> None:
        self.llm = llm or OllamaLlmClient()

    def run(self, globals: PipelineGlobals, config: EventCleanupConfig) -> EventCleanupResult: ...
```

Properties:

- **Constructor injection.** Tests construct a stage with fake deps that implement the relevant `typing.Protocol`. No monkeypatching of module-level functions. No service locator. No `contextvars`.
- **Signature `run(globals, config) -> Result` is preserved across stages.** Stages without external deps have an empty `__init__`. Stages with deps take them as keyword arguments with sensible defaults.
- **Stateless after init.** No instance attribute is mutated by `run()`. Stages exist to hold deps, not state. If a stage needs runtime state, that is a design bug.
- **Stage owns its cache.** `run()` reads inputs from `globals.workdir`, reads its own sidecar, decides cache hit or miss, recomputes if needed, writes output + sidecar. The orchestrator never inspects sidecars.

### 3.2 Composite streamed library: `frame_processing`

The streamed Stages 3 / 4 / 5 (diff / mask / compose) of ADR-0002 are *not* stages in the orchestrator's view. They form a library exposed by `pipeline/frame_processing/`:

```python
def iter_composed_frames(
    globals: PipelineGlobals,
    alignment_result: AlignmentResult,
    config: FrameProcessingConfig,
    start_at_fansub_idx: int = 0,
) -> Iterator[ComposedFrame]: ...


@dataclass
class ComposedFrame:
    fansub_frame_idx: int
    image: np.ndarray  # RGB uint8, H×W×3
```

`ComposedFrame` is a runtime-only dataclass (never persisted, never serialized). Pydantic is reserved for persisted boundaries.

`iter_composed_frames` is consumed by `OcrStage` directly. The orchestrator never sees `frame_processing`. The workdir directories `03_diff/`, `04_mask/`, `05_compose/` only carry debug PNGs under `--debug-images`; they are empty in production.

`OcrStage`'s sidecar includes `FrameProcessingConfig` in its cache-key payload, because `OcrStage` invokes `iter_composed_frames` and is the architectural owner of the diff→mask→compose→OCR pipe. This is the documented exception to the strict isolation rule: *if a stage invokes another module as a library, it includes that library's config in its sidecar.*

### 3.3 Orchestrator

```python
# cli.py (sketch)

def build_stages() -> list[PipelineStage]:
    return [
        ConformStage(),
        AlignmentStage(),
        OcrStage(),
        GroupStage(),
        AnimationStage(),  # MVP: present as stub returning empty, per ADR-0003
        ColorStage(),
        EventCleanupStage(),
        DocCleanupStage(),
        ExportStage(),
    ]


def run_pipeline(globals: PipelineGlobals, config: PipelineConfig) -> None:
    for stage in build_stages():
        stage_config = config.section_for(stage)
        stage.run(globals, stage_config)
```

Sequential execution. No data threading between stages. The orchestrator's only responsibility besides invocation is the top-level `try`/`except PipelineError` (cf. §7).

`config.section_for(stage)` is a small helper that returns the correct sub-config (`config.ocr` for `OcrStage`, etc.). Implementation detail: a class-level attribute `CONFIG_FIELD = "ocr"` on each stage, used by `section_for` to do the lookup.

## 4. Configuration

### 4.1 `PipelineGlobals`

Data that is knowable at boot and useful to multiple stages. Carried as a Pydantic model, passed to every stage's `run()`:

```python
class PipelineGlobals(BaseModel):
    workdir: Path
    hardsub_path: Path
    raw_path: Path
    out_path: Path
    fps: Fraction              # probed via ffprobe at boot, never float
    fansub_width: int
    fansub_height: int
    fansub_total_frames: int
    debug_images: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)  # for Fraction
```

`fps` is a `Fraction` (24000/1001 = 23.976), never a `float`. The rounding error of `float(23.97)` accumulates visibly over 35,000-frame episodes.

Principle: *a global is data knowable at boot AND useful to multiple stages.* Data that does not satisfy both criteria belongs in the producing stage's sidecar, not in `PipelineGlobals`. This is the guard rail against globals creep.

### 4.2 `PipelineConfig` (nested)

One Pydantic sub-model per stage, plus root-level CLI args:

```python
class PipelineConfig(BaseModel):
    # Root-level — wiring
    ar_strategy: Literal["error", "letterbox", "crop"] = "error"
    synopsis_path: Path | None = None
    color_cluster_threshold: float = 10.0

    # Per-stage sub-configs
    conform: ConformConfig = Field(default_factory=ConformConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    frame_processing: FrameProcessingConfig = Field(default_factory=FrameProcessingConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    group: GroupConfig = Field(default_factory=GroupConfig)
    animation: AnimationConfig = Field(default_factory=AnimationConfig)
    color: ColorConfig = Field(default_factory=ColorConfig)
    event_cleanup: EventCleanupConfig = Field(default_factory=EventCleanupConfig)
    doc_cleanup: DocCleanupConfig = Field(default_factory=DocCleanupConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
```

Each stage receives *only* its sub-config (via `config.section_for(stage)`), enforcing the strict isolation rule: stage X cannot read stage Y's config.

The CLI argparse layer is structured in groups (`--alignment-*`, `--ocr-*`, `--color-*`) that map 1:1 onto the sub-models. A small helper converts argparse namespace → nested `PipelineConfig`.

### 4.3 `NoCacheKey` annotation

Most config fields invalidate the cache when changed. A few do not (parallelism, device with auto-fallback, progress-bar style). Annotation-based opt-out:

```python
class NoCacheKey:
    """Marker indicating that changes to this field do not invalidate the stage cache."""


class OcrConfig(BaseModel):
    language: str = "latin"                                                # cache-invalidating
    device: Annotated[Literal["auto", "cuda", "rocm", "cpu"], NoCacheKey] = "auto"
    parallelism: Annotated[int, NoCacheKey] = 1


def cache_invalidating_dict(config: BaseModel) -> dict:
    out = {}
    for name, field in config.model_fields.items():
        if any(isinstance(m, NoCacheKey) for m in field.metadata):
            continue
        out[name] = getattr(config, name)
    return out
```

Default polarity: forgetting to annotate marks a field as cache-invalidating, which produces a false-negative cache hit (unnecessary recompute) rather than a false-positive (stale result). The safer error.

## 5. Persistence and cache

### 5.1 `BaseMeta` schema

Every stage writes a sidecar `*.meta.json` alongside its output. The sidecar schema is shared:

```python
class FileFingerprint(BaseModel):
    path: Path
    size: int
    mtime: float
    head_tail_hash: str | None = None
    full_hash: str | None = None


class BaseMeta(BaseModel):
    stage_name: str                                # e.g. "07_group"
    stage_version: int                             # manually bumped, see §5.3
    config: dict                                   # cache_invalidating_dict(stage_config)
    globals_subset: dict                           # fields of PipelineGlobals used by this stage
    input_fingerprints: dict[str, FileFingerprint] # keyed by stable input identifier
    written_at: datetime                           # informational only
```

Cache comparison at re-run: reconstruct a candidate `BaseMeta` from current config + current fingerprints; compare field-by-field against the persisted sidecar; mismatch on `stage_version`, `config`, `globals_subset`, or any `input_fingerprint` → invalidate and recompute. `written_at` is excluded from the comparison.

`globals_subset` rather than the full `PipelineGlobals` so that, e.g., changing `--debug-images` does not invalidate Stage 7 (which does not use it). Each stage declares its relevant globals as a class-level constant:

```python
class GroupStage:
    GLOBALS_USED: ClassVar[tuple[str, ...]] = ("fansub_width", "fansub_height")
```

### 5.2 `FileFingerprint`

Hybrid strategy keyed on file size:

- **Large files** (`> 1 MB`, e.g. videos): `mtime + size + head_tail_hash` (SHA-256 of first 1 MB + last 1 MB). Robust against `touch` and against most file content swaps; cost ~10 ms per file.
- **Small files** (`≤ 1 MB`, e.g. synopsis text): `mtime + size + full_hash` (SHA-256 of full content). Cost negligible.
- **Workdir intermediates** (controlled by the pipeline, never hand-edited): `mtime + size` only, no hash. Cheapest.

Single helper:

```python
def fingerprint(path: Path, full_hash_max_bytes: int = 1_000_000) -> FileFingerprint: ...
```

### 5.3 `STAGE_VERSION`

Each stage module exposes `STAGE_VERSION: int`, a manually-bumped constant. Any code change that may affect the stage's binary output must bump it. Mechanical alternatives (source-file hash, AST hash, git SHA) over-invalidate (reformatting / unrelated commits invalidate the cache). Manual bump forces explicit reflection on "did I change the output?" at commit time.

ADR convention: the `STAGE_VERSION` constant lives at the top of each stage module:

```python
# pipeline/group.py

STAGE_VERSION = 1
```

### 5.4 `JsonlWriter`

For slow stages with per-item resume (Stage 6 OCR, Stage 10 event cleanup):

```python
class JsonlWriter[T: BaseModel]:
    def __init__(
        self,
        path: Path,
        model_cls: type[T],
        fsync_every: int = 1,
    ) -> None: ...

    def resume_index(self) -> int: ...
    def iter_persisted(self) -> Iterator[T]: ...
    def append(self, item: T) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args) -> None: ...
```

Design:

- **No internal buffer.** Each `append()` immediately writes one line (`model_dump_json() + "\n"`) to the file.
- **`fsync_every`** knob controls fsync cadence (default 1 = fsync every line; can be raised if profiling shows it matters).
- **Resume**: `resume_index()` counts valid JSON lines; corruption mid-file (not just a partial last line) raises `CacheCorruptionError` (hard fail, surfaced to the user). The user removes the corrupted file or the whole workdir to recover.
- **SIGINT handling**: none custom. The writer is a context manager; `KeyboardInterrupt` traverses the `with`, `__exit__` runs `close()` which fsyncs the trailing data, the file remains coherent.
- **Ordering**: not a concern. Producers use `ProcessPoolExecutor.map()` or `ThreadPoolExecutor.map()`, both of which preserve submission order. The writer just appends what arrives.

Reference usage:

```python
with JsonlWriter(workdir / "06_ocr" / "results.jsonl", FrameOcrResult) as w:
    skip = w.resume_index()
    frames_to_do = aligned_frames[skip:]
    with ProcessPoolExecutor(max_workers=config.parallelism) as ex:
        for result in ex.map(ocr_frame, frames_to_do):
            w.append(result)
```

## 6. Concurrency

Concurrency is exclusively via `concurrent.futures` (`ThreadPoolExecutor`, `ProcessPoolExecutor`). No raw `threading` / `multiprocessing` / `asyncio` anywhere.

Rationale: `concurrent.futures.map()` preserves order, integrates trivially with `JsonlWriter`'s append-as-you-go model, and the higher-level abstraction is sufficient for every stage's parallelism needs (OCR, event cleanup, audio cross-correlation).

## 7. Errors

### 7.1 Hierarchy

All known pipeline failures inherit from `PipelineError`. Unknown exceptions (`KeyError`, `RuntimeError`, programmer bugs) propagate as Python tracebacks.

```python
# src/subtitles_ocr/exceptions.py

class PipelineError(Exception):
    """Expected pipeline failure. Caught by the orchestrator, formatted cleanly,
    causes a non-zero exit. Exceptions outside this hierarchy are bugs and
    propagate as raw tracebacks."""

    def __init__(self, message: str, *, stage: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.hint = hint


# Named subclasses for each documented hard-stop condition.
class InputProbeError(PipelineError): ...           # Stage 1 — ffprobe / ffmpeg
class AspectRatioMismatch(PipelineError): ...       # Stage 1 — AR mismatch under --ar-strategy error
class AlignmentRatioTooLow(PipelineError): ...      # Stage 2 — orphan_ratio > 30%
class OcrDeviceInitError(PipelineError): ...        # Stage 6 — explicit device init fails
class LlmRetryExhausted(PipelineError): ...         # Stages 10 / 11
class LlmResponseSchemaError(PipelineError): ...    # Stage 11 — event count / id mismatch
class LlmPromptTooLarge(PipelineError): ...         # Stage 11 — exceeds context window
class CacheCorruptionError(PipelineError): ...      # JsonlWriter — mid-file corruption
```

All subclasses live in `src/subtitles_ocr/exceptions.py`. No exceptions are defined elsewhere in the codebase.

### 7.2 Hard stop vs soft warning

The distinction is encoded by **control flow**, not by an exception attribute:

- **Soft warning** = `logger.warning(...)` + continue. No exception.
- **Hard stop** = `raise SubclassOfPipelineError(...)`. Caught at the top-level by the orchestrator, formatted, exit non-zero.

No `severity="warning" | "error"` field. No try/except branches that decide to continue or abort based on a flag. Code that continues simply does not raise; code that aborts raises.

### 7.3 Stage discipline

- A stage **does not catch** `PipelineError`. It catches third-party exceptions (ffmpeg, paddleocr, httpx) and converts them to a `PipelineError` subclass with `stage=` set and a `hint` when actionable.
- The orchestrator catches `PipelineError` at the top level, formats:
  ```
  [stage 02_alignment] AlignmentRatioTooLow: 32.5% of fansub frames are orphaned (threshold: 30%).
  Hint: provide --hardsub-skip / --raw-skip ranges to exclude known non-corresponding sections.
  ```
  Then `sys.exit(1)`.
- Anything outside `PipelineError` propagates unaltered. Bugs are not disguised as user-friendly errors.

## 8. Logging

### 8.1 Stack

`logging` stdlib. No `loguru`, no `structlog`. Each module declares `logger = logging.getLogger(__name__)` at the top. Setup lives exclusively in `cli.py`.

### 8.2 Format

```
2026-05-19T14:23:01.234+02:00 [INFO] [alignment] aligned 35832/35960 frames (99.6%), 128 orphans
```

ISO 8601 local timestamp with offset, millisecond precision, level in brackets, stage short name (last segment of the logger name) in brackets, message. No color, no emoji.

Implementation reference:

```python
class TzAwareFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None) -> str:
        return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")


class ShortNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.short_name = record.name.rsplit(".", 1)[-1]
        return True


def setup_logging(stdout_level: int, log_file: Path) -> None:
    fmt = "%(asctime)s [%(levelname)s] [%(short_name)s] %(message)s"
    formatter = TzAwareFormatter(fmt)
    short_filter = ShortNameFilter()

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setLevel(stdout_level)
    stdout_h.setFormatter(formatter)
    stdout_h.addFilter(short_filter)

    file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)
    file_h.addFilter(short_filter)

    root = logging.getLogger("subtitles_ocr")
    root.setLevel(logging.DEBUG)
    root.handlers = [stdout_h, file_h]
    root.propagate = False

    header = (
        f'==== run started {datetime.now().astimezone().isoformat(timespec="seconds")} '
        f'cmd="{shlex.join(sys.argv)}" ===='
    )
    file_h.stream.write(header + "\n")
    file_h.stream.flush()
```

### 8.3 Destinations

- **stdout**: INFO+ by default, DEBUG+ under `--debug`.
- **`workdir/pipeline.log`**: DEBUG+ always. Append mode, no rotation. Each run writes a header line `==== run started <ts> cmd="<argv>" ====` so multiple runs in a single file are visually delimited.

### 8.4 Progress bars

One `tqdm` per stage with `desc=<stage_short_name>`. Stages run sequentially, so progress bars do not nest. Cohabitation with `logger` is handled by `tqdm.contrib.logging.logging_redirect_tqdm` around any block where a pbar is active.

## 9. Frame indexing conventions

- **Zero-based.** First frame of any video has index `0`.
- **Half-open intervals** `[start, end)` everywhere a range is expressed. `range(start, end)` yields exactly the frames covered.
- **Naming**: `fansub_frame_idx` for fansub-side indices, `raw_frame_idx` for raw-side. Never plain `frame_idx`.
- **`fansub_total_frames` is a count**, not an index. A range covering the entire video writes `start=0, end=fansub_total_frames`.
- **`fps` is always a `Fraction`**, never a `float`. Helpers in `timing.py`:

```python
# src/subtitles_ocr/timing.py

def frame_to_ms(frame_idx: int, fps: Fraction) -> int:
    return int(round(frame_idx * 1000 / fps))


def ms_to_frame(ms: int, fps: Fraction) -> int:
    return int(round(ms * fps / 1000))
```

All time conversions go through these helpers. No ad-hoc `frame_idx * 1000 / 24` calculations anywhere.

## 10. External dependency injection

Every external dependency that a stage uses is fronted by a `typing.Protocol` in its own file, with the production implementation in a sibling file in the same sub-directory.

### 10.1 `LlmClient`

```python
# src/subtitles_ocr/llm/protocol.py

T = TypeVar("T", bound=BaseModel)


class LlmClient(Protocol):
    def complete(self, prompt: str, response_schema: type[T], *, model: str) -> T: ...


class LlmCallFailed(Exception):
    """Raised by LlmClient implementations when all retries are exhausted.
    Not a PipelineError; the caller converts to LlmRetryExhausted with stage context."""
```

```python
# src/subtitles_ocr/llm/ollama.py

class OllamaLlmClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        retry: RetryConfig = DEFAULT_LLM_RETRY,
        request_timeout_seconds: float = 60.0,
    ) -> None: ...

    def complete(self, prompt: str, response_schema: type[T], *, model: str) -> T: ...
```

### 10.2 `OcrEngine`

```python
# src/subtitles_ocr/ocr_engine/protocol.py

class OcrEngine(Protocol):
    def detect(self, image: np.ndarray) -> list[OcrDetection]: ...
```

```python
# src/subtitles_ocr/ocr_engine/paddle.py

class PaddleOcrEngine:
    def __init__(self, lang: str = "latin", device: str = "auto") -> None: ...
    def detect(self, image: np.ndarray) -> list[OcrDetection]: ...
```

### 10.3 `FfmpegRunner`

```python
# src/subtitles_ocr/ffmpeg/protocol.py

class FfmpegRunner(Protocol):
    def probe(self, path: Path) -> VideoMetadata: ...
    def transcode(self, args: TranscodeArgs) -> None: ...
    def extract_audio(self, path: Path, track_index: int, out: Path) -> None: ...
```

The exact shape of `VideoMetadata` and `TranscodeArgs` is decided when implementing Stage 1; this ADR only commits to the existence of those three methods.

### 10.4 Re-exports

Each sub-package's `__init__.py` re-exports the protocol and the default implementation:

```python
# src/subtitles_ocr/llm/__init__.py

from subtitles_ocr.llm.ollama import OllamaLlmClient
from subtitles_ocr.llm.protocol import LlmCallFailed, LlmClient

__all__ = ["LlmClient", "LlmCallFailed", "OllamaLlmClient"]
```

## 11. Retry

### 11.1 Generic decorator

```python
# src/subtitles_ocr/retry.py

class RetryConfig(BaseModel):
    max_retries: int
    backoff_seconds: tuple[float, ...]

    @model_validator(mode="after")
    def _lengths_match(self) -> Self:
        if len(self.backoff_seconds) != self.max_retries:
            raise ValueError("backoff_seconds must have exactly max_retries entries")
        return self


def retry_method(is_retryable: Callable[[Exception], bool]):
    """Method decorator: retries the wrapped method per self.retry config.
    On final exhaustion, re-raises the last retryable exception (the caller wraps
    into a domain-specific PipelineError). Non-retryable exceptions propagate
    immediately."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            config: RetryConfig = self.retry
            for attempt in range(config.max_retries + 1):
                try:
                    return fn(self, *args, **kwargs)
                except Exception as e:
                    if not is_retryable(e):
                        raise
                    if attempt == config.max_retries:
                        raise
                    delay = config.backoff_seconds[attempt]
                    logger.warning(
                        "%s attempt %d/%d failed (%s: %s). Retrying in %.1fs.",
                        fn.__qualname__, attempt + 1, config.max_retries + 1,
                        type(e).__name__, e, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator
```

### 11.2 LLM-specific use

```python
# src/subtitles_ocr/llm/ollama.py

DEFAULT_LLM_RETRY = RetryConfig(max_retries=2, backoff_seconds=(1.0, 3.0))


def _is_llm_retryable(e: Exception) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code >= 500
    return isinstance(e, (
        httpx.TimeoutException,
        httpx.NetworkError,
        json.JSONDecodeError,
        ValidationError,
    ))


class OllamaLlmClient:
    def __init__(self, host="http://localhost:11434",
                 retry: RetryConfig = DEFAULT_LLM_RETRY,
                 request_timeout_seconds: float = 60.0) -> None:
        self.retry = retry
        ...

    @retry_method(is_retryable=_is_llm_retryable)
    def _complete_once(self, prompt: str, schema: type[T], *, model: str) -> T:
        ...

    def complete(self, prompt: str, schema: type[T], *, model: str) -> T:
        try:
            return self._complete_once(prompt, schema, model=model)
        except Exception as e:
            if _is_llm_retryable(e):
                raise LlmCallFailed(
                    f"LLM call exhausted after {self.retry.max_retries + 1} attempts"
                ) from e
            raise
```

The calling stage converts `LlmCallFailed` into `LlmRetryExhausted` with stage context:

```python
# pipeline/event_cleanup.py

try:
    resp = self.llm.complete(prompt, CleanedEvent, model=config.model)
except LlmCallFailed as e:
    raise LlmRetryExhausted(
        f"LLM cleanup failed for event_id={event.event_id}",
        stage="10_event_cleanup",
        hint="Check Ollama logs and `--event-cleanup-model` availability.",
    ) from e
```

## 12. Module layout

```
src/subtitles_ocr/
  __init__.py
  cli.py                          # entry point, argparse, orchestration
  exceptions.py                   # PipelineError + named subclasses
  config.py                       # PipelineGlobals, PipelineConfig, StageConfig sub-models, NoCacheKey
  meta.py                         # BaseMeta, FileFingerprint, fingerprint(), cache_invalidating_dict()
  io.py                           # JsonlWriter
  timing.py                       # frame_to_ms, ms_to_frame
  retry.py                        # RetryConfig, @retry_method
  llm/
    __init__.py                   # re-exports LlmClient, LlmCallFailed, OllamaLlmClient
    protocol.py                   # LlmClient (Protocol), LlmCallFailed
    ollama.py                     # OllamaLlmClient (default impl), _is_llm_retryable, DEFAULT_LLM_RETRY
  ocr_engine/
    __init__.py
    protocol.py                   # OcrEngine (Protocol)
    paddle.py                     # PaddleOcrEngine (default impl)
  ffmpeg/
    __init__.py
    protocol.py                   # FfmpegRunner (Protocol), VideoMetadata, TranscodeArgs
    subprocess_runner.py          # SubprocessFfmpegRunner (default impl)
  pipeline/
    __init__.py
    conform.py                    # class ConformStage
    alignment/
      __init__.py                 # re-exports AlignmentStage
      stage.py                    # class AlignmentStage
      audio.py                    # silero-vad + hierarchical cross-correlation
      phash.py                    # fallback per-frame phash matcher
    frame_processing/
      __init__.py                 # re-exports iter_composed_frames, ComposedFrame
      iterator.py                 # iter_composed_frames + ComposedFrame dataclass
      diff.py
      mask.py
      compose.py
    ocr.py                        # class OcrStage
    group.py                      # class GroupStage
    animation.py                  # class AnimationStage (ADR-0003)
    color.py                      # class ColorStage
    event_cleanup.py              # class EventCleanupStage
    doc_cleanup.py                # class DocCleanupStage
    export.py                     # class ExportStage
tests/
  conftest.py                     # tmp_workdir, mock_globals
  ... (mirror of src/)
```

Rules:

- **No directory for a single file**, *except* when separating a Protocol from its default implementation. The two-file split is justified by representing two distinct conceptual units (interface + implementation).
- **Schemas are distributed per stage**, not centralized in a `models.py`. Each stage defines its `Result` and supporting Pydantic models in its own module. Imports form a DAG that mirrors the pipeline execution DAG: stage X may import types from stages preceding it, never from stages after it.
- **No god module.** `exceptions.py` is the documented exception (single file containing the full hierarchy), justified by its consumption pattern (orchestrator + tests need to import the whole set in one go).

## 13. Testing conventions

### 13.1 Stack

`pytest`. TDD obligatory per CLAUDE.md — no production code without a failing test first. `uv run python -m pytest`, never `uv run pytest`.

### 13.2 Fixtures

Hierarchical `conftest.py`:

- `tests/conftest.py` — global fixtures (`tmp_workdir`, `mock_globals`).
- `tests/pipeline/conftest.py` — pipeline-wide fixtures.
- `tests/pipeline/alignment/conftest.py`, etc. — stage-local fixtures (`populate_alignment_result`, `mock_ocr_jsonl`, …).

Reference root fixtures:

```python
# tests/conftest.py

@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    for d in ("01_conform", "02_alignment", "04_mask", "06_ocr",
              "07_group", "08_animation", "09_color",
              "10_event_cleanup", "11_doc_cleanup"):
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_globals(tmp_workdir: Path) -> PipelineGlobals:
    return PipelineGlobals(
        workdir=tmp_workdir,
        hardsub_path=Path("/dev/null/fake_hardsub.avi"),
        raw_path=Path("/dev/null/fake_raw.mkv"),
        out_path=tmp_workdir / "out.ass",
        fps=Fraction(24, 1),
        fansub_width=1920,
        fansub_height=1080,
        fansub_total_frames=1000,
        debug_images=False,
    )
```

### 13.3 Stage isolation pattern

Each stage test populates the immediate previous stage's output in `tmp_workdir`, constructs the stage with fake deps where applicable, calls `run()`, asserts on the resulting `Result` and on the written workdir files:

```python
def test_group_stage_basic(tmp_workdir, mock_globals):
    populate_ocr_jsonl(
        tmp_workdir / "06_ocr" / "results.jsonl",
        frames=[FrameOcrResult(...), ...],
    )
    populate_meta(tmp_workdir / "06_ocr" / "results.meta.json", ...)

    stage = GroupStage()
    result = stage.run(mock_globals, GroupConfig())

    assert len(result.events) == 1
    assert result.events[0].fansub_frame_start == 0
    assert (tmp_workdir / "07_group" / "events.json").exists()
```

### 13.4 DI in tests

Tests inject fake implementations of `LlmClient`, `OcrEngine`, `FfmpegRunner` via the stage constructor. No monkeypatching of module-level functions:

```python
class FakeLlm:
    def __init__(self, responses: list[BaseModel]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def complete(self, prompt, response_schema, *, model):
        self.call_count += 1
        return self.responses.pop(0)


def test_event_cleanup_with_fake_llm(tmp_workdir, mock_globals):
    fake = FakeLlm(responses=[CleanedEvent(text="hello")])
    stage = EventCleanupStage(llm=fake)
    result = stage.run(mock_globals, EventCleanupConfig())
    assert result.events[0].cleaned_text == "hello"
    assert fake.call_count == 1
```

### 13.5 Anti-patterns proscribed

- **Scripts containing only `print()` statements verify nothing.** All verification lives in `tests/` as pytest with `assert`. (Re-statement of CLAUDE.md rule, repeated here so agents who skip CLAUDE.md still encounter it.)
- **No committed binary fixtures.** All fixtures generated in-test via numpy (per ADR-0001 §12).
- **Mocking ollama / paddleocr / ffmpeg via monkeypatch is forbidden.** Use the Protocol + fake-impl pattern instead.

## 14. Preliminary synchronous phase — concrete deliverables

Before parallel stage implementation begins, the following must exist with passing tests:

1. **`src/subtitles_ocr/exceptions.py`** — `PipelineError` + all named subclasses listed in §7.1.
2. **`src/subtitles_ocr/config.py`** — `PipelineGlobals`, `NoCacheKey` marker, `PipelineConfig` root + all `StageConfig` sub-models (initially with empty fields, populated incrementally as each stage is implemented).
3. **`src/subtitles_ocr/meta.py`** — `BaseMeta`, `FileFingerprint`, `fingerprint()`, `cache_invalidating_dict()`. Tests cover all three fingerprint paths (large file head-tail-hash, small file full-hash, intermediate mtime+size only) and the cache-key annotation extraction.
4. **`src/subtitles_ocr/io.py`** — `JsonlWriter`. Tests cover happy path, resume after partial write, mid-file corruption (raises `CacheCorruptionError`), `fsync_every` knob.
5. **`src/subtitles_ocr/timing.py`** — `frame_to_ms`, `ms_to_frame`. Tests cover round-trip stability at 23.976 fps over 35,000-frame ranges (no drift).
6. **`src/subtitles_ocr/retry.py`** — `RetryConfig`, `@retry_method`. Tests cover happy path, retry-then-success, exhaustion-propagates-last, non-retryable-propagates-immediately, backoff timing (with `backoff_seconds=(0.0, 0.0)` for speed).
7. **`src/subtitles_ocr/llm/protocol.py`** — `LlmClient` Protocol, `LlmCallFailed`. No test (Protocol-only).
8. **`src/subtitles_ocr/llm/ollama.py`** — `OllamaLlmClient` with retry wired in. Tests cover: 200 → returns parsed model; 503 then 200 → returns; 503×3 → `LlmCallFailed`; 400 → propagates immediately; malformed JSON → triggers retry; schema mismatch → triggers retry. Real httpx mocked via its built-in `MockTransport`, *not* monkeypatched.
9. **`src/subtitles_ocr/ocr_engine/protocol.py`** — `OcrEngine` Protocol. No test.
10. **`src/subtitles_ocr/ffmpeg/protocol.py`** — `FfmpegRunner` Protocol. No test.
11. **`tests/conftest.py`** — `tmp_workdir`, `mock_globals` fixtures. Smoke test that asserts the fixtures construct correctly.
12. **`src/subtitles_ocr/cli.py`** — argparse → `PipelineGlobals` + `PipelineConfig` wiring, `setup_logging()`, stage orchestration loop. Tests cover argparse parsing (a handful of representative invocations) and the orchestrator's `try`/`except PipelineError` (with a fake stage that raises).

Deliverables 9 and 10 (`OcrEngine`, `FfmpegRunner` protocols) only commit to method names and signatures; their default implementations (`paddle.py`, `subprocess_runner.py`) are built as part of Stage 6 and Stage 1 implementation respectively, since their internal shape is intimately tied to the consuming stage's needs.

Old code to delete during this phase, per ADR-0001 §10:

- `src/subtitles_ocr/pipeline/prefilter.py` and its tests
- `src/subtitles_ocr/vlm/` (entire directory)
- `src/subtitles_ocr/pipeline/filter.py`
- `src/subtitles_ocr/models.py` (replaced by per-stage schema distribution)
- `src/subtitles_ocr/cli.py` (rewritten from scratch)

## 15. Out of scope (declined)

- **Configuration via YAML / TOML file.** CLI flags + Pydantic defaults suffice for the foreseeable scope.
- **Service locator or `contextvars` for dependency injection.** Rejected as ambient magic in favor of constructor injection.
- **Sophisticated retry strategies** (jitter, circuit breaker, exponential ramp). The two-retry-with-fixed-backoff baseline is sufficient for the LLM cleanup stages; no other stage needs retry today.
- **Structured logging (JSON output).** Plain-text format suffices for a single-user CLI tool.
- **Automatic stage versioning** (source hash, AST hash, git SHA). Manual `STAGE_VERSION` constant chosen for honesty and intentional control.
- **A central `protocols.py`.** Each Protocol lives next to its default impl in a dedicated sub-package, per §10.
- **Multi-provider LLM abstraction** (LiteLLM, AISuite, etc.). Ollama via OpenAI-compat is sufficient for the MVP. The `LlmClient` Protocol leaves the door open without committing.

## 16. Decision log (key forks from this session)

| Fork | Decision | Why |
|---|---|---|
| Level of abstraction | Synchronous preliminary phase with shared utilities extracted a priori (vs minimal conventions, vs a posteriori extraction) | Small bounded set of universal needs; cheaper than post-hoc refactor across parallel branches |
| Stage granularity (streamed sub-stages) | Composite library `frame_processing/` with one file per sub-step (diff / mask / compose) but a single public iterator | Forcing one-stage-per-numbered-directory is formalism; streamed sub-stages have no independent cache or config |
| Stage signature | Class with `__init__(deps)` + `run(globals, config) -> Result` (vs free function, vs function-with-deps-arg) | Clean constructor injection without polluting `run()` signature; uniform across stages with and without external deps |
| Cache ownership | Stage owns its sidecar comparison internally (vs orchestrator-driven) | Sidecar schema is the stage's affair; orchestrator-driven check would leak abstractions |
| Config structure | Nested Pydantic, one sub-model per stage, strict isolation (vs flat) | Sidecar slice = sub-model dump; tests construct minimal sub-config; CLI groups map 1:1 |
| Globals separation | `PipelineGlobals` separate from `PipelineConfig`; globals knowable at boot AND used by ≥2 stages | Prevents globals creep; clean lifecycle (no runtime-mutable globals) |
| Cache-key marking | `NoCacheKey` opt-out annotation, default cache-invalidating | Safer error polarity: forgetting an annotation means unnecessary recompute, not stale result |
| `fps` lifecycle | Probed at CLI boot, stored in `PipelineGlobals` as `Fraction` (vs Stage-1-produced, vs runtime mutable globals) | Avoids cross-stage sidecar coupling; `Fraction` eliminates float drift on long episodes |
| Input fingerprinting | Hybrid: head+tail hash for large files, full hash for small, mtime+size for intermediates | Cheap and robust enough; full-hashing 4 GB videos every cache check is intractable |
| Stage versioning | Manual `STAGE_VERSION` constant, bumped by author on output-affecting changes | Honest and intentional; mechanical alternatives (source hash, git SHA) over-invalidate |
| JSONL writer design | No internal buffer, per-item append, `fsync_every` knob | `concurrent.futures.map()` preserves order, so chunking buffer was redundant; per-item write is simpler and correct |
| Concurrency primitives | `concurrent.futures` only (no raw `threading` / `multiprocessing` / `asyncio`) | Higher-level abstraction is sufficient and uniform across stages |
| Exception hierarchy | `PipelineError` + named subclasses, all in `exceptions.py` | Discoverability for agents implementing new stages; orchestrator imports one module |
| Hard stop vs soft warning | Encoded by control flow (`raise` vs `logger.warning`), not by an exception attribute | Simpler and auditable; no try/except branches that decide severity from a flag |
| LLM error surface | `LlmCallFailed` (non-`PipelineError`) raised by client; stage caller converts to `LlmRetryExhausted` with stage context | Client cannot know which stage called it; cleaner than mutating exception attributes post-hoc |
| Logging stack | `logging` stdlib (vs loguru, vs structlog) | Zero new dependency; setup is one function in `cli.py` |
| Log destinations | stdout (INFO+, configurable) + `workdir/pipeline.log` (DEBUG+, append, no rotation) | Append preserves crash-then-retry history without user maintenance |
| Log timestamp | ISO 8601 local with offset, millisecond precision | Unambiguous (offset eliminates DST), readable without mental conversion, ms-precision needed at OCR speed |
| Log header | `==== run started <ts> cmd="<argv>" ====` at each run open | Visual delimiter in append-mode file; preserves the invocation for post-mortem |
| Frame indexing | Zero-based, half-open `[start, end)`, `fansub_frame_idx` / `raw_frame_idx`, `Fraction` fps | Eliminates off-by-one and float-drift footguns documented across stages |
| Dependency injection | Constructor injection via class `__init__` + `typing.Protocol` (vs monkeypatch, vs service locator) | Tests construct fakes explicitly; no ambient magic |
| Protocol + impl layout | Sub-package per external dep, `protocol.py` + `<technology>.py` (vs flat `protocols.py`, vs single-file co-location) | "One file per conceptual unit" + clean re-exports |
| Retry implementation | Generic `@retry_method` decorator in `retry.py`, reads `self.retry` (vs inline in client, vs hardcoded constants) | Reusable, testable in isolation, configurable per-instance |
| Schema distribution | Per-stage modules define their own Pydantic schemas (vs central `models.py`) | Cohesion: a stage's `Result` is its public API; eliminates the god-file model class |
| Testing strategy | DI-driven, monkeypatch forbidden, conftest hierarchical, populate-previous-stage-output pattern | Tests exercise real I/O and serialization paths; failures point to design bugs not test plumbing |

## 17. Conversation provenance

This ADR was produced by a `pick-my-brain` session triggered by the user's observation that ADR-0001 / 0002 / 0003 are excellent per-stage specs but do not lock down the cross-cutting infrastructure needed for parallel implementation. The user's stated preference at session start: "phase préliminaire synchrone plutôt que du n'importe quoi parallèle."

The session traversed 15 structured questions in order: (1) level of abstraction, (2) stage granularity, (3) stage signature, (4) config structure, (5) `fps` lifecycle and globals, (6) `BaseMeta` schema + fingerprint strategy + stage versioning, (7) `JsonlWriter` design (including a productive user pushback that eliminated chunking in favor of per-item append, after the agent surfaced that `.map()` ordering removed the buffer's primary motivation), (8) `frame_processing` status (library vs stage), (9) exception hierarchy, (10) logging and progress bars, (11) frame indexing conventions, (12) module layout, (13) testing conventions (including a productive user pushback that replaced monkeypatch with constructor injection — leading to the class-based stage refactor of Q3), (14) retry policy (with another user pushback that moved retry from inline to a generic decorator), (15) log format precise.

Three inflection moments where the user pushed back productively:

- **Q7** (JSONL chunking): user asked whether chunking served any purpose with `concurrent.futures.map()` preserving order. The agent reconsidered and confirmed chunking was largely vestigial; design was simplified to per-item append.
- **Q13.3** (mocking strategy): user rejected monkeypatch in favor of clean DI. This forced a structural revision of Q3 — stages became classes (with `__init__(deps)`) instead of free functions, to enable constructor injection without polluting `run()`.
- **Q12** (module layout, post-revision): user required Protocol and default impl in separate files, prompting the sub-package pattern (`llm/protocol.py` + `llm/ollama.py`) instead of co-located definitions.

These pushbacks materially improved the design and are reflected in §3, §10, §11.
