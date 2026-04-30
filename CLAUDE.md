# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Iron rules

- **Never use `python` or `pip` directly.** Every Python invocation goes through `uv run`. Every dependency change goes through `uv add` / `uv remove`. Every install goes through `uv sync`. No exceptions unless a specific operation has no `uv` equivalent — in that case, document why before doing it.
- **Never edit `pyproject.toml` dependencies by hand.** Use `uv add <pkg>` and `uv remove <pkg>` so the lockfile stays consistent.
- **`uv run python -m pytest`, never `uv run pytest`.** The system Python 3.14 shadows the venv Python 3.12 when invoking `pytest` directly.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run python -m pytest

# Run a single test file
uv run python -m pytest tests/test_prefilter.py

# Run a single test
uv run python -m pytest tests/test_prefilter.py::test_yes_response_returns_true

# Run the CLI
uv run subtitles-ocr <video.mkv>
```

**Important:** always use `uv run python -m pytest`, never `uv run pytest`. The system has Python 3.14 which conflicts with the venv Python 3.12 — `uv run pytest` picks up the wrong interpreter.

Never use `pip` directly — always `uv`.

`docs/superpowers/` is gitignored (specs and plans from brainstorming sessions).

## Architecture

The tool runs a 6-step sequential pipeline, fully resumable: each step checks its output file and skips if already done. Intermediate files live in a work directory (`<video>_subtitles_ocr/` by default).

```
[1] extract       → manifest.json, video_info.json   (ffmpeg, all frames at native FPS)
[2] pHash filter  → groups.jsonl                     (imagehash, groups consecutive identical frames)
[3] pre-filter    → filter.jsonl                     (smolvlm2:256m via Ollama, N parallel workers)
[4] analyze       → analysis.jsonl                   (qwen3-vl:8b via Ollama, sequential)
[5] group_events  → events.json                      (merges consecutive identical FrameAnalysis)
[6] serialize     → output.ass
```

Steps 3 and 4 use JSONL files as position-indexed checkpoints: line N = group N. Resume works by counting existing lines and slicing `groups[n_done:]`.

**Two-model design:** `smolvlm2:256m` (pre-filter) and `qwen3-vl:8b` (analyze) never coexist in VRAM — the pre-filter pass finishes completely before the analyze pass begins. This is intentional.

**Pre-filter contract:** conservative, zero false negatives. A response containing "no" (word boundary) → `False`. A response containing "yes" (word boundary) → `True`. Anything else (ambiguous, error, empty) → `True`. This is implemented with `re.search(r"\byes\b")` / `re.search(r"\bno\b")` — do not simplify to substring matching (`"no" in response` would match "cannot").

**Pydantic v2** throughout: use `model_validate()`, `model_validate_json()`, `model_dump(mode="json")`, `model_dump_json()`.

### Key files

- `src/subtitles_ocr/models.py` — all data models (`Frame`, `FrameGroup`, `FrameAnalysis`, `SubtitleElement`, `SubtitleEvent`, `VideoInfo`)
- `src/subtitles_ocr/cli.py` — orchestration, resume logic, CLI options
- `src/subtitles_ocr/vlm/client.py` — thin wrapper around `ollama.chat()`, raises `RuntimeError` on any failure
- `src/subtitles_ocr/vlm/prompt.py` — `SYSTEM_PROMPT` (analysis) and `PREFILTER_PROMPT` (pre-filter)
- `src/subtitles_ocr/pipeline/prefilter.py` — `ThreadPoolExecutor` parallel pre-filter
- `src/subtitles_ocr/pipeline/filter.py` — pHash grouping (`HASH_DISTANCE_THRESHOLD = 10`)

## Tests

Each pipeline module has a corresponding test file. Tests use `unittest.mock` — Ollama and ffmpeg are always mocked, never called in tests.

When writing new verification code: it belongs in `tests/` as a pytest test with `assert`, not as a standalone script. A script with only `print()` statements verifies nothing.
