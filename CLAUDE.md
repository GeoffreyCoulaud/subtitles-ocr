# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start here

**Always read `README.md` first.** It is the canonical source for project context: what the tool does, the pipeline steps, models used, CLI options, and setup instructions. The README may link to sub-documents under `docs/`.

## Iron rules

- **Never use `python` or `pip` directly.** Every Python invocation goes through `uv run`. Every dependency change goes through `uv add` / `uv remove`. Every install goes through `uv sync`. No exceptions unless a specific operation has no `uv` equivalent — in that case, document why before doing it.
- **Never edit `pyproject.toml` dependencies by hand.** Use `uv add <pkg>` and `uv remove <pkg>` so the lockfile stays consistent.
- **`uv run python -m pytest`, never `uv run pytest`.** `uv run pytest` may invoke a system-level pytest instead of the project's own.
- **Never run a script that only prints to verify reasoning.** A `python -c` (or any script) containing only `print()` statements narrates a conclusion — it does not verify anything. Real verification means executing the code under test and asserting on its behavior. All verification belongs in `tests/` as pytest tests with `assert`. This rule applies to subagents (e.g. code-reviewer) too.
- **Keep `README.md` up to date.** Any change that affects documented behavior — pipeline steps, model names or sizes, CLI options, intermediate file names — must be reflected in the README in the same commit.
- **Skip spec review for TDD tasks.** Passing tests prove spec compliance — a separate spec review is redundant.
- **Code quality review is only required when the task produces executable logic** (functions, branches, conditionals). Skip it for dependency additions, doc-only edits, renames, and config-only changes.

## Commands

See [docs/development.md](docs/development.md) for install, test, and run commands.

**Critical:** always use `uv run python -m pytest`, never `uv run pytest`. `uv run pytest` may invoke a system-level pytest instead of the project's own.

`docs/superpowers/` is gitignored (specs and plans from brainstorming sessions).

## Code-level details

**Pre-filter contract:** conservative, zero false negatives. Uses `json_mode=True`; expects `{"has_text": true/false}`. A boolean `has_text` field → use its value directly. Anything else (parse error, missing/non-boolean field, API error) → `True`. This keeps the conservative invariant without regex heuristics.

**Pydantic v2** throughout: use `model_validate()`, `model_validate_json()`, `model_dump(mode="json")`, `model_dump_json()`.

### Key files

- `src/subtitles_ocr/models.py` — all data models (`Frame`, `FrameGroup`, `FrameAnalysis`, `SubtitleElement`, `SubtitleEvent`, `VideoInfo`)
- `src/subtitles_ocr/cli.py` — orchestration, resume logic, CLI options
- `src/subtitles_ocr/vlm/client.py` — thin wrapper around `openai.chat()`, raises `RuntimeError` on any failure
- `src/subtitles_ocr/vlm/prompt.py` — `SYSTEM_PROMPT` (analysis) and `PREFILTER_PROMPT` (pre-filter)
- `src/subtitles_ocr/pipeline/prefilter.py` — `ThreadPoolExecutor` parallel pre-filter
- `src/subtitles_ocr/pipeline/filter.py` — edge-similarity grouping

## Tests

Each pipeline module has a corresponding test file. Tests use `unittest.mock` — Ollama and ffmpeg are always mocked, never called in tests.

When writing new verification code: it belongs in `tests/` as a pytest test with `assert`, not as a standalone script. A script with only `print()` statements verifies nothing.
