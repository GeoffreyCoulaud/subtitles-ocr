# Development

## Install

```bash
uv sync --extra dev
```

## Run tests

```bash
# All tests
uv run python -m pytest

# Single test file
uv run python -m pytest tests/test_prefilter.py

# Single test
uv run python -m pytest tests/test_prefilter.py::test_yes_response_returns_true
```

**Important:** always use `uv run python -m pytest`, never `uv run pytest`. `uv run pytest` may invoke a system-level pytest instead of the project's own, leading to wrong results.

## Run the CLI

```bash
uv run subtitles-ocr <video.mkv>
```

## Notes

- `docs/superpowers/` is gitignored (specs and plans from brainstorming sessions).