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

**Important:** always use `uv run python -m pytest`, never `uv run pytest`. The system Python 3.14 shadows the venv Python 3.12 when invoking `pytest` directly — `uv run pytest` picks up the wrong interpreter.

## Run the CLI

```bash
uv run subtitles-ocr <video.mkv>
```

## Notes

- `docs/superpowers/` is gitignored (specs and plans from brainstorming sessions).