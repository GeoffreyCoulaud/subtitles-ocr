# subtitles-ocr

A work-in-progress harcoded subtitles extractor

The goal is to pass as an input a video with hardcoded subtitles (hardsubs) and output a clean, ready to use, `.ass` subtitles file.
Focus is primarily on fansubbed anime, where no official subs exist, but good quality raws have appeared.
Dialogue, lyrics and "forced" in-frame translations all count as subtitles.
Original text positions should not be altered. This is a pure extraction program.

## What is not a subtitle

- Forced text already in the source material (eg. credits, character names)
- In-scene text already in the source material (eg. signs)
- Generally, any other text not part of the subtitles

## Setup

### Prerequisites

- **Python 3.12+**
- **[ffmpeg](https://ffmpeg.org/download.html)** — used to extract frames and read video metadata
- **[Ollama](https://ollama.com)** — local VLM inference server

### Install

```bash
# Clone the repo
git clone https://github.com/GeoffreyCoulaud/subtitles-ocr
cd subtitles-ocr

# Install dependencies with uv
uv sync
```

### Pull the VLM models

The pipeline uses two models that never coexist in VRAM:

- **Pre-filter** (`moondream`, 1.7 GB) — fast yes/no pass to skip frames with no text
- **Analysis** (`qwen3-vl:8b`, 6.1 GB) — full subtitle extraction on frames that passed the pre-filter

```bash
ollama pull moondream
ollama pull qwen3-vl:8b
```

Both can be overridden with `--filter-model` and `--model`. Make sure Ollama is running before invoking the tool.

## Usage

```bash
uv run subtitles-ocr <video>
```

This produces `<video>.ass` next to the input file, and a `<video>_subtitles_ocr/` work directory containing intermediate files (frames, analysis JSONL, etc.).

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `<video>.ass` | Path to the output `.ass` file |
| `-w`, `--workdir` | `<video>_subtitles_ocr/` | Directory for intermediate files |
| `-m`, `--model` | `qwen3-vl:8b` | Ollama model for subtitle extraction |
| `--filter-model` | `moondream` | Ollama model for the pre-filter pass |
| `--filter-workers` | `4` | Parallel workers for the pre-filter pass |

### Example

```bash
# Basic usage
uv run subtitles-ocr episode01.mkv

# Custom output path and model
uv run subtitles-ocr episode01.mkv -o subs/episode01.ass -m llava:13b
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run python -m pytest
```