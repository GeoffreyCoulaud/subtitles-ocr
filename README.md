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

### Pull a VLM model

Any multimodal Ollama model works. Recommended options (best to lightest):

| Model | Size | Notes |
|-------|------|-------|
| `qwen3-vl:8b` | ~5 GB | Best general vision model (default) |
| `qwen2.5vl:7b` | ~5 GB | Previous generation, still very capable |
| `deepseek-ocr:3b` | ~2 GB | Lightweight, OCR-specialized |

```bash
ollama pull qwen3-vl:8b
```

Make sure Ollama is running before invoking the tool.

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
| `-m`, `--model` | `qwen2-vl:7b` | Ollama model to use |

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
uv run pytest
```