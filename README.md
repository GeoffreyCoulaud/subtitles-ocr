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

## How it works

The pipeline runs 8 sequential steps. Every step writes its output to the work directory, so interrupted runs resume from the last completed step — delete an intermediate file to rerun from that point.

| Step | Name | Output | Description |
|------|------|--------|-------------|
| 1 | Extract | `manifest.json`, `video_info.json` | ffmpeg extracts every frame at native FPS |
| 2 | pHash filter | `groups.jsonl` | Consecutive frames with an identical perceptual hash are collapsed into one group |
| 3 | Pre-filter | `filter.jsonl` | `llava:7b` classifies each group as containing text or not — fast binary pass to skip blank frames |
| 4 | Analyze | `analysis.jsonl` | `qwen2.5vl:3b` extracts text, style, color, position, and alignment from each text-bearing group |
| 5 | Group events | `events.json` | Consecutive identical analyses are merged into subtitle events |
| 6 | Fuzzy group | `fuzzy_groups.jsonl` | Similar events are clustered using trigram similarity; short gaps between similar events are bridged |
| 7 | Reconcile | `reconciled.jsonl` | Each cluster is collapsed into one canonical event — `gemma3:1b-it-qat` reconciles noisy text readings; majority vote picks style/color/alignment |
| 8 | Serialize | `<output>.ass` | The reconciled events are written to an ASS subtitle file |

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

The pipeline uses three models. The two vision models never coexist in VRAM:

- **Pre-filter** (`llava:7b`, 4.7 GB) — fast yes/no pass to skip frames with no text
- **Analysis** (`qwen2.5vl:3b`, 3.2 GB) — full subtitle extraction on frames that passed the pre-filter
- **Reconciliation** (`gemma3:1b-it-qat`, ~300 MB) — text-only model that merges OCR variations across frames into clean subtitle text

```bash
ollama pull llava:7b
ollama pull qwen2.5vl:3b
ollama pull gemma3:1b-it-qat
```

All three can be overridden with `--filter-model`, `--model`, and `--reconcile-model`. Make sure Ollama is running before invoking the tool.

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
| `-m`, `--model` | `qwen2.5vl:3b` | Ollama model for subtitle extraction |
| `--filter-model` | `llava:7b` | Ollama model for the pre-filter pass |
| `--filter-workers` | `4` | Parallel workers for the pre-filter pass |
| `--analyze-workers` | `1` | Parallel workers for the analysis pass (requires `OLLAMA_NUM_PARALLEL` ≥ value in Ollama's env) |
| `--hash-distance` | `10` | pHash distance threshold for frame grouping |
| `--similarity-threshold` | `0.75` | Trigram similarity threshold for fuzzy event grouping |
| `--gap-tolerance` | `0.5` | Max gap in seconds to bridge between similar events |
| `--reconcile-model` | `gemma3:1b-it-qat` | Ollama model for text reconciliation |
| `--reconcile-workers` | `8` | Parallel workers for reconciliation |
| `--ollama-host` | `http://localhost:11434` | Base URL of the Ollama server or LiteLLM proxy |

### Example

```bash
# Basic usage
uv run subtitles-ocr episode01.mkv

# Custom output path and model
uv run subtitles-ocr episode01.mkv -o subs/episode01.ass -m llava:13b
```

## Distributed inference

By default the tool talks to a local Ollama instance at `http://localhost:11434`. Pass `--ollama-host` to point it at any OpenAI-compatible endpoint — another machine running Ollama, or a [LiteLLM](https://github.com/BerriAI/litellm) proxy that fans out to multiple machines.

### Single remote machine

```bash
uv run subtitles-ocr episode01.mkv --ollama-host http://gpu-server:11434
```

### Multiple machines with LiteLLM proxy

LiteLLM proxy runs as a standalone service (not part of this codebase). It routes each model to the right machines and uses **least-busy** load balancing: a faster machine finishes requests sooner, frees its slot, and naturally receives more traffic — no manual weights needed.

**Install and start the proxy:**

```bash
pip install litellm
litellm --config litellm.yaml
```

**Example `litellm.yaml`:**

```yaml
router_settings:
  routing_strategy: least-busy

model_list:
  # llava:7b is large — only the machine with enough VRAM
  - model_name: llava:7b
    litellm_params:
      model: ollama/llava:7b
      api_base: http://big-machine:11434

  # smaller models can run on all machines
  - model_name: qwen2.5vl:3b
    litellm_params:
      model: ollama/qwen2.5vl:3b
      api_base: http://big-machine:11434
  - model_name: qwen2.5vl:3b
    litellm_params:
      model: ollama/qwen2.5vl:3b
      api_base: http://small-machine:11434

  - model_name: gemma3:1b-it-qat
    litellm_params:
      model: ollama/gemma3:1b-it-qat
      api_base: http://big-machine:11434
  - model_name: gemma3:1b-it-qat
    litellm_params:
      model: ollama/gemma3:1b-it-qat
      api_base: http://small-machine:11434

  # catch-all: any custom model (--model, --filter-model, --reconcile-model)
  # not listed above is forwarded to the main machine as-is
  - model_name: "*"
    litellm_params:
      model: "ollama/*"
      api_base: http://big-machine:11434
```

**Point the tool at the proxy:**

```bash
uv run subtitles-ocr episode01.mkv --ollama-host http://localhost:4000
```

Each machine must have the relevant models pulled (`ollama pull <model>`) before the proxy starts routing to it.

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run python -m pytest
```