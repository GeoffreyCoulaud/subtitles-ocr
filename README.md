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

| Step | Name         | Output                             | Description                                                                                                                                       |
|------|--------------|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Extract      | `manifest.json`, `video_info.json` | ffmpeg extracts every frame at native FPS                                                                                                         |
| 2    | pHash filter | `groups.jsonl`                     | Consecutive frames with an identical perceptual hash are collapsed into one group                                                                 |
| 3    | Pre-filter   | `filter.jsonl`                     | `llava:7b` classifies each group as containing text or not — fast binary pass to skip blank frames                                                |
| 4    | Analyze      | `analysis.jsonl`                   | `qwen2.5vl:3b` extracts text, style, color, position, and alignment from each text-bearing group                                                  |
| 5    | Group events | `events.json`                      | Consecutive identical analyses are merged into subtitle events                                                                                    |
| 6    | Fuzzy group  | `fuzzy_groups.jsonl`               | Similar events are clustered using trigram similarity; short gaps between similar events are bridged                                              |
| 7    | Reconcile    | `reconciled.jsonl`                 | Each cluster is collapsed into one canonical event — `gemma3:1b-it-qat` reconciles noisy text readings; majority vote picks style/color/alignment |
| 8    | Serialize    | `<output>.ass`                     | The reconciled events are written to an ASS subtitle file                                                                                         |

## Setup

### Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — manages Python and dependencies
- **[ffmpeg](https://ffmpeg.org/download.html)** — used to extract frames and read video metadata
- **An OpenAI-compatible inference server** — [Ollama](https://ollama.com) is the recommended option; see [distributed inference](docs/distributed-inference.md) for remote setups

### Install

```bash
# Clone the repo
git clone https://github.com/GeoffreyCoulaud/subtitles-ocr
cd subtitles-ocr

# Install dependencies with uv
uv sync
```

### Inference server

The pipeline requires three VLM models. See [docs/inference-setup.md](docs/inference-setup.md) for model details and how to configure local Ollama, a remote machine, or a multi-machine LiteLLM proxy.

## Usage

```bash
uv run subtitles-ocr <video>
```

This produces `<video>.ass` next to the input file, and a `<video>_subtitles_ocr/` work directory containing intermediate files (frames, analysis JSONL, etc.).

### Options

| Option                   | Default                  | Description                                                                                     |
|--------------------------|--------------------------|-------------------------------------------------------------------------------------------------|
| `-o`, `--output`         | `<video>.ass`            | Path to the output `.ass` file                                                                  |
| `-w`, `--workdir`        | `<video>_subtitles_ocr/` | Directory for intermediate files                                                                |
| `-m`, `--model`          | `qwen2.5vl:3b`           | Ollama model for subtitle extraction                                                            |
| `--filter-model`         | `llava:7b`               | Ollama model for the pre-filter pass                                                            |
| `--filter-workers`       | `4`                      | Parallel workers for the pre-filter pass                                                        |
| `--analyze-workers`      | `1`                      | Parallel workers for the analysis pass (requires `OLLAMA_NUM_PARALLEL` ≥ value in Ollama's env) |
| `--edge-diff-threshold`  | `8.0`                    | Edge difference threshold for frame grouping                                                    |
| `--similarity-threshold` | `0.75`                   | Trigram similarity threshold for fuzzy event grouping                                           |
| `--gap-tolerance`        | `0.5`                    | Max gap in seconds to bridge between similar events                                             |
| `--reconcile-model`      | `gemma3:1b-it-qat`       | Ollama model for text reconciliation                                                            |
| `--reconcile-workers`    | `8`                      | Parallel workers for reconciliation                                                             |
| `--inference-url`        | `http://localhost:11434` | Base URL of the OpenAI-compatible inference server                                              |

### Example

```bash
# Basic usage
uv run subtitles-ocr episode01.mkv

# Custom output path and model
uv run subtitles-ocr episode01.mkv -o subs/episode01.ass -m llava:13b
```

## Documentation

- [Install, test, and run commands](docs/development.md)
- [Inference setup — local Ollama, remote machine, or LiteLLM proxy](docs/inference-setup.md)
- [Example frames showing supported subtitle types](docs/examples/README.md)