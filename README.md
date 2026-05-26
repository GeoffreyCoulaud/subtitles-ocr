# subtitles-ocr

Extract clean `.ass` subtitle files from videos with hardcoded subtitles
("hardsubs") by diffing them against a matching subtitle-free raw video.

The tool targets fansubbed anime where no official subs exist but a clean
(typically Blu-ray) raw of the same episode is available. Dialogue, lyrics,
and forced in-frame translations all count as subtitles. Original positions
are preserved — this is a pure extraction tool, not a re-typesetter.

## How it works

The pipeline runs 9 sequential stages (ADR-0002 §2, revised by ADR-0003 §3).
Stages 3–5 (diff / mask / compose) are streamed inside Stage 6 and do not
appear as separate orchestrated stages.

| #  | Stage           | Description                                                                                                  |
|----|-----------------|--------------------------------------------------------------------------------------------------------------|
| 1  | Conform         | ffmpeg downscales the raw to the fansub's resolution; output cached as lossless FFV1 MKV                     |
| 2  | Alignment       | Hybrid audio (silero-VAD + hierarchical cross-correlation) + phash refinement; phash-only fallback           |
| 3  | OCR             | PaddleOCR PP-OCRv5 server runs on composed (diff×mask) frames; streaming `iter_composed_frames` upstream     |
| 4  | Group           | Per-quad trajectory tracking groups detections into events; identical text + IoU > 0.5 continues a trajectory |
| 5  | Animation       | MVP: passthrough — emits a static `AnimatedEvent` per group event. Reserved for `\move` / `\fad` (Phase 6)   |
| 6  | Color           | Per-event color extraction: quad-rectified temporal median → Otsu → distance-transform → HSV mode clustering  |
| 7  | Event cleanup   | Per-event LLM call reconciles OCR variants and fixes confusables; skipped if all variants strictly identical  |
| 8  | Normalize       | Deterministic text normalization: Unicode NFKC, invisible-char stripping, whitespace collapse, OCR-noise pruning (ADR-0005) |
| 9  | Export          | pysubs2 writes `.ass`; styles synthesised by position × color cluster (ΔE76 in LAB)                          |

Stages 3-5 of ADR-0002 (diff, mask, compose) live as a streaming library
(`pipeline/frame_processing/`) owned by the OCR stage rather than as
independent CLI-visible stages.

## Workdir layout

Each run writes a workdir of numbered sub-directories (ADR-0003 §5):

```
workdir/
  01_conform/        raw.mkv, raw.meta.json
  02_alignment/      hardsub_audio.wav, raw_audio.wav, *.meta.json, alignment.json
  03_diff/           (empty in prod, debug/ under --debug-images)
  04_mask/           frames/<idx>.png
  05_compose/        (empty in prod, frames/<idx>.png under --debug-images)
  06_ocr/            results.jsonl, results.meta.json
  07_group/          events.json, events.meta.json
  08_animation/      animation.json, animation.meta.json
  09_color/          colors.json, colors.meta.json
  10_event_cleanup/  cleaned.jsonl, cleaned.meta.json
  11_normalize/      normalized.json, normalized.meta.json
  pipeline.log
```

The final `.ass` is written to the path given by `--out`, not to the workdir.

## Resume strategy

Resume is implicit — there is no `--resume` flag.

- **Fast stages** (1, 2, 4, 5, 6, 8, 9) write an atomic JSON or final artefact
  at end-of-stage plus a `*.meta.json` sidecar recording the inputs and
  config that affect the output. On re-run, the sidecar is compared against
  the current inputs+config; if they match the stage is skipped, otherwise
  it re-runs.
- **Slow stages** (3 OCR, 7 event cleanup) append per-chunk to a JSONL
  (`JsonlWriter`). After a crash, the next run re-reads the JSONL and
  restarts after the last persisted entry.
- Cache-affecting fields are listed in each stage's `cache_invalidating_dict`
  (everything except runtime knobs like parallelism — see ADR-0004 §4).
- To force a stage to re-run, delete its sub-directory or the whole workdir.

## CLI

```
subtitles-ocr \
  --hardsub <fansub.avi> \
  --raw <bluray.mkv> \
  --out <output.ass> \
  --workdir <intermediates/> \
  [--language latin] \
  [--debug-images] \
  [--ar-strategy error|letterbox|crop] \
  [--hardsub-audio-track <idx>] \
  [--raw-audio-track <idx>] \
  [--hardsub-skip "HH:MM:SS-HH:MM:SS"] (repeatable) \
  [--raw-skip "HH:MM:SS-HH:MM:SS"] (repeatable) \
  [--ocr-device auto|cuda|rocm|cpu] \
  [--event-cleanup-model <ollama-name>] \
  [--event-cleanup-parallelism <int>] \
  [--color-cluster-threshold <float>] \
  [--debug]
```

### Flags

| Flag                            | Default     | Description                                                                          |
|---------------------------------|-------------|--------------------------------------------------------------------------------------|
| `--hardsub`                     | (required)  | Path to the hardsubbed video (fansub)                                                |
| `--raw`                         | (required)  | Path to the clean raw video (Blu-ray / WEB-DL)                                       |
| `--out`                         | (required)  | Output `.ass` file path                                                              |
| `--workdir`                     | (required)  | Directory for intermediate artefacts                                                 |
| `--language`                    | `latin`     | PaddleOCR language code                                                              |
| `--debug-images`                | off         | Persist diff/compose debug PNGs (large; for diagnostics only)                        |
| `--ar-strategy`                 | `error`     | Aspect-ratio mismatch policy (only `error` is implemented in MVP)                    |
| `--hardsub-audio-track`         | none        | Audio track index in the hardsub for audio alignment (Stage 2a)                       |
| `--raw-audio-track`             | none        | Audio track index in the raw for audio alignment (Stage 2a)                          |
| `--hardsub-skip`                | none        | Time range (HH:MM:SS-HH:MM:SS) in the hardsub to exclude from alignment (repeatable) |
| `--raw-skip`                    | none        | Time range to exclude from the raw side of the alignment (repeatable)                |
| `--ocr-device`                  | `auto`      | `auto` warns and falls back to CPU on GPU failure; explicit values hard-fail         |
| `--event-cleanup-model`         | none        | Ollama model name used by Stage 7                                                    |
| `--event-cleanup-parallelism`   | `1`         | ThreadPoolExecutor size for Stage 7                                                  |
| `--color-cluster-threshold`     | `10.0`      | ΔE76 distance threshold for grouping events into shared `.ass` styles                |
| `--debug`                       | off         | Lowers stdout log level to DEBUG                                                     |

## Setup

### Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — manages Python and dependencies
- **[ffmpeg](https://ffmpeg.org/download.html)** — `ffmpeg` + `ffprobe` on `$PATH`
- **An OpenAI-compatible LLM server** — [Ollama](https://ollama.com) is
  recommended for Stage 7 (event cleanup); see
  [docs/inference-setup.md](docs/inference-setup.md) for remote and
  multi-machine setups

### Install

```bash
git clone https://github.com/GeoffreyCoulaud/subtitles-ocr
cd subtitles-ocr
uv sync
```

### Run

```bash
uv run subtitles-ocr \
  --hardsub /path/to/fansub.mkv \
  --raw /path/to/bluray.mkv \
  --out /path/to/episode01.ass \
  --workdir /path/to/work
```

## Evaluating output quality

Once the pipeline has produced an `.ass`, you can score it against a known-good reference `.ass` (e.g. a human-made fansub for the same source video):

```bash
uv run subtitles-ocr-evaluate \
  --output workdir/12_export/output.ass \
  --reference path/to/reference.ass \
  --fps 24000/1001
```

The score is a 0-1 weighted sum of nine independent sub-scores: `text_plain`, `text_exact`, `timing`, `recall`, `precision`, `line_breaks`, `styling`, `position`, `fade`. Each is reported alongside its effective weight. See [ADR-0006](docs/ADR-0006-Subtitle-Output-Scoring.md) for the full specification.

Pass `--json` for a machine-readable `ScoreReport`. Pass `--weights weights.json` to override the default integer weights (a JSON object matching the `Weights` Pydantic model).

## Documentation

- [Install, test, and run commands](docs/development.md)
- [Inference setup](docs/inference-setup.md)
- [ADR-0001 — initial scoping](docs/ADR-0001-OCR-Pipeline.md)
- [ADR-0002 — detailed pipeline design](docs/ADR-0002-Pipeline-Detailed-Design.md)
- [ADR-0003 — animation reconstruction scope](docs/ADR-0003-Animation-Reconstruction.md)
- [ADR-0004 — shared infrastructure](docs/ADR-0004-Shared-Infrastructure.md)
- [ADR-0005 — normalize stage refactor (supersedes ADR-0002 §3 Stage 10)](docs/ADR-0005-Normalize-Stage-Refactor.md)
- [ADR-0006 — subtitle output scoring](docs/ADR-0006-Subtitle-Output-Scoring.md)
