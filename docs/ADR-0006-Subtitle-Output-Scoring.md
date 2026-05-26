# ADR-0006: Subtitle Output Scoring

Branch: `feat/raw-diff-implementation`
Status: Designed, not yet implemented.
Origin: Pick-my-brain session between user and Claude, building on ADR-0001 → ADR-0005.
Supersedes: nothing.
Revises: nothing.

This ADR defines an objective, deterministic scoring engine that compares a pipeline-produced `.ass` to a known-good human-made `.ass` for the same source video, and produces a 0-1 score. The score is the weighted sum of nine independent 0-1 sub-scores. Each sub-score isolates a single dimension of correctness, so failures are diagnosable from the breakdown alone.

The scoring engine is delivered as a pure library (callable from code), a CLI entry point (for ad-hoc comparisons), and is designed to be wrappable in a pytest regression test later. It is **not** a pipeline stage — it lives beside the pipeline, not inside it.

---

## 1. Context

ADR-0001 through ADR-0005 design and refine the OCR pipeline. After ADR-0005 (the `normalize` stage refactor), the pipeline is feature-complete for the MVP: it produces an `.ass` from a hardsub video and a raw video. There is currently no objective way to measure whether a given output is close to a human-reconstructed reference; iteration relies on visual inspection.

A single ground-truth reference exists today: `/home/geoffrey/Projets/subtitles-ocr-kenichi/kenichi-S01E01-fansub-no-opening-no-ending-av1-50.ass`, produced by hand for the same source video the pipeline processes.

The dominant use case for scoring is **absolute quality reporting** ("this pipeline reaches X% of human quality on this sample") — interpretable, calibrated, comparable across pipeline revisions. A secondary use case is **A/B tuning** (comparing variants on the same input); if the hand-picked weights of v1 are later replaced by calibrated weights fitted from a labelled corpus, these two use cases collapse into one.

Out-of-scope motivations:
- **Regression detection in CI**: a pytest wrapper is *enabled* by this design but not produced as part of this ADR.
- **Quality-of-output metrics** (reading speed, characters-per-second, line-length sanity): the score measures correctness only. Improvements are downstream concerns ("done by hand, or by another pipeline").

---

## 2. Decision

A new module `src/subtitles_ocr/evaluation/` exposes a pure function:

```python
score(output_ass: Path, reference_ass: Path, weights: Weights, fps: Fraction) -> ScoreReport
```

returning nine sub-scores, a combined final score, and a debug-friendly matched-pair table.

A CLI entry point invokes this function and prints the report. Whether the entry point is a subcommand of `subtitles-ocr` or a sibling console script (`subtitles-ocr-evaluate`) is an implementation detail and not constrained by this ADR.

The scoring function takes weights as a parameter — they are never hard-coded into the algorithm. v1 ships with hand-picked defaults (see §6). Future calibration replaces these defaults; no other code change is required.

---

## 3. Sub-scores

Nine sub-scores, each in `[0, 1]`. Each is computed independently from the matched-pair table produced by the alignment step (§4).

| Sub-score | Signal |
|---|---|
| `text_plain` | Text similarity after heavy normalisation. Score-of-meaning. |
| `text_exact` | Text similarity preserving punctuation and case. Bonus credit for surface form. |
| `timing` | Closeness of cue start/end times, with frame-tied tolerance. |
| `recall` | Fraction of reference cues that were matched. |
| `precision` | Fraction of output cues that were matched. |
| `line_breaks` | Whether intra-cue line breaks (`\N`) match. |
| `styling` | Override tags governing text *appearance* — italics, bold, underline, colour, font size, rotation. Colour comparison is perceptual (CIE-LAB, ΔE76). |
| `position` | Where the text actually sits in the frame, resolved from `\an` + `\pos` (or `\move`) + estimated text bounding box. |
| `fade` | `\fad` parameters. Tied to ADR-0003 Stage 8 priority A. |

The sub-scores are independent by design. They are not normalised against each other; they each score a single property of the matched pairs (plus, for `recall` and `precision`, unmatched cues on each side).

---

## 4. Alignment

Output cues are aligned to reference cues by **maximum temporal overlap**. No text-based rescue pass is applied.

Rationale: the pipeline derives timestamps from observed video frames (the input is the same source video as the reference). Timing offsets larger than a frame are pipeline bugs, not tuning noise. Rescuing mistimed-but-correct pairs by text similarity would mask exactly the failure mode that should be surfaced as a hard error. A misaligned pair becomes a `recall` miss + a `precision` miss + an unmatched timing — three signals, all pointing at the same root cause.

### 4.1 Algorithm

For each reference cue `r`, compute the temporal IoU with every output cue. Pair `r` with the output cue of maximum IoU, provided IoU > 0. If multiple references contend for the same output cue, the reference with the highest IoU wins; the others remain unmatched.

Edge cases:
- IoU = 0 (no temporal overlap): unmatched.
- Empty reference: all sub-scores undefined → final score undefined (raised as an error, not silently 0).
- Empty output: `recall = 0`, `precision` undefined (vacuous: pipeline produced nothing). v1 reports `precision = 1.0` for the empty-output case and surfaces a warning in the report; this is a deliberate convention, not a measurement.

---

## 5. Per-sub-score definitions

### 5.1 `text_plain`

Per matched pair, compute a normalised character-level Levenshtein similarity on the two normalised strings:

```
score_pair = 1 - levenshtein(a, b) / max(len(a), len(b))
```

Normalisation pipeline (applied to both sides identically):

1. Strip override tag blocks (`{\...}`) entirely.
2. Replace `\N` and `\n` with a single space.
3. Apply Unicode NFKC (consistent with the existing `normalize` stage, ADR-0005).
4. Lowercase.
5. Normalise quotes: `‘`, `’`, `‚` → `'`; `“`, `”`, `„` → `"`.
6. Normalise ellipsis: `…` → `...`.
7. Strip ASCII punctuation (`.`, `,`, `;`, `:`, `!`, `?`, `…`, `"`, `'`).
8. Collapse runs of whitespace to a single space; trim.

Aggregation: arithmetic mean of `score_pair` over the matched pairs.

If there are zero matched pairs: `text_plain` is undefined, reported as `null`, and excluded from the weighted sum (with the weights rescaled — see §6.1).

### 5.2 `text_exact`

Same per-pair Levenshtein formula as `text_plain`, but with a lighter normalisation:

1. Strip override tag blocks.
2. Replace `\N` and `\n` with a single space.
3. Apply Unicode NFC (NOT NFKC — preserves compatibility-equivalent distinctions).
4. Collapse runs of whitespace to a single space; trim.

Case, punctuation, smart quotes, and ellipsis variants are kept. Aggregation: arithmetic mean.

### 5.3 `timing`

For each matched pair, compute the start-time delta and end-time delta in **frames** (using the source video framerate `fps`):

```
delta_start_frames = abs(out.start - ref.start) * fps
delta_end_frames   = abs(out.end   - ref.end  ) * fps
```

For each delta, compute an endpoint score with a frame-tied threshold:

```
endpoint_score(d_frames) =
    1.0                                 if d_frames < 1
    max(0.0, 1.0 - d_frames / K)        otherwise
```

Where `K` is a configurable tolerance horizon expressed in frames; v1 default `K = 10` (so a 10-frame delta scores 0). The two endpoint scores are averaged into the pair's `timing` score. Aggregation: arithmetic mean across matched pairs.

This formulation realises the rule "sub-frame is correct, one frame is not": at `d_frames < 1`, score = 1.0; at `d_frames = 1`, score = 1 - 1/K < 1.0 (a visible step at exactly one frame).

### 5.4 `recall`

```
recall = matched_pairs / total_reference_cues
```

If the reference is empty, `recall` is undefined → scoring raises.

### 5.5 `precision`

```
precision = matched_pairs / total_output_cues          (if total_output_cues > 0)
precision = 1.0                                        (if total_output_cues = 0; vacuous, warning emitted)
```

### 5.6 `line_breaks`

For each matched pair, compare the line-break structure after stripping override tags but **before** collapsing whitespace.

Let `n_ref`, `n_out` be the number of `\N` and `\n` separators in each. Per-pair score:

```
score_pair = 1.0 if n_ref == n_out and all break positions match (character index after light normalisation), else 0.0
```

"Light normalisation" here means NFC + trimming, nothing more. Aggregation: arithmetic mean across matched pairs.

A coarser "count-only" variant (1.0 if `n_ref == n_out`, else 0.0) is acceptable for v1 if position-matching proves tricky; document the choice in the implementation.

### 5.7 `styling`

For each matched pair, parse the override-tag state for each token of the text and compute component scores; combine them into a per-pair `styling` score.

Components considered:

| Component | Tags | Comparison |
|---|---|---|
| Italic | `\i` | Binary (1.0 if both italic or both non-italic, else 0.0) |
| Bold | `\b` | Binary |
| Underline | `\u` | Binary |
| Strikeout | `\s` | Binary |
| Primary colour | `\c`, `\1c` | Perceptual distance in CIE-LAB, mapped to 0-1 |
| Outline colour | `\3c` | Perceptual distance in CIE-LAB, mapped to 0-1 |
| Font size | `\fs` | `min(a, b) / max(a, b)` |
| Rotation Z | `\frz` | `max(0, 1 - abs(a - b) / 180)` |

Font name (`\fn`) is **not** scored — the pipeline does not extract it.

**Colour comparison detail.** Parse the `.ass` `&HBBGGRR` value → sRGB → linear RGB → CIE-XYZ (D65) → CIE-LAB. Compute **ΔE76** (the plain Euclidean distance in LAB):

```
dE = sqrt((L1 - L2)^2 + (a1 - a2)^2 + (b1 - b2)^2)
```

Map to 0-1:

```
colour_score = clamp(0, 1, 1 - max(0, dE - 1) / (dE_max - 1))
```

with `dE_max = 25` by default. Below ΔE ≤ 1 (imperceptible difference): score = 1.0.

ΔE76 is chosen over ΔE2000 because subtitle colours are coarse (white, yellow, orange, red, cyan) — the distinctions that matter are far above the threshold where ΔE76 and ΔE2000 disagree. The sRGB→LAB conversion is ~20 lines of arithmetic and stays in-project (no dependency added).

Aggregation within a pair: arithmetic mean of the component scores that *both* sides specify. Components left at default on both sides do not penalise; components specified differently (one side default, one side overridden) count as a 0.0 for that component. Aggregation across pairs: arithmetic mean.

### 5.8 `position`

What matters is **where the text actually sits in the frame**, not how it's encoded. Two cues using different anchor codes (`\an2` + `\pos(960, 1040)` vs `\an5` + a different `\pos`) can render at the same on-screen location. The comparison must resolve both sides to a common reference and compare those.

**Resolution to a common reference.** For each cue, compute the **estimated rendered text centre** in PlayRes coordinates:

1. Determine the anchor point `(ax, ay)`:
   - If `\pos(x, y)` is present: `(ax, ay) = (x, y)`.
   - Otherwise: derive from the Style's `MarginV`, `MarginL`, `MarginR` and the script's `PlayResX`, `PlayResY` (default ASS positioning rules).
2. Estimate the text bounding-box dimensions:
   - `text_height ≈ fs × line_count` where `fs` is the active font size (Style default or overridden by `\fs`), and `line_count` counts `\N`/`\n` separators.
   - `text_width ≈ fs × 0.55 × max_chars_per_line` (a rough average-glyph-width ratio; refined empirically if needed).
3. Translate the anchor point to the **text centre** using the `\an` code (which corner/edge `(ax, ay)` refers to). For example, `\an2` (bottom-centre): `centre = (ax, ay - text_height / 2)`. The `\an` code is consumed here, not compared.

**Per-pair score.** Euclidean distance between the two estimated centres, normalised by the frame diagonal `sqrt(PlayResX² + PlayResY²)`, mapped to 0-1 via:

```
position_score = clamp(0, 1, 1 - normalised_distance / d_max)
```

with `d_max = 0.10` (10 % of the frame diagonal is "clearly mispositioned").

**`\move`.** Compare the resolved centres at the move's start and end points (apply the same anchor/bounding-box resolution at each endpoint). Per-pair `\move` score is the average of the two endpoint position scores. If `\move` specifies optional `(t1, t2)` on either side, compare them in frames using the §5.3 endpoint-score formula; combine multiplicatively with the spatial score.

**Pairs to include.** A pair where neither side specifies `\pos` *and* neither specifies `\move` *and* both inherit the same default Style margins → the comparison degenerates to "both render at the default spot" and contributes 1.0 with low signal. To avoid drowning the metric in trivial 1.0s, only pairs where at least one side specifies `\pos` or `\move`, or where the resolved centres differ, are included in the mean. Aggregation: mean across qualifying pairs. If no pair qualifies: `position` is `null` and excluded (§6.1).

### 5.9 `fade`

For each matched pair, parse `\fad(t_in_ms, t_out_ms)`:

- Both sides have no fade: pair contributes nothing (excluded from the mean).
- One side has fade, the other does not: per-pair score = 0.0.
- Both sides have fade: compare `t_in` and `t_out` in **frames** using the same endpoint-score formula as §5.3. Per-pair score is the average of the two.

Extension to `\fade(...)` (priority B per ADR-0003) is straightforward: compare the alpha envelope at sample points. Out of scope for v1.

Aggregation: mean across pairs where at least one side specifies a fade. If no pair qualifies: `fade` is reported as `null` and excluded with rescaling.

---

## 6. Combination

Weights are **non-negative integers**, not floats. This makes hand-tuning trivial: bumping `timing` from `20` to `25` is a self-explanatory diff, and the relative effect on each other dimension is obvious by inspection. Normalisation by the running sum is the engine's job.

```
final = sum_i (w_i * s_i)  over all i where s_i is not null
       / sum_i (w_i)        over the same i
```

No clipping. The weighted sum is reported as-is, with the per-sub-score breakdown so degenerate combinations are visible in the report.

### 6.1 Null-rescaling

`position`, `fade`, and (in extreme cases) text/line-breaks may legitimately be `null` for a given pair of `.ass` files. When a sub-score is `null`, both its numerator contribution AND its denominator contribution drop out — the remaining weights renormalise. The report shows which sub-scores were dropped.

### 6.2 v1 hand-picked weights

```
text_plain   : 25
text_exact   :  5
timing       : 20
recall       : 15
precision    : 10
line_breaks  :  5
styling      :  5
position     :  5
fade         :  5
              ---
              95
```

Reasoning:
- `text_plain` is the dominant signal (right words).
- `timing` is bumped relative to a naïve "text-first" weighting: timing is grounded by the video itself (§4 rationale), so a timing miss is a *strong* signal of pipeline failure, not noise.
- `text_exact` is a surface-form bonus, not a primary axis.
- `recall > precision` — missing real subtitles is worse than the occasional hallucinated cue.
- `styling`, `position`, `fade` are equal at `5` each — they collectively monitor appearance + animation reconstruction (ADR-0003), but none individually dominates.

These numbers are defaults; the engine reads them from a `Weights` instance and never hard-codes any. Calibration will replace them.

---

## 7. Architecture

### 7.1 Module layout

```
src/subtitles_ocr/evaluation/
├── __init__.py              # re-exports score(), ScoreReport, Weights
├── alignment.py             # temporal-IoU pairing
├── text.py                  # text_plain, text_exact, normalisation pipelines
├── timing.py                # timing sub-score, endpoint_score()
├── line_breaks.py
├── styling.py               # styling sub-score, colour ΔE2000
├── position.py
├── fade.py
├── report.py                # ScoreReport, Weights (Pydantic v2 models)
├── score.py                 # orchestration: score(output, reference, weights, fps)
└── cli.py                   # console-script entry point
```

Pydantic v2 throughout, per project convention (see CLAUDE.md).

### 7.2 Data model

```python
class Weights(BaseModel):
    text_plain: int = Field(ge=0)
    text_exact: int = Field(ge=0)
    timing: int = Field(ge=0)
    recall: int = Field(ge=0)
    precision: int = Field(ge=0)
    line_breaks: int = Field(ge=0)
    styling: int = Field(ge=0)
    position: int = Field(ge=0)
    fade: int = Field(ge=0)

class MatchedPair(BaseModel):
    ref_index: int
    out_index: int
    iou: float
    text_plain: float
    text_exact: float
    timing: float
    line_breaks: float
    styling: float | None
    position: float | None
    fade: float | None

class ScoreReport(BaseModel):
    final: float
    sub_scores: dict[str, float | None]
    effective_weights: dict[str, float]   # after null-rescaling
    n_ref: int
    n_out: int
    n_matched: int
    matched_pairs: list[MatchedPair]
    warnings: list[str]
```

The report is JSON-serialisable (`model_dump_json()`) and is what the CLI prints.

### 7.3 Parsing

`.ass` parsing uses `pysubs2` (added via `uv add pysubs2`). Override tag parsing for `styling`, `position`, `fade` is done on the raw event text — `pysubs2` exposes it as `Event.text`. A lightweight regex/state-machine in `evaluation/_tags.py` extracts tag states; full pysubs2 tag rendering is not required.

### 7.4 CLI entry point

The CLI accepts:

```
subtitles-ocr-evaluate \
  --output path/to/output.ass \
  --reference path/to/reference.ass \
  --fps 24000/1001 \
  [--weights path/to/weights.json] \
  [--json]
```

Prints either a human-readable table or a `ScoreReport.model_dump_json()` payload (with `--json`). Exit code 0 on success, non-zero if either file is missing/unparseable.

### 7.5 Tests

`tests/evaluation/` contains:

- Per-module unit tests for each sub-score, using minimal hand-built `.ass` fixtures.
- An integration test that scores the Kenichi pipeline output against the Kenichi reference and asserts the resulting `ScoreReport` has the expected shape (no threshold assertion — that's calibration, not the engine).
- Tests for null-rescaling: weights renormalise correctly when sub-scores drop out.

No external services. No `print()`-only verification (per CLAUDE.md: "Vérification = pytest uniquement").

---

## 8. Synthetic corpus methodology

A method to produce additional reference/input pairs is documented here but not executed as part of this ADR. Production is deferred until calibration is needed.

### 8.1 Recipe

Inputs:
- A clean, high-quality source video with no embedded subtitles.
- An `.ass` track for that video (official sub, fansub, or hand-made).

Procedure:

1. **Hardsub** the `.ass` into the clean video using ffmpeg's `subtitles` filter:
   ```
   ffmpeg -i clean.mkv -vf "subtitles=track.ass" -c:v libsvtav1 -crf 50 -preset 6 -c:a copy hardsubbed.mkv
   ```
   The CRF and codec choice control "fansub degradation" — CRF 50 with AV1 simulates an aggressive low-bitrate encode. Adjust to taste; the goal is to produce realistic OCR pressure.

2. **Keep `track.ass` unchanged** as the reference.

3. **Run the pipeline** on `hardsubbed.mkv` (as `--hardsub`) plus `clean.mkv` (as `--raw`) to produce `output.ass`.

4. **Score** with `subtitles-ocr-evaluate --output output.ass --reference track.ass --fps <source fps>`.

### 8.2 Corpus design guidance

When eventually building a corpus:
- Mix dialogue-heavy and sign-heavy content (signs exercise `\pos` / `\move`).
- Include fade-rich tracks to stress `fade`.
- Mix encoders and quality settings to vary degradation realism.
- Aim for 10-20 samples for first-pass calibration; more if weights remain unstable.

---

## 9. Out of scope (declined)

- **Auto-calibration** (gradient descent on weights against a labelled corpus). The architecture supports it: weights are function parameters. The fitting routine, the labelled-rating UI, and the corpus are all future work.
- **Reading-speed / characters-per-second / line-length / "quality-of-output"** metrics. The score measures correctness, not quality. Improvements are downstream.
- **Multi-track `.ass`**. Assumes a single Dialogue style track on each side.
- **Cross-language evaluation**. Assumes output and reference are in the same language.
- **Animated colours** (`\t(\c...)`). Already out of scope per ADR-0003.
- **Karaoke tags** (`\k`, `\kf`, `\ko`). Already out of scope per ADR-0001.
- **CI integration** of a regression test. The pytest wrapper is enabled by this design but not produced.
- **Partial-fade scoring** (`\fade(...)` with non-trivial alpha envelopes). Priority B per ADR-0003; deferred.

---

## 10. Open questions

- **Default `K` for timing tolerance.** v1 uses `K = 10` frames. Once the engine is running on Kenichi, the typical distribution of timing deltas will inform whether this is too lenient or too strict.
- **`dE_max` for colour scoring.** v1 uses 25. Subject to revision once real pipeline-vs-reference colour distributions are observed.
- **Position metric: text-bounding-box estimate.** §5.8 estimates `text_width ≈ fs × 0.55 × max_chars_per_line`. The 0.55 ratio is a guess; once running on real data, the gap between the estimated centre and the true rendered centre (for known good pairs) will tell us whether to refine the ratio per language/script or whether the approximation is good enough.
- **Line-break position matching**. v1 may ship with the simpler "count-only" variant if position-matching proves fiddly; document the choice in the implementation commit.
