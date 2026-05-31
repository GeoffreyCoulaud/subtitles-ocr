# ADR-0010: Detecting fades from the subtitle mask

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: Designed — implementation pending.
Revises: ADR-0003 §4.2 sub-stage B (the current fade detector).

## 1. What problem are we solving?

Many fansubs fade their subtitles in and out (an episode title slowly
appearing, a character intro softening). In an ASS file this looks
like `\fad(500, 0)` — "fade in over 500 ms, no fade out".

Our pipeline already tries to detect these and emit a matching `\fad`
on every event. On the KenIchi test episode:

- The human-made reference has 8 events with a `\fad`.
- Our detector emits ~27 `\fad` candidates — but **none** of them line
  up with any of the 8 reference ones.
- The fade sub-score stays stuck at 0.

We need a detector that actually finds real fades.

## 2. Why does the current detector miss them?

It samples the *diff intensity* — how visually different each pixel
of the fansub looks from the clean raw video — and then tries to fit
a straight line through those samples to extract a "ramp slope".

Two things go wrong:

1. **The signal is the wrong thing to measure.** A subtitle being
   drawn at 30 % opacity is still visually different from a frame
   with no subtitle at all, so diff intensity reflects "how strongly
   coloured" the text is, not "how much of it is there". On top of
   that, the background changes between fansub and raw (re-encode
   noise, different encoders) also count as diff intensity. The fade
   signal is buried under that noise.

2. **The fit is too picky.** The detector requires the data points
   to look like a clean straight line (R² ≥ 0.30 — basically "no
   negative correlation"). On a noisy signal, even a real fade
   produces a fit so bad the gate rejects it. Hence the
   `R²=-9.808 < 0.30; reverting` warnings.

## 3. What we have available

Stage 4 of the pipeline already produces, for every frame, a **binary
mask** — a black-and-white image saying "this pixel is part of a
subtitle, that pixel isn't". This is much more directly a measure of
*how much subtitle is visible*. As a subtitle fades in, the mask goes
from "almost nothing" to "the full shape of the text". As it fades
out, the mask goes back to "almost nothing". The colour of the text
doesn't matter — only the area covered does.

Concretely, if we count the fraction of "subtitle" pixels inside an
event's bounding box on each frame, we get a curve that goes 0 → 1 →
1 → 0 over the event's lifetime. Fade-in and fade-out are the two
ramps at the ends.

## 4. The plan

### 4.1. Make the mask data available to the animation stage

Today the OCR stage produces a small companion file called
`diff_grid.npz` that holds the diff-intensity values per frame, so
the animation stage can read them without re-doing all the heavy
video work.

> **`.npz` aside.** This is NumPy's compressed archive format — a
> ZIP file containing one or more arrays of numbers. It's the
> standard way scientific Python code persists numeric data to disk.
> We use it because it's small, fast to read, and a one-liner to
> load with `numpy.load`.

We'll add a parallel file called `mask_grid.npz` that holds the
**mask coverage** per frame: for each frame, the image is shrunk to
a 16×16 grid where each cell holds the fraction of pixels in that
region that were "subtitle". 16×16 is coarse enough to keep the file
small (~32 MB for a 31 000-frame episode) but fine enough that an
event's bounding box covers several cells.

This file is written by the OCR stage as it streams through frames,
the same way `diff_grid.npz` is written today. The animation stage
notices the file exists and reads from it.

### 4.2. Change the detector to find threshold crossings

Instead of fitting a line and gating on R², the new detector looks
for **where the mask coverage drops to half its in-event value**.

For each event running from frame `start` to frame `end`:

1. Measure the "fully visible" reference: the median mask coverage
   inside the event's bounding box across frames `[start, end]`. Call
   this `reference`. If `reference` is very small (less than 30 %),
   the event's OCR was probably broken — skip fade detection for it.
2. Look at frames *before* the event, walking backwards from
   `start`. Find the first frame where the mask coverage drops below
   `reference / 2`. That's the half-fade point. The fade started
   roughly twice as far back as that point. So:
   ```
   fade_in_ms = 2 × (start − half_fade_frame) × ms_per_frame
   ```
3. Symmetrically for fade-out: walk forward from `end`, find the
   first frame below `reference / 2`, and compute `fade_out_ms`.
4. Apply the same sanity checks the current detector uses:
   - fades shorter than 125 ms are noise → set to 0;
   - fades longer than 1000 ms are usually catching the next subtitle
     → cap at the limit;
   - `fade_in_ms + fade_out_ms` must not exceed the event's duration
     → otherwise zero both.

That's it. No least-squares fit, no R² gate. The robustness comes
from using the *median* in-event coverage as the reference and from
walking inward to find the *closest* crossing.

### 4.3. Scoring stays the same

The scoring side already does the right thing: it compares the
output's `\fad(t_in, t_out)` against the reference's, scoring each
endpoint separately on how close the timings match. We're only
changing how the pipeline *produces* these numbers, not how they're
*compared*. No changes in `evaluation/`.

### 4.4. Switch the default back on

Today `emit_animation_fades` defaults to `False` because the old
detector only added noise. With the new one finding real fades, the
default flips back to `True`. The flag stays as an escape hatch.

## 5. Open questions

- **The half-fade threshold (0.5).** It's the obvious starting
  point — "halfway through the fade". If the real fade curve isn't
  linear (some fansubs use ease-in / ease-out), the halfway crossing
  shifts by a frame or two. One number to retune if the validation
  numbers miss.
- **The minimum in-event coverage cut-off (30 %).** Defensive
  switch for events where OCR mostly failed. Could be lowered if it
  rejects too many legitimate events.
- **Removing the old detector.** The new code path is strictly
  better on the corpus we test against. The old code becomes dead
  weight. We remove it outright in the same commit, rather than
  keeping a config flag to choose between the two.

## 6. How we'll build it (TDD)

1. Add the persistence helpers `MaskPresenceRecorder` and
   `PersistedMaskSource` (mirroring the existing diff-intensity
   versions). Unit test: record a hand-built mask, save it, reload
   it, check the coverage value matches what you'd compute by hand.
2. Hook the recorder into the OCR stream so `mask_grid.npz` gets
   written. Existing tests use fakes — no change.
3. Teach the animation stage to discover `mask_grid.npz` and use it,
   the same way it already auto-discovers `diff_grid.npz`.
4. Rewrite the fade detector to use the threshold-crossing approach.
   Unit tests: feed a fake mask source with a known fade curve and
   check the extracted `fade_in_ms` / `fade_out_ms` are within one
   frame of expectation.
5. Remove the obsolete config knobs (`fade_fit_r2_threshold`,
   `fade_score_fit_range`) and the now-dead diff-intensity fade
   path.
6. Flip `emit_animation_fades` to default `True`; restore the
   existing fade test to use the default config.
7. Re-export KenIchi and measure (criteria below).
8. Mark this ADR Implemented.

## 7. What "good enough" looks like

On the KenIchi episode, after this change:

- At least **5 of the 8** reference `\fad` events must be matched by
  a non-zero output `\fad`.
- The `fade` sub-score must rise from 0 to **at least 0.30**.
- The final score must rise by **at least +0.015** (the fade weight
  is 5, so a 0.30 sub-score lift is 5 × 0.30 / 100 ≈ +0.015).

If the first run misses, the half-fade threshold (§5) is the one
knob to try before considering deeper changes.

## 8. Out of scope

- Non-linear fade shapes (`\fade(...)` with custom alpha keyframes).
  ADR-0006 §5.9 already defers these.
- Letting the OCR stream expose a per-event time-series of mask
  coverage directly, instead of going through the 16×16 grid. Worth
  considering if the grid resolution proves too coarse, but the
  16×16 mirrors what the diff side does today, so we start there.
- Cross-script fade duration normalisation. Not needed: `\fad` is
  inline-only and the scorer already handles fps.
