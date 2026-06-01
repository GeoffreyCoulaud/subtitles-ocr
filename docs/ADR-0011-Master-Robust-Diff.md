# ADR-0011: Master-robust diff for the subtitle mask

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: **Rejected** — implemented, measured, reverted. See §8 for the
empirical outcome. Superseded by ADR-0012 (edge-diff with per-scene
spatial alignment).
Revises: (none — reverted before merge).

## 1. What problem are we solving?

The subtitle mask is fed by a per-pixel "is this different from the
clean raw video?" signal. Today that signal is so noisy that the mask
fires almost everywhere, even on frames with no subtitle. The
consequences:

- The mask is essentially saturated (~99 % coverage on KenIchi at the
  conformed 640×480 resolution).
- ADR-0010's mask-based fade detector can't find real fade ramps —
  there is no "no-subtitle" period in the signal to ramp away from.
- More broadly, any future work that needs a clean "presence of
  burnt-in text" signal hits the same wall.

The root cause is *not* in the mask logic. It's in the diff: the
fansub and the BluRay raw don't differ only where the subtitle is,
they differ slightly *everywhere* because the two videos come from
different masters with different encoding pipelines (CRF, codec,
chroma subsampling, sharpening, colour grading). The current diff
treats that everywhere-noise as signal.

## 2. Why the current diff still picks up the noise

Stage 3 today:

1. Convert each frame to grayscale (BT.709 luma).
2. Apply local contrast normalisation (LCN): subtract a
   Gaussian-smoothed local mean, divide by a Gaussian-smoothed local
   standard deviation.
3. Compute Sobel edge magnitude.
4. Take the absolute difference of the edge magnitudes between fansub
   and raw.

Two problems with this pipeline:

- **LCN is the wrong tool for master noise.** It's purely local: it
  compensates for slow spatial variation (vignette, gradient) but
  doesn't equalise the *global* tone shift between two masters, so
  master-wide brightness/contrast differences survive into the Sobel.
- **No denoising step.** High-frequency encoder noise — the kind two
  different codecs leave on flat regions — flows straight into Sobel,
  which dutifully turns it into spurious edges. The mask hysteresis
  then accepts those edges as subtitle pixels.

Worse, LCN actively makes things harder once we add proper
preprocessing: it divides by a local standard deviation, which on a
denoised flat region amplifies whatever residual noise remains, and
it produces overshoot halos around real edges that Sobel reads as
extra gradient. The right move is to replace LCN entirely.

## 3. The plan

Replace LCN with two preprocessing steps that actually address the
master-noise problem. Both steps are applied identically to the fansub
and the raw, so the comparison stays symmetric.

### 3.1. Per-frame colour normalisation

For each frame, compute the mean and standard deviation of the
grayscale image and rescale so that the result has mean 0 and std 1.
This is a per-frame z-score on the luma.

Why this works: two masters of the same scene have the same *shape*
of brightness distribution but their absolute brightness and contrast
can shift (e.g. one master is slightly darker, the other slightly
more contrasty). After per-frame z-scoring, both frames sit in the
same numeric range, and the comparison only sees their *relative*
structure.

Why per-frame and not per-clip: scenes change. A bright outdoor shot
and a dim night shot would pull a clip-wide mean into a meaningless
average. Per-frame normalisation tracks the actual content.

### 3.2. Edge-preserving denoising

Before computing the Sobel, smooth each frame with a **bilateral
filter**. The bilateral filter averages nearby pixels but only if they
have similar values — so flat regions get smoothed (encoder noise
removed) while sharp edges (text glyphs against background) stay
sharp.

This matters because Sobel is an edge detector. Any encoder noise
that survives normalisation turns into spurious "edges" in the output,
which the diff then treats as subtitle signal. A bilateral filter
collapses that noise without touching the real text edges.

A simpler alternative would be a Gaussian blur. We use bilateral
specifically because it preserves the text edges — a Gaussian blur
would soften them, weakening the signal we *do* want.

### 3.3. The full revised pipeline

```
fansub_frame, raw_frame
    ↓ BT.709 luma (unchanged)
    ↓ per-frame z-score normalisation (NEW)
    ↓ bilateral filter (NEW)
    ↓ Sobel edge magnitude (unchanged)
    ↓ abs(gmag_fansub - gmag_raw) (unchanged)
diff
```

LCN is removed. Z-score replaces LCN's job of equalising global
appearance (and does it better, because LCN was only local). Bilateral
handles the encoder noise LCN couldn't touch. Sobel and the
abs-difference stay exactly as they are.

If a future failure mode turns out to need *spatial* equalisation that
z-score can't provide (e.g. master gamma curves that diverge across
the frame), LCN can be reintroduced — but we wait for the evidence
before paying its cost.

### 3.4. Configuration knobs

Add two fields to `FrameProcessingConfig`:

- `bilateral_diameter` (int, default 5) — the neighbourhood radius for
  the bilateral filter, in pixels.
- `bilateral_sigma` (float, default 25.0) — both the colour and the
  spatial sigma. The same value works for both in practice.

The z-score normalisation has no knob (mean and std are computed from
the frame itself).

## 4. How we'll know it worked

Two independent checks, run on the KenIchi full episode:

1. **Mask coverage drops in non-subtitle regions.** Dump the mask grid
   for ten frames where the reference `.ass` has *no* dialogue active.
   The grid mean must drop from ~0.99 (current) to **below 0.30**.
2. **Fade detector finds real ramps.** Re-run the existing ADR-0010
   detector on the new diff/mask. At least **5 of the 8** reference
   `\fad` events must produce a non-zero output `\fad`, and the
   `fade` sub-score must rise from 0 to **at least 0.30**.

If (1) fails, the preprocessing isn't strong enough — tighten the
bilateral sigma. If (2) fails despite (1) passing, the problem is
upstream of preprocessing (e.g. alignment slip) and a separate
investigation.

## 5. Open questions

- **Bilateral sigma calibration.** 25.0 is a reasonable starting
  point for 8-bit luma. The acceptance check in §4(1) is the dial; if
  the mask still saturates, double the sigma; if real text gets
  blurred out (OCR text_plain drops more than 1 pt), halve it.
- **Computational cost.** Bilateral is slower than Gaussian. LCN's
  removal claws back some of that cost (two Gaussian passes + per-pixel
  arithmetic gone). Net effect on Stage 3-5 is probably a wash; the
  OCR pass downstream dwarfs both either way.

## 6. How we'll build it (TDD)

1. Add a unit test that two different-master frames of the *same
   underlying scene* (synthesised: same image with one channel
   gain-shifted and Gaussian noise added) produce a near-zero diff
   after preprocessing.
2. Add a unit test that a fansub frame with a synthetic white
   rectangle overlaid (the "subtitle") differs from its raw
   counterpart only at the rectangle's edges — even after
   preprocessing.
3. Implement `_per_frame_normalise` and `_bilateral_denoise` in
   `pipeline/frame_processing/diff.py`. Wire them into `compute_diff`
   between luma conversion and Sobel. Remove the LCN call and its
   helper. Drop any LCN-specific config fields from
   `FrameProcessingConfig`.
4. Add the two bilateral config knobs to `FrameProcessingConfig`.
5. Re-run the OCR stage on KenIchi (regenerates `diff_grid.npz` and
   `mask_grid.npz`), then run the §4 checks.
6. If both checks pass, mark this ADR Implemented and update
   ADR-0010's §7.1 to reflect the new outcome.

## 7. Out of scope

- Reintroducing LCN behind a feature flag. If a future failure mode
  proves LCN was load-bearing, we'll add it back deliberately — not
  preemptively as a safety net.
- Changing the mask thresholds (`mask_t_high`, `mask_t_low`,
  `mask_area_max`). The point of this ADR is to make the diff
  give the mask a cleaner input; the existing thresholds should
  then do the right thing.
- Histogram matching as an alternative to z-score. Histogram
  matching makes the two frames look identical *in shape*, which is
  stronger but trickier to implement correctly across scene cuts.
  z-score is the lower-effort first try; revisit only if it doesn't
  suffice.
- Frame alignment / motion compensation. We rely on Stage 2
  (alignment) having paired the right fansub frame with the right
  raw frame. If that's wrong, no diff preprocessing will help.

## 8. Outcome on KenIchi — why this ADR is Rejected

The plan in §3 was implemented and measured end-to-end. The result
forced a rethink of the entire approach. Recording it here so the
next attempt (ADR-0012) starts from the right premise.

### 8.1. What we measured

After implementing the bilateral + per-frame z-score preprocessing
(plus a robust median+IQR variant of the z-score to avoid bias on
high-contrast subtitle overlays), we ran the full pipeline on KenIchi
S01E01:

| Metric | Old LCN pipeline | This ADR | Acceptance target |
|---|---|---|---|
| Mask coverage on no-dialogue frames | ~0.99 | 0.71 | < 0.30 |
| Fade sub-score | 0.000 | 0.000 | ≥ 0.30 |
| Final score | 0.8964 | 0.5968 | not regressed |
| `text_plain` sub-score | 0.977 | 0.130 | n/a |
| Output events | 421 | 345 | n/a |
| Output events of length 1 word | (not measured) | 127 (37 %) | n/a |

Both acceptance criteria from §4 failed *and* we induced a 0.30-point
score regression. The preprocessing change had to be reverted.

### 8.2. Why mask coverage barely moved

The bilateral + z-score combination removed most of the encoder-noise
component of the diff (good — the synthetic master-noise unit test
went from mean(diff) ≈ 0.28 down to ≈ 0.07). But the *real* dominant
signal on KenIchi was never noise. It was **edges of the underlying
animation that survive both encodings** — character outlines, scene
boundaries, building lines. Sobel of either pipeline lights these up
identically, and they form one very large connected component that
sits below `mask_area_max = 500 000` at the conformed 640×480
resolution. The mask hysteresis then accepts the whole component.

In other words: no amount of denoising of the *flat* regions changes
the *edge* regions, and the edge regions are what saturate the mask.

### 8.3. Why the regression was worse than no change

The previous LCN pipeline saturated the mask at ~99 %. At that level,
the per-frame mask is essentially a uniform 1, so the composed image
fed to OCR is effectively `fansub × 1 = fansub`. PaddleOCR ran on the
original fansub frames and produced excellent text. The "mask" was
doing nothing — the pipeline was an OCR-the-whole-fansub-frame
pipeline in disguise.

This ADR's preprocessing dropped coverage to ~71 %, but not by
isolating subtitle pixels. It dropped coverage by *fragmenting* the
giant connected component: large patches of background now fell out,
while other large patches survived. The composed image fed to OCR
became a mosaic of original-pixels and black holes. PaddleOCR
treated the black-bordered patches as text candidates and emitted
spurious detections — small, high-confidence, noisy strings like
`"OLOPT"`, `"LSHE GAUS"`, `"jky"`, `"0GEB8BNE2500817"`. 127 of the
345 output events (37 %) were 1-word phantoms of this kind.

The mask change therefore *removed* the accidental safety net the
old pipeline had been relying on, without delivering the actual text
isolation the design assumed.

### 8.4. Sweeping `mask_area_max` (out-of-scope per §7, but checked)

To see whether the whole-frame-component theory held up, we ran a
quick sweep with `mask_area_max` lowered from 500 000 to 100 000
(below the 307 200 pixels of a 640×480 frame). On no-dialogue
frames, coverage collapsed from ~0.71 to 0.004 — confirming the
theory. But on dialogue frames, the *text* coverage also collapsed
to near zero, because the text edges are connected via Sobel to
neighbouring anatomy edges and form a single component. Lowering
`mask_area_max` rejects the giant component as a whole, taking the
text with it.

So adjusting the existing thresholds doesn't open a useful operating
point either.

### 8.5. The structural lesson

A diff that asks "where do the gradient magnitudes of fansub and raw
disagree?" cannot distinguish three things at once:

1. Burnt-in subtitle pixels (target),
2. Encoder noise on flat surfaces (suppressible by denoising),
3. Anatomy edges that re-quantise differently across encoders
   (not suppressible — the edges are there in both frames, just
   slightly different magnitudes).

This ADR addressed (2) and left (3) untouched. Even a hypothetical
perfect denoiser would not have helped, because the surviving signal
(3) is what dominates.

A useful subtitle mask needs a signal that is **specific to text**,
not a generic gradient. The two practical options are:

- a **text-edge-presence** comparison instead of a gradient-magnitude
  comparison (Canny on each frame, count which edge pixels are
  *added* in fansub) — robust because anatomy edges register as
  Canny-positive in *both* frames and cancel out;
- a **text detector** (PaddleOCR's own DBNet/EAST, or classical
  SWT/MSER) used upstream of the mask, with the fansub-vs-raw diff
  used only to validate "burnt-in vs natively in-frame".

ADR-0012 pursues the first option, with per-scene spatial alignment
(ECC) to absorb the ±1-px residual mis-registration that would
otherwise turn every anatomy edge into a phantom "added" edge.

### 8.6. Status

Reverted on the branch before merge. The bilateral + z-score code,
the two new unit tests (`test_compute_diff_robust_to_master_noise`,
`test_compute_diff_glyph_edge_survives_master_noise`), and the
`bilateral_diameter` / `bilateral_sigma` config knobs are no longer
in the codebase. `lcn_sigma` / `std_floor` remain — they are still
the active defaults until ADR-0012 lands. This document is kept as
the record of the attempt and the reasoning that motivates the
follow-up.
