# ADR-0012: Edge-diff with per-scene spatial alignment

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: **Rejected** — implemented as designed, then patched ("Plan A"
in §8), then reverted. See §8 for the empirical outcome. Superseded by
ADR-0013 (text detector upstream + diff-as-validator).
Revises: (none — reverted before merge).

## 1. What problem are we solving?

We need a subtitle mask that actually isolates burnt-in subtitle
pixels — not the whole frame, not random patches of background.
Today's diff (`abs(Sobel(fansub) − Sobel(raw))`, both passed through
local contrast normalisation) fails because it confuses three things
that all produce non-zero gradient differences:

1. The subtitle pixels we want (target).
2. Encoder noise on flat regions.
3. Anatomy edges — character outlines, building lines, etc. — that
   re-quantise slightly differently between the fansub encoding and
   the BluRay encoding, even though they represent the same drawn
   strokes.

(1) and (3) look identical to a generic gradient detector: both are
sharp, both differ in magnitude between the two videos. ADR-0011
tried to suppress (2) with a bilateral filter and a global per-frame
normalisation. It worked for (2), but (3) was the dominant cause of
the saturated mask all along, and (3) cannot be removed by smoothing
or rescaling — the edges are physically there in both frames.

So we need a different *kind* of signal: one that's specific to
"edge added by the subtitle overlay" rather than "places where the
two gradient maps disagree".

## 2. What we have available

Two complementary observations:

- **A Canny edge map says present-or-absent, not by-how-much.** If
  an anatomy edge is in both fansub and raw, Canny marks both — and
  a logical AND-NOT between the two maps cancels it cleanly. The
  small magnitude differences that defeated the gradient-magnitude
  diff don't matter at all for a binary edge detector.

- **The two frames should be spatially aligned after Stage 1
  (conform).** Stage 1 downscales the raw to match the fansub's
  resolution, but doesn't enforce sub-pixel registration. In
  practice the residual offset is usually under a pixel, sometimes
  a couple of pixels, and sometimes a small scale or rotation
  difference (different rescalers in the two encoding pipelines).
  Any residual mis-registration turns every anatomy edge into a
  phantom "added" edge — so we need to absorb it.

## 3. The plan

### 3.1. Per-scene spatial registration

Before any per-frame work, compute a single affine transformation
that aligns the raw to the fansub *for the duration of one scene*.
We use Stage 2's existing `AlignmentSegment` boundaries as scene
breaks — they already mark continuous spans where the fansub-to-raw
frame index offset is constant, which is a strong proxy for "same
camera, same cut".

For each `ALIGNED` segment:

1. Pick a representative frame near the middle of the segment (less
   sensitive to fade-in / fade-out artefacts than the boundaries).
2. Read the matched fansub and raw frame, convert to BT.709 luma.
3. Compute the affine transform `T` that maximises image similarity
   between them. We use OpenCV's `cv2.findTransformECC` with
   `motionType=MOTION_AFFINE` (translation + rotation + scale +
   skew). This is a well-established image registration method
   that does iterative refinement on a normalised correlation
   objective.
4. **Safety bounds.** If the resulting transform shifts by more than
   3 % of frame width/height in translation, or rescales by more
   than 20 %, treat the registration as failed and fall back to
   identity. Same fallback if ECC fails to converge within its
   iteration budget. These bounds are deliberately loose — they
   exist to catch runaway optimisations on near-uniform frames
   (skies, fades to black) where ECC has no signal to lock onto.
5. **Subtitle-region exclusion mask.** During the optimisation, mask
   out the bottom 25 % of the frame (where dialogue typically sits)
   from the ECC similarity calculation. This prevents the optimiser
   from accidentally registering the raw onto the *fansub's
   subtitle* if the rest of the frame is featureless. OpenCV's
   `findTransformECC` supports this via its `inputMask` parameter.

The result is cached per-segment in the alignment sidecar. Typical
runtime: one ECC fit per scene, so a handful of seconds per episode
even at full resolution — negligible next to OCR.

> **ECC aside.** "Enhanced Correlation Coefficient" is one of
> several gradient-based image registration algorithms. It treats
> the registration as a numerical optimisation: pick a starting
> transform (identity here), warp one image with it, measure how
> well it matches the other under a normalised correlation score,
> nudge the transform parameters to improve the score, repeat. It
> handles translation up to several pixels and small scale/rotation
> changes well, and converges in a few tens of iterations on
> typical anime frames.

### 3.2. Per-frame edge diff

For each aligned fansub frame:

1. Apply the segment's transform `T` to warp the raw frame so it
   lines up with the fansub.
2. Convert both to BT.709 luma.
3. Run Canny on each, with conservative thresholds (the same
   thresholds for both frames so the comparison stays symmetric).
4. Dilate the raw edge map by 1 pixel. This is a cheap safety net
   against the residual sub-pixel mis-registration that survives
   ECC.
5. Compute `added_edges = fansub_edges AND NOT raw_edges_dilated`.
6. Morphological close on `added_edges` with a small kernel
   (3×3 to 5×5) so that adjacent stroke fragments of the same
   character merge into a connected blob.
7. Connected components, then filter by size: drop blobs whose area
   falls outside the existing `mask_area_min` / `mask_area_max`
   range.

The output is a binary mask in the same format the downstream
pipeline already consumes — no changes needed in compose, OCR,
animation, or anywhere else.

### 3.3. The full revised stage

```
fansub_frame, raw_frame, scene_transform T
    ↓ luma both
    ↓ warpAffine(raw, T)               (NEW — alignment refinement)
    ↓ Canny each                       (NEW — binary edge maps)
    ↓ dilate(raw_edges, 1px)           (NEW — sub-pixel buffer)
    ↓ fansub_edges AND NOT raw_dilated (NEW — text-specific signal)
    ↓ morpho close 3×3                 (NEW — merge strokes)
    ↓ connected components + size filter
mask
```

The LCN, Sobel, and abs-difference of the old Stage 3 are all gone.
The compose, mask hysteresis (`mask_t_high` / `mask_t_low`), and
downstream stages stay exactly as they are: the binary edge-diff
already produces a clean binary signal, so a continuous-to-binary
hysteresis threshold is no longer needed and `mask_t_high` /
`mask_t_low` become inert. They stay in the config for backwards
compatibility but are no longer read.

### 3.4. Configuration knobs

Add to `FrameProcessingConfig`:

- `canny_low_threshold` (int, default 50) and
  `canny_high_threshold` (int, default 150). Standard 1:3 ratio for
  uint8 luma; conservative enough to skip low-contrast noise edges
  but low enough to catch faint text strokes.
- `edge_diff_dilation_px` (int, default 1) — the safety buffer
  applied to the raw edge map before the AND-NOT.
- `edge_diff_close_kernel` (int, default 3) — kernel size for the
  morphological close that merges stroke fragments.

Add to `AlignmentConfig` (alignment is its scope):

- `ecc_max_translation_pct` (float, default 0.03) — 3 % of frame
  size, per axis.
- `ecc_max_scale_deviation` (float, default 0.20) — 20 % of unity.
- `ecc_subtitle_exclusion_pct` (float, default 0.25) — bottom
  fraction of the frame ignored during ECC fitting.
- `ecc_max_iterations` (int, default 50).

Retire from `FrameProcessingConfig`:

- `lcn_sigma`, `std_floor` (LCN is gone).
- The mask hysteresis knobs (`mask_t_high`, `mask_t_low`) stay in
  the schema but are documented as inert under ADR-0012.

## 4. How we'll know it worked

Three independent checks on the KenIchi full episode after the
implementation lands:

1. **Mask coverage drops on no-dialogue frames.** Take ten frames
   where the reference `.ass` has no event active. The 16×16 mask
   grid mean must be **below 0.05** on those frames.
2. **The final score does not regress below 0.8964** (the current
   measured baseline with the legacy LCN pipeline whose mask was
   effectively a no-op). Same scorer, same reference.
3. **At least 5 of the 8 reference `\fad` events are matched**, and
   the `fade` sub-score reaches **at least 0.30**. This is the
   downstream signal that would have failed in ADR-0011 even if
   that ADR's preprocessing had succeeded — here we expect it to
   succeed because the mask now actually fluctuates with subtitle
   presence.

If (1) passes but (2) regresses, OCR is missing real text — most
likely cause is Canny too aggressive or edge dilation eating
characters; loosen the relevant knob. If (3) fails despite (1) and
(2) passing, the ADR-0010 fade detector itself needs revisiting; not
in scope here.

## 5. Open questions

- **Canny threshold sensitivity.** 50 / 150 is the textbook default
  for uint8 luma. The acceptance check in §4(1) will tell us if it's
  too aggressive (text fragments dropped) or too loose (anatomy
  noise leaking through).
- **Per-frame vs per-scene ECC.** Per-scene is the default. If real
  episodes have intra-scene camera motion (slow pans, zoom-ins) that
  doesn't show up as a phash jump, we'd see drift inside long
  segments. Sub-segmenting on phash deltas is a cheap follow-up.
- **AV1 raw read cost.** Each ECC fit reads two frames out of the
  conformed raw. We rely on the existing frame reader; no new code
  path here. If the per-scene reads become a bottleneck, persist
  the per-segment transforms in the alignment sidecar (already the
  intended behaviour — flagged here for emphasis).

## 6. How we'll build it (TDD)

1. Add a unit test for ECC alignment: take a fansub frame, shift it
   by a known (dx, dy, small rotation), and check the recovered
   transform inverts to within 0.5 pixels.
2. Add a unit test for the safety-bounds fallback: feed two
   completely unrelated frames (or matching frames with a huge
   synthetic offset) and check the returned transform is identity.
3. Add a unit test for the edge diff: take a synthetic background
   image, add a white rectangle as the "subtitle" to one copy, and
   check the AND-NOT produces edges only along the rectangle.
4. Add a unit test for the subtitle-exclusion mask in ECC: build a
   pair of frames identical in the top 75 % but with a high-contrast
   patch added only in the bottom 25 % of one. The recovered
   transform should still be identity (the mask hides the patch
   from the similarity score).
5. Implement the per-segment ECC pass and persist results in the
   alignment sidecar. Wire it into `iter_composed_frames`.
6. Implement the new edge-diff path in `pipeline/frame_processing/`,
   replacing the LCN-Sobel call in `compute_diff`. Retire the LCN
   helpers and config fields.
7. Run the full KenIchi pipeline (delete `06_ocr/results.jsonl` and
   the sidecars to force a clean re-OCR) and check the three
   acceptance criteria from §4.
8. Update ADR-0010 §7.1 to reflect the new mask outcome. Mark this
   ADR Implemented.

## 7. Out of scope

- **Re-tuning `mask_t_high`, `mask_t_low`, `mask_area_min`,
  `mask_area_max`.** ADR-0011 already explored this dimension; the
  conclusion is that the existing thresholds are reasonable as long
  as the diff input is clean. The hysteresis pair becomes inert
  under the binary edge-diff signal anyway. The area thresholds
  still gate connected components and are kept as-is.
- **A learned text detector** (PaddleOCR's own DBNet, or CRAFT /
  EAST). This is the bigger architectural alternative. We're
  starting with edge-diff because it's classical, transparent, and
  doesn't introduce a second neural network. If edge-diff doesn't
  reach the §4 acceptance, a learned detector becomes the next ADR.
- **Sub-segmenting scenes by phash jump.** Mentioned in §5 as a
  follow-up. Default segmentation uses `AlignmentSegment`
  boundaries only.
- **Non-affine misregistration** (homography, optical flow). The
  fansub/raw pair is the same camera, same cut — a global affine
  should suffice. If a real episode breaks this assumption we'll
  see it in §4(1) and can revisit.

## 8. Outcome on KenIchi — why this ADR is Rejected

This is the second consecutive failed attempt at fixing the subtitle
mask via the existing diff-based architecture. Recording in detail
because the failures together motivate ADR-0013's structural pivot.

### 8.1. What we measured

We ran a 3-minute slice of KenIchi S01E01 end-to-end on three code
states, scored each against the same human reference `.ass`:

| Config | mask no-dlg | mask dlg | dlg/no-dlg | text_plain | recall | precision | final |
|---|---|---|---|---|---|---|---|
| LCN baseline (Stage 3 pre-ADR-0011) | 0.99 | 0.99 | 1.00 | 0.981 | 0.954 | 0.925 | **0.8958** |
| ADR-0012 v1 (Canny + ECC + dilation 1 + close 3) | 0.225 | 0.242 | 1.07 | 0.834 | 0.831 | 0.740 | 0.8040 |
| ADR-0012 Plan A (v1 minus ECC, +border 15 px, dilation 3) | 0.102 | 0.124 | 1.21 | 0.539 | 0.585 | 0.613 | 0.6399 |

Each iteration reduced no-dialogue mask coverage further (good in
isolation), but each also dropped OCR-relevant metrics further. No
operating point on this knob axis approaches the LCN baseline.

### 8.2. Why v1 didn't reach §4 acceptance

The §4 target was no-dialogue mask < 0.05. We landed at 0.225 —
4-5× too high. Three things contributed:

1. **ECC degraded alignment on KenIchi.** The fit reported a
   consistent ~(7, 7) px translation across every segment. Applying
   it made the diff *worse* than identity (0.276 → 0.332 on a
   typical no-dialogue frame). Root cause: the conformed raw on this
   episode carries a small black border (4-12 px on three sides)
   that the fansub doesn't have. ECC's correlation objective is
   dominated by the strong content/border boundary, so it pulls the
   raw a few pixels inward to "match" content — and then drags the
   border into the fansub's content area, creating phantom edges
   along the new border/content boundary.
2. **Anatomy edges drift sub-pixel between encoders.** Even with
   perfect frame alignment, the fansub (CRF50) and BluRay (CRF26)
   encodings produce edges at positions that differ by 1-3 px in
   many places. 1 px of dilation on `raw_edges` doesn't bridge
   that, so the AND-NOT leaves a dense field of phantom "added"
   edges across the whole frame — exactly the failure ADR-0012 was
   meant to prevent.
3. **The fragmented mask hurts OCR more than the saturated mask
   did.** Under LCN saturation the composed image was effectively
   `fansub × 1 = fansub` and PaddleOCR saw a clean frame. Under v1
   the composed image is a mosaic of original pixels and black
   patches; PaddleOCR finds real text in most regions but also
   imagines text shapes around the patch boundaries, producing
   high-confidence noise events. Hence text_plain 0.834 (vs LCN
   0.981) and a high one-word-event rate.

### 8.3. Why Plan A (less ECC, more dilation, border guard) made it worse

After v1's diagnosis we tried three patches simultaneously:

- `ecc_enabled = False` (ECC was demonstrably hurting),
- `edge_diff_border_px = 15` (zero the outer ring so the
  border/content boundary stops firing),
- `edge_diff_dilation_px = 3` (bigger buffer against anatomy drift).

The patches did what they were supposed to do at the mask level:
no-dialogue coverage dropped 0.225 → 0.102. But they cut **into the
text signal** at the same rate — dialogue coverage also dropped
0.242 → 0.124. The discriminator (dlg/no-dlg) barely improved
(1.07 → 1.21), and the absolute signal was now so weak that
PaddleOCR couldn't recover most of the subtitle text from the
composed image. text_plain collapsed to 0.54; 53 % of output events
became single words; recall 0.58; final 0.64.

In other words: the dilation knob trades subtitle text directly
against anatomy noise. There is no operating point that keeps both
acceptable.

### 8.4. The structural lesson

ADR-0011 and ADR-0012 share the same architectural premise: build the
mask from a generic pixel-domain comparison between fansub and raw
(gradient magnitude in ADR-0011, binary edges in ADR-0012). The
empirical evidence from both attempts converges on a single
conclusion:

> Pixel-domain diff signals cannot distinguish "subtitle stroke" from
> "anatomy edge that re-quantises slightly differently between
> encoders". Both are sharp luminance transitions, both differ
> between fansub and raw at comparable magnitudes. Any threshold or
> morphological operation tight enough to suppress one suppresses
> the other too.

The implicit assumption — that "subtitle pixels" is the dominant
contributor to the diff — does not hold on KenIchi (and probably on
most encoded anime). The dominant contributor is encoder-specific
edge drift.

Two architectural escapes:

- Use a **text-specific signal** (a trained text detector, e.g.
  PaddleOCR's DBNet, or classical SWT/MSER) to propose candidate
  regions, and use the fansub-vs-raw diff *downstream* only to
  validate each candidate as "burnt-in" vs "natively in-frame"
  (signs, book covers, in-scene text). This is ADR-0013.
- Drop the mask entirely from the OCR input and feed PaddleOCR the
  raw fansub frame (the implicit behaviour of the LCN baseline,
  whose mask saturated). Use the diff only for fade-detection
  signal (ADR-0010). This is the conservative fallback if ADR-0013
  doesn't reach acceptance either.

### 8.5. Status

Reverted on the branch before merge. The Canny edge-diff in
`diff.py`, the per-scene ECC in `registration.py`, the four new unit
tests, and the new config fields (`canny_*`, `edge_diff_*`,
`ecc_*`) are no longer in the codebase. `lcn_sigma` / `std_floor`
and the legacy Sobel-magnitude diff are restored as the active
implementation.

ADR-0013 picks up the architecture from §8.4 — text detector
upstream, diff as validator downstream.
