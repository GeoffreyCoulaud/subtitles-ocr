# ADR-0013: Text detector upstream, diff-as-validator downstream

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: **Rejected** — implemented end-to-end on a 3-minute slice, then
reverted. See §8 for the empirical outcome.
Revises: (none — reverted before merge).

## 1. What problem are we solving?

After two consecutive failed attempts (ADR-0011 noise-robust
preprocessing, ADR-0012 binary edge-diff + per-scene alignment), the
empirical conclusion is the same in both cases: a pixel-domain
comparison between fansub and raw cannot reliably tell a subtitle
stroke apart from an anatomy edge that re-quantises slightly
differently between encoders. Both signals look identical at the
gradient level. Any threshold or morphological operation tight
enough to suppress one suppresses the other.

We need a different approach. The job of the mask was always to
isolate burnt-in subtitle pixels so that OCR sees only them. The
diff was a proxy for "is this pixel part of a burnt-in subtitle?",
and that proxy is too noisy. We can stop using the diff as a
detector and use it only as a verifier.

## 2. What changes architecturally

Today's flow (since ADR-0002):

```
fansub frame ──┐
               ├─ diff ─ mask ─ compose ─ PaddleOCR(composed) ─ detections
raw frame ─────┘
```

Proposed flow:

```
fansub frame ─ PaddleOCR(fansub) ─ candidate detections ──┐
                                                          ├─ keep where burnt-in
raw frame, fansub frame ──── per-bbox diff score ─────────┘
```

PaddleOCR's built-in text detector (DBNet, trained specifically to
find text in natural images) replaces the diff/mask as the "where is
the text" signal. We feed it the unmodified fansub frame, so it has
every cue the original image provides — including small, faint, or
faded text the old mask would have absorbed into the saturated
whole-frame component.

The fansub-vs-raw diff still has a job, but a much simpler one: for
each candidate detection PaddleOCR proposes, look inside the
detection's bbox. If the fansub differs strongly from the raw there,
the text is burnt-in (the subtitle isn't in the raw). If they're
similar, the text is natively in-frame (a sign, a book cover, an
on-screen UI element) — we drop it.

This is a clean separation of concerns: a learned text detector
finds *where* text is, a cheap diff metric tells us *whether the
text was added by the fansubber*. Neither side has to do the other's
job.

## 3. The plan

### 3.1. Feed PaddleOCR the unmodified fansub frame

Stop running `compose(fansub, mask)` ahead of OCR. The OCR stage
calls `ocr_engine.detect(fansub_image)` directly. The
diff/mask/compose pipeline is no longer in the OCR critical path.

This single change recovers the LCN-baseline behaviour for OCR text
quality — PaddleOCR sees a clean natural image, which is exactly
what it was trained on.

> **PaddleOCR aside.** PaddleOCR runs a two-stage internal pipeline
> per call: a *detector* (DBNet by default in PP-OCRv5) finds
> text-likely regions and emits a quad per region, then a
> *recogniser* reads each quad's pixels. The current
> `PaddleOcrEngine.detect()` wrapper returns both — text and quad
> per detection. We don't need to split the two stages; we use the
> existing API and post-filter the returned list.

### 3.2. Per-detection burnt-in validator

For each `OcrDetection` returned by PaddleOCR, compute a single
scalar "burnt-in score" using the fansub and the aligned raw frame:

1. Compute the axis-aligned bbox of the detection's quad.
2. Pad the bbox by ~4 px on each side so we capture the immediate
   surroundings (helps the metric stay stable on tight crops).
3. Crop both the fansub and the raw to that padded bbox, convert to
   BT.709 luma.
4. Compute the **mean absolute pixel intensity difference** between
   the two crops, normalised to 0-1 (divide by 255).
5. Compare against `burnt_in_diff_threshold` (default 0.10 — to be
   calibrated; see §4).

Pixels in a burnt-in subtitle bbox are bright/dark strokes that the
raw frame does not have, so the mean absolute difference is large.
Pixels in a sign or book-cover bbox are the same in both frames, so
the mean absolute difference is small (dominated only by encoder
noise). The metric is cheap (one luma conversion + one
subtraction + one mean per detection) and doesn't depend on any
spatial alignment beyond what Stage 2 already provides.

The metric stays robust to encoder noise without any
preprocessing — averaging over a few hundred pixels in the bbox
washes the random per-pixel encoder noise out to a stable bias well
below the 0.10 threshold.

### 3.3. Where the validation runs

Inside the OCR stage's per-frame loop, immediately after the
`ocr_engine.detect(...)` call. We already have both the fansub
frame (we just OCR'd it) and the aligned raw frame (we just read
it). Filter the detections list, then persist the kept ones via the
existing `FrameOcrResult` schema. No new sidecar, no new stage, no
new persistence concern.

The `iter_composed_frames` driver no longer does `compose`. We
rename it `iter_aligned_frames` and yield
`(fansub_frame_idx, fansub_image, raw_image)` triples instead.

### 3.4. Remove mask, compose, and the LCN-Sobel diff

Nothing downstream of OCR needs the mask anymore, and the validator
in §3.2 already establishes mean absolute luma difference as the
canonical "how strongly does fansub differ from raw here?" metric.
We collapse the codebase onto that one metric:

- Delete `pipeline/frame_processing/mask.py` and `compose.py`.
- Delete `MaskPresenceRecorder` / `PersistedMaskSource` in
  `mask_presence.py`. The `06_ocr/mask_grid.npz` sidecar disappears
  with them.
- **Simplify `compute_diff` to plain mean absolute luma diff.**
  The function becomes a one-liner: `abs(luma(fansub) - luma(raw))`
  as float32 in `[0, 255]`. The LCN normaliser, the Sobel
  gradient, and `_local_contrast_normalize` / `_sobel_magnitude`
  helpers all go.
- Drop the `mask_sink` argument from `iter_aligned_frames`. Drop
  every `mask_*` field and `lcn_sigma` / `std_floor` from
  `FrameProcessingConfig`. The class now has no fields — remove
  it entirely from `config.py` and `OcrConfig`; the iterator no
  longer needs a config parameter at all.
- Delete the corresponding tests.

`iter_aligned_frames` still calls the (now trivial) `compute_diff`
and pipes its result through `diff_sink` so the validator and the
fade detector see a consistent signal. The `--debug-images` PNG
dump in `03_diff/debug/` keeps working — the diff is now an 8-bit
absolute-difference image, more legible than the LCN-magnitude one.

The metric is unified: a single mean-absolute-luma-diff signal
appears at two granularities — per-detection bbox (the validator,
§3.2) and per-frame 16×16 grid (the fade detector, §3.5).

### 3.5. Pivot fade detection to diff-intensity

ADR-0010 currently reads `mask_grid.npz` and looks at per-event
bbox-averaged mask alpha as the "how visible is this subtitle"
signal. With the mask gone we use `diff_grid.npz` instead — same
16×16 grid layout, but each cell now holds the mean absolute luma
difference between fansub and raw (per §3.4) rather than the
fraction of pixels that survived a hysteresis threshold.

The threshold-crossing algorithm transposes one-to-one. Per event:

1. **Reference intensity.** Take the median of the per-frame
   bbox-averaged diff intensity across frames `[start, end]` of the
   event. Call this `reference`.
2. **Skip very-low-signal events.** If `reference` falls below a
   floor `min_in_event_intensity` (analogue of
   `min_in_event_alpha`), the event probably lost OCR or doesn't
   have a real subtitle — emit no fade.
3. **Walk inside-out.** From `start`, walk backwards; find the
   first frame whose bbox intensity drops below
   `reference × fade_intensity_threshold` (default 0.5). The fade
   half-point is roughly there; fade-in duration is
   `2 × (start − half_frame) × ms_per_frame`. Symmetric for
   fade-out from `end`.
4. **Same sanity gates** as today: min 125 ms, cap 1000 ms, total
   ≤ event duration.

Renames in `AnimationConfig`:

- `min_in_event_alpha` → `min_in_event_intensity`. The new scale is
  the same as the validator's threshold scale — luma diff in
  `[0, 255]`. A reasonable starting value tracks
  `burnt_in_diff_threshold × 255 ≈ 25` (a fade frame is "below
  half of a typical burnt-in event"); tune in §4 if the
  acceptance check fails.
- `fade_alpha_threshold` → `fade_intensity_threshold` (default
  0.5 unchanged — the halfway-crossing constant is signal-source
  agnostic).

The `MaskAlphaSource` Protocol becomes `DiffIntensitySource` with
the obvious one-method change. `PersistedDiffSource` was already
introduced for the v1 diff sidecar; it just needs to expose a
`bbox_intensity(frame_idx, bbox)` method matching the new Protocol.

### 3.6. Configuration

Add to `OcrConfig`:

- `burnt_in_diff_threshold` (float, default 0.10) — the threshold
  on mean absolute luma difference, normalised to `[0, 1]`.
- `burnt_in_bbox_padding_px` (int, default 4) — how many pixels to
  pad each side of the detection bbox before sampling.
- `enable_burnt_in_filter` (bool, default True) — escape hatch in
  case a future video has a fundamentally different validator
  characteristic.

`FrameProcessingConfig` is removed entirely (per §3.4): there are
no per-frame knobs left once the mask, the LCN normalisation, and
the Sobel gradient are gone. `OcrConfig` loses its
`frame_processing` field; `iter_aligned_frames` takes no config
parameter.

## 4. How we'll know it worked

Empirical acceptance on the KenIchi 3-minute slice (cheapest
iteration we have) and then on the full episode:

1. **Final score must reach LCN baseline (0.8958) on the slice.**
   That's the floor — beating LCN means we kept everything that
   worked while gaining the ability to reject natively in-frame
   text.
2. **Precision improves over LCN.** LCN's precision on the slice
   was 0.925. The whole point of the validator is to drop false
   positives (in-frame text, encoder-noise OCR hallucinations).
   Target: ≥ 0.94 on the slice.
3. **No catastrophic recall regression.** LCN's recall on the slice
   was 0.954. With a 0.10 threshold the validator should almost
   never reject a real burnt-in subtitle. Target: ≥ 0.94.
4. **Fade sub-score reaches at least 0.30.** With the signal pivot
   in §3.5 the fade detector finally has a usable continuous
   curve (the saturated mask never gave it one). At least 5 of
   the 8 reference `\fad` events must produce a non-zero output
   fade. This is the same bar ADR-0010 §7 set; we're meeting it
   from the other side now.

If (1) fails by a small margin (~0.01-0.03), it's almost certainly
the threshold. Sweep `burnt_in_diff_threshold` over {0.05, 0.08,
0.12, 0.15} on the slice and pick the best. If (1) fails by a wide
margin (~0.05+) or (2) regresses, the validator is misfiring on a
specific class of detection — diagnose with per-detection diff
histograms before tuning. If (4) fails while (1)-(3) pass, the
issue is in the diff-intensity fade detector's calibration
(`min_in_event_intensity`, `fade_intensity_threshold`), not the
text validator — tune in isolation.

## 5. Open questions

- **Detector recall on faded subtitles.** PaddleOCR's DBNet is
  trained on opaque text; we don't know how it handles \fad-style
  alpha ramps. If the detector misses early/late fade frames, we
  lose some events. Acceptance check (3) is the canary.
- **Threshold portability across episodes / fansubs.** 0.10 is a
  starting point calibrated on KenIchi. Different fansubs use
  different color schemes (white-on-black vs yellow-on-shadow vs
  hard-outlined). The validator metric works on intensity diff
  which is colour-agnostic, but the magnitude varies. Episodes
  with low-contrast subtitles may need a lower threshold; a future
  ADR could auto-calibrate per-episode if needed.
- **Per-detection diff cost.** ~30 OCR detections per frame, each
  bbox ~50×30 px → ~1500 pixels per detection × 30 = 45 k pixels
  per frame for the diff metric. Negligible next to PaddleOCR's
  inference cost.

## 6. How we'll build it (TDD)

Five blocks of work, each preceded by its failing test(s):

**A. Burnt-in validator (new module).**

1. Unit test: identical (fansub, raw) crops → score below
   threshold (in-frame text case).
2. Unit test: same pair but with bright text added to the fansub
   only → score above threshold (burnt-in case).
3. Unit test: identical pair plus Gaussian noise σ=4 on one side
   → score stays below threshold (robustness check).
4. Implement `pipeline/frame_processing/validator.py` exposing
   `is_burnt_in(quad, fansub, raw, threshold, padding) -> bool`.

**B. OCR stage rewiring.**

5. Unit test the post-filter: mock an OCR engine returning three
   detections — one over a matching region, two over differing
   regions; assert only the two are kept.
6. Rewrite `iter_composed_frames` → `iter_aligned_frames`: yield
   `(idx, fansub, raw)` triples, drop `compose` and `make_mask`
   calls, keep `compute_diff` + `diff_sink`.
7. Modify the OCR stage to call `ocr_engine.detect(fansub)` and
   filter detections through the validator before persisting.

**C. Mask removal + diff simplification.**

8. Delete `pipeline/frame_processing/mask.py`,
   `pipeline/frame_processing/compose.py`,
   `pipeline/frame_processing/mask_presence.py`. Delete their
   tests.
9. Simplify `compute_diff` to `abs(luma(fansub) - luma(raw))`
   (float32, range `[0, 255]`). Delete `_local_contrast_normalize`,
   `_sobel_magnitude`, and their unit tests.
10. Remove `FrameProcessingConfig` from `config.py` and the
    `frame_processing` field from `OcrConfig`. Drop the config
    argument from `iter_aligned_frames` and `compute_diff`. The
    OCR stage stops importing the field.

**D. Fade signal pivot.**

11. Unit test: feed the fade detector a synthetic diff-intensity
    time series with a known ramp; the recovered fade duration is
    within one frame of expectation.
12. Rename `mask_alpha` → `bbox_intensity` everywhere; rename the
    Protocol and `PersistedMaskSource` →
    `PersistedDiffIntensitySource` (or reuse the existing diff
    intensity helper if its API already matches). Rename the
    `AnimationConfig` fields per §3.5.

**E. Validation.**

13. Run the 3-minute slice end-to-end; verify §4(1)-(4).
14. Run the full episode; confirm.
15. Update ADR-0002 §3 stage table to reflect the new flow.
    Update ADR-0010 §7.1 to mark follow-up #1 done and link here.
    Mark this ADR Implemented.

## 7. Out of scope

- **Splitting PaddleOCR's detector and recogniser into two stages.**
  The wrapper already returns text+quad per detection; we use them
  both. If the recognizer's text on a burnt-in detection is
  garbage, that's PaddleOCR's problem, not ours.
- **Replacing PaddleOCR with a different OCR engine.** Out of
  scope for this branch.
- **Auto-calibrating `burnt_in_diff_threshold` per-episode.**
  Possible follow-up if §5's portability concern materialises on
  a second test episode.
- **Improving fade detection algorithm.** §3.5 is a signal-source
  pivot only — the threshold-crossing algorithm itself stays as
  ADR-0010 §4 specifies. Algorithmic changes (non-linear fade
  shapes, alpha keyframes) remain ADR-0010 §8 out-of-scope.

## 8. Outcome on KenIchi — why this ADR is Rejected

This is the third consecutive attempt at improving the subtitle
mask/diff pipeline. ADR-0011 (preprocessing) and ADR-0012 (edge-diff
+ ECC) regressed and were reverted; ADR-0013 lands at parity with the
LCN baseline but doesn't beat it. Documenting the empirical result so
the next iteration starts from the right place.

### 8.1. What we measured

The 3-minute KenIchi slice was run end-to-end on each code state.

| Code | mask cov no-dlg | text_plain | recall | precision | 1-word events | fade | final |
|---|---|---|---|---|---|---|---|
| LCN baseline (Stage 3 pre-ADR-0011) | 0.99 | 0.981 | 0.954 | 0.925 | — | 0.0 | **0.8958** |
| ADR-0011 (bilateral + z-score) | 0.71 | 0.130 | — | — | — | 0.0 | 0.5968 |
| ADR-0012 v1 (Canny + ECC + dil 1) | 0.225 | 0.834 | 0.831 | 0.740 | 25 % | 0.0 | 0.8040 |
| ADR-0012 Plan A (Canny + dil 3 + border 15) | 0.102 | 0.539 | 0.585 | 0.613 | 53 % | 0.0 | 0.6399 |
| ADR-0013 v1 (validator, fade thresh 0.5) | n/a | 0.970 | 0.939 | 0.910 | 16 % | 0.0 | 0.8865 |
| ADR-0013 v2 (validator, fade thresh 0.75) | n/a | 0.981 | 0.939 | 0.924 | 16 % | 0.0 | 0.8928 |

Best result on this branch is ADR-0013 v2: **0.8928 vs LCN's
0.8958**, i.e. **-0.003**. Within scoring noise of parity; not an
improvement.

### 8.2. What worked architecturally

- **Fansub-direct OCR** (§3.1). With PaddleOCR seeing the natural
  image instead of the masked composite, text_plain matched LCN
  exactly (0.981). The composed-frame OCR-input architecture was a
  liability we removed cleanly.
- **Per-detection validator** (§3.2). Mean abs luma diff over the
  detection bbox is a cheap, well-behaved signal that does
  discriminate burnt-in from in-frame text at the bbox level.
  Precision held at 0.924 (matching LCN's 0.925) and the noise-event
  rate dropped to 16 % (vs 25-53 % in the ADR-0012 attempts).
- **Mask removal + diff simplification** (§3.4). LCN, Sobel,
  `make_mask`, `compose`, `MaskPresenceRecorder`, and
  `FrameProcessingConfig` all came out cleanly with no functional
  regression — the production tests stayed green throughout.

### 8.3. Why the final score still didn't beat LCN

Two unrecovered losses:

1. **Recall: -0.015 (0.954 → 0.939).** The validator drops a small
   number of real burnt-in subtitles whose bbox happens to overlap
   moving anatomy (so the diff inside the bbox is high for both
   reasons, not just text). The bbox padding (4 px) doesn't change
   the discriminator; a lower threshold would recover recall but
   cost precision symmetrically. There's no operating point that
   beats LCN on both metrics without per-event tuning.
2. **Fade: 0.0, unchanged.** The §3.5 signal pivot is sound (per-bbox
   diff intensity at 16×16 resolution does discriminate event vs
   non-event), but the threshold-crossing *algorithm* assumes the
   fade ramp lives in the pre/post-event window. KenIchi's reference
   places the ramp *inside* the event (e.g. `\fad(500, 0)` means
   "ramp in over the first 500 ms of the event"). The detector walks
   outward from `event.start`, sees baseline noise, and never finds a
   ramp because the ramp is in the other direction. ADR-0010's
   algorithm fundamentally doesn't fit this reference convention.

The combined budget (the recall lost + the fade we couldn't capture)
is small enough that ADR-0013 v2 is *par* with LCN, but the LCN
baseline only "works" because its saturated mask makes compose a
no-op — it's an accidental architecture, not a designed one.

### 8.4. Why we reverted anyway

The user goal is "maximise the score". ADR-0013 didn't move it. The
code cleanup (~400 LOC of dead diff/mask/compose plumbing) has real
maintenance value but no score impact, so it's not the win this
branch was supposed to produce. Shipping a structural refactor that
doesn't show up in the user-visible metric isn't honest progress.

Three combined attempts (ADR-0011, 0012, 0013) failed to beat the
LCN baseline at the score level. The pattern is stable: any change
that makes the mask/diff *more* selective either hurts OCR input
(0011, 0012) or replaces an accidental working passthrough with a
designed-but-equivalent one (0013). The next attempt should look
elsewhere — style, position, intent, fade-algorithm — rather than
revisit the diff/mask plumbing.

### 8.5. Status

Reverted on the branch before merge. The validator, the new diff
implementation, the renamed iterator, the fade signal pivot, and the
mask/compose removals are all gone from the codebase. `lcn_sigma`,
`std_floor`, `FrameProcessingConfig`, `ComposedFrame`,
`iter_composed_frames`, `make_mask`, `compose`,
`MaskPresenceRecorder`, and the mask-alpha fade signal source are
restored. The three ADR documents (0011, 0012, 0013) and their
README links are kept as the record of what was tried and why each
attempt failed.

### 8.6. What's worth preserving for the next attempt

If a future ADR revisits text-detection-as-mask-source:

- The per-detection mean-abs-luma-diff validator from §3.2 is a
  good baseline metric. It discriminates well at the bbox level on
  KenIchi (precision 0.924, recall 0.939).
- Fansub-direct OCR (§3.1) is the right pattern. The composed-input
  architecture cost text_plain noticeably while gaining nothing.
- An intra-event fade algorithm (walk start → start+window looking
  for the climb from baseline to in-event reference) is the
  natural next step for the fade sub-score — but it's an ADR-0010
  follow-up, not a mask/diff pipeline change.
