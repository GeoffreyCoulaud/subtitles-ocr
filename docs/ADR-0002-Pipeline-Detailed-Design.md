# ADR-0002: Pipeline Detailed Design

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: Designed, not yet implemented.
Origin: Pick-my-brain follow-up session between user and Claude, building on ADR-0001.
Supersedes: nothing.
Revises: sections of ADR-0001 listed in §1 below.

This ADR transforms ADR-0001 (scoping decisions) into an implementation-grade design. Each stage now has an algorithmic recipe, an I/O contract, a persistence policy, and tunable parameters explicitly identified. Empirical tuning values still defer to prototyping (cf. §9 and ADR-0001 §17).

---

## 1. Revisions to ADR-0001

ADR-0001 stands as the historical scoping record. This ADR revises the following parts and is the current source of truth for them:

| ADR-0001 section | Status | This ADR |
|---|---|---|
| §4 step 2 (Adaptive phash alignment) | Replaced | §3 Stage 2 — hybrid audio + phash |
| §4 step 6 (PaddleOCR PP-OCRv4) | Updated | §3 Stage 6 — PP-OCRv5 server |
| §4 steps 7 ↔ 8 ordering (color before group) | Inverted | §2 revised sequence |
| §4 step 9 (per-event LLM cleanup) | Extended | §3 Stage 9 — now also does text reconciliation |
| §5 (alignment failure policy: ≥10s contiguous = hard stop) | Replaced | §3 Stage 2 — ratio-based hard stop (>30% orphan) + user skip ranges |
| §9 (CLI) | Extended | §4 (full consolidated CLI) |
| §10 (workdir intermediates) | Made concrete | §5 workdir layout |
| §17 (empirical tunables) | Extended | §9 |

All other ADR-0001 sections (problem statement, inputs, in-scope/out-of-scope styles, compute target, test fixtures, logging, etc.) remain unchanged and are not duplicated here.

---

## 2. Revised pipeline sequence

```
 1. Spatial conform
 2. Adaptive alignment      (audio-primary + phash refinement + phash-only fallback)
 3. Diff
 4. Mask formation
 5. Compose
 6. OCR                     (PaddleOCR PP-OCRv5 server)
 7. Group                   (per-quad tracking, was ADR-0001 step 8)
 8. Color extraction        (per-event, was ADR-0001 step 7)
 9. Per-event LLM cleanup   (now includes text reconciliation across OCR variants)
10. Whole-document LLM cleanup
11. Export ASS
```

Stage 7 and 8 are swapped vs ADR-0001: color extraction now runs after grouping so it can aggregate pixels across all member frames of an event (pixel pool with quad-rectified perspective warp), producing more stable color estimates than per-frame extraction.

---

## 3. Per-stage detailed spec

### Stage 1 — Spatial conform

- Raw is pre-conformed via ffmpeg in an upstream pass; fansub is read in streaming from its original file (no transcode of fansub).
- Python ffmpeg bindings preferred (`ffmpeg-python` or `PyAV`) for progress reporting; subprocess fallback acceptable.
- CLI flag `--ar-strategy {letterbox,crop,error}` default `error` for aspect-ratio mismatches between fansub and raw. Test pair (both 4:3) is unaffected.
- Filter chain:
  - Downscale: `scale=...:flags=area` (smooth, no ringing — preserves clean edges for the diff downstream).
  - Bit depth: `format=yuv420p` truncation, no dither (dither would add per-frame uncorrelated noise that degrades the diff).
  - Colorspace: detect-and-convert via `ffprobe` (e.g. BT.709 source → BT.601 to match SD fansub when applicable). Generic — not hardcoded to a specific source pair.
- Output: `ffv1` lossless in `mkv` container, audio and subtitles dropped (`-an -sn`). ~3–5 GB per episode.
- Cache: `01_conform/raw.mkv` + `raw.meta.json` sidecar (source path, mtime, size, target resolution, ar-strategy, conform code version). Re-run reuses cache iff sidecar matches; otherwise re-transcodes.

### Stage 2 — Adaptive alignment (hybrid)

Three sub-stages with explicit decision tree:

**Sub-stage 2a — Audio coarse alignment** (only if both `--hardsub-audio-track` and `--raw-audio-track` provided):
- Extract both audio tracks via ffmpeg upstream pass; mono downmix at 16 kHz; cached as WAV in `02_alignment/` with sidecars.
- VAD: `silero-vad` via ONNX (avoids the torch dependency; ~2 MB model). Natural frame rate ~31 Hz (one VAD probability per 32 ms window). Sub-frame precision for our 23.976 fps target.
- Signature: continuous VAD probabilities, **z-score normalized per source** to eliminate systematic level biases between fansub and raw mixes.
- Matching algorithm: **hierarchical batch cross-correlation in 3 passes**:
  - Pass 1: split both audios into coarse windows (~30–60 s), compute normalized cross-correlation per window, record `(offset, peak_height, peak_to_noise)` where `peak_to_noise = peak_height / std(curve_without_peak_neighborhood)`.
  - Pass 2: identify suspicious windows (ambiguous verdict OR offset diverging from neighbors).
  - Pass 3: subdivide suspicious windows recursively until clean verdict or minimum window size reached.
- Verdict per window:
  - `peak_height < THRESH_LOW` → `no_match` (candidate orphan).
  - `peak_height ≥ THRESH_HIGH` AND `peak_to_noise ≥ THRESH_SNR` → `confident_match`.
  - Otherwise → `ambiguous` → subdivide. Final ambiguous at min size → orphan (conservative).
- Post-Pass-3 filter: a matched segment is **reclassified ORPHAN** if its duration < `MIN_MATCH_S` AND it has no immediate neighbor (left or right) with a coherent offset (delta < `OFFSET_TOLERANCE_FRAMES`). Eliminates coincidental tiny matches in mostly-orphan regions, silence-matches-silence artefacts, and periodic-content false matches.

**Sub-stage 2b — Phash refinement** (only if 2a ran successfully):
- For each fansub frame `N` in an `ALIGNED` segment from 2a, compute phash distance to raw frames in window `[predicted-W, predicted+W]` with W=2 (fixed).
- Phash variant: `cv2.img_hash.PHash` (64-bit DCT-based; robust to local subtitle differences which are mostly high-frequency).
- `best_k = argmin(distance)`; `d_best = distance[best_k]`.
- If `d_best ≤ THRESH_AGREE` (~10/64 bits baseline) → `agreement_flag = True`. Otherwise `agreement_flag = False` (accept best-k anyway, but flag).
- Aggregate ratio of `agreement_flag == False` over all frames; if `> THRESHOLD_DISAGREE` (~30% baseline) → trigger Cas E (fallback to 2c).

**Sub-stage 2c — Phash fallback** (ADR-0001 original algorithm, formalized):
- Per-frame phash matching with adaptive search window.
- Initial window `W_INITIAL`, grows by `grow_step` on miss, shrinks by `shrink_step` after consecutive hits, bounded by `[W_MIN, W_MAX]`.
- Match threshold `THRESH_MATCH`. Misses logged; contiguous miss spans collapse into ORPHAN segments.
- Same post-processing as 2a+2b (orphan ratio threshold, isolated match filter).

**Decision tree (Q18)**:
- Tracks audio absent (one or both flags missing) → skip 2a/2b, run 2c directly.
- Audio extraction fails (corrupt track, ffmpeg error) → skip 2a/2b, run 2c with warning.
- 2a completes but `aligned_ratio < 70%` → **hard stop** (sources likely incompatible; phash-only fallback would not recover).
- 2b detects systematic disagreement > 30% → fallback to 2c (audio and video desynced).
- Otherwise → 2a + 2b is the result.

**Failure policy**:
- ADR-0001 §5 ("≥10s contiguous unaligned = hard stop") is **replaced** by: orphan ratio `> 30%` after all post-processing = hard stop. Equivalently, alignment must succeed on `≥70%` of fansub frames.
- New CLI mechanism: `--hardsub-skip "HH:MM:SS-HH:MM:SS"` and `--raw-skip "HH:MM:SS-HH:MM:SS"` (both repeatable). Pre-declared ranges are excluded from alignment entirely and from the denominator of the 70% threshold. Used when the user already knows parts of the videos don't have counterparts (extended OPs, censored scenes, recap segments).

**Output schema** (`02_alignment/alignment.json`, atomic):

```python
class AlignmentSegment(BaseModel):
    fansub_frame_start: int          # inclusive
    fansub_frame_end: int            # exclusive
    raw_frame_start: int | None      # None if ORPHAN or USER_SKIPPED
    raw_frame_end: int | None
    offset_frames: int | None        # raw_start - fansub_start if ALIGNED
    status: Literal["ALIGNED", "ORPHAN", "USER_SKIPPED"]
    confidence_avg: float | None     # mean peak_to_noise (2a) or inverse Hamming (2c)

class AlignmentResult(BaseModel):
    fansub_total_frames: int
    raw_total_frames: int
    method_used: Literal["audio+phash_refinement", "phash_only"]
    aligned_ratio: float
    orphan_ratio: float
    user_skipped_ratio: float
    segments: list[AlignmentSegment]  # cover [0, fansub_total_frames] without gap or overlap
    warnings: list[str]
```

### Stage 3 — Diff

- Grayscale BT.709 luma conversion at entry; the rest of the pipeline operates on one channel.
- Local contrast normalization (LCN): `(pixel - mean_local) / max(std_local, std_floor)`. Gaussian for `mean_local`/`std_local`. Suppresses both brightness and contrast differences globally and locally (handles regrade between fansub and Bluray remaster).
- Sobel 3×3 gradient on the LCN-normalized image; `magnitude = sqrt(gx² + gy²)`.
- Diff: `abs(gmag_fansub - gmag_raw)`, float32.
- Persistence: streamed into stage 4 in production (no diff cache on disk — ~42 GB would be intractable). Under `--debug-images`, save visualization PNGs in `03_diff/debug/`.
- Logically separate module from stage 4, physically composed in streaming.

### Stage 4 — Mask formation

- Gaussian smoothing of the diff map (σ TBD empirically, order 1–2 px).
- **Hysteresis thresholding** (Canny-style two-threshold): pixels with diff `> T_high` are seeds; pixels with diff `> T_low` are accepted iff connected (8-neighbor) to a seed. Ratio `T_high/T_low ~ 2–3`. Robust to threshold choice.
- Connected-components filter: size only (`AREA_MIN`, `AREA_MAX` TBD). No aspect-ratio, position, or density filters — subtitle shapes are too varied. DBNet at stage 6 handles fine spatial selection.
- Final morphological dilation 3×3 (1 iteration) to ensure the outline is included in the mask.
- Output: `04_mask/frames/<fansub_frame_idx>.png` (binary PNG, ~2–5 KB/frame). Resume by file presence.

### Stage 5 — Compose

- `composed = fansub * (mask / 255)[..., None]`. Background where `mask == 0` is pure black RGB(0,0,0).
- Black is optimal for downstream DBNet (zero edges → zero false detections out of mask) and contrasts well with the dominant fansub style (white/yellow fill + black outline).
- Streamed into stage 6 in production. Under `--debug-images`, save RGB PNGs in `05_compose/frames/`.

### Stage 6 — OCR

- PaddleOCR **PP-OCRv5 server** (default model since recent paddleocr versions; supersedes the v4 mention in ADR-0001).
- Initialization:

  ```python
  ocr = PaddleOCR(
      lang=args.language,                  # default "latin"
      use_doc_orientation_classify=False,  # video frame, orientation known
      use_doc_unwarping=False,             # no page curvature
      use_textline_orientation=True,       # CRUCIAL — anime signs can be rotated
  )
  ```

- API: `result = ocr.predict(image)`; access via `result[i].json` → `rec_texts`, `rec_scores`, `rec_polys` (oriented quads).
- Device handling: `--ocr-device {auto,cuda,rocm,cpu}`.
  - `auto` (default): detect, prefer GPU if available, **fall back to CPU with warning** if GPU init/test fails.
  - `cuda`/`rocm`: **hard fail** if init/test fails (no silent fallback when user is explicit).
  - `cpu`: baseline that must always work.
- Realistic device constraints: PaddlePaddle has solid CUDA support, experimental ROCm (limited ops), no MPS. M3 Pro and RX 7700XT may end up CPU-only in practice; to validate at prototyping.
- Single-frame loop (batching deferred as a future optimization).
- Output: `06_ocr/results.jsonl`, one line per ALIGNED frame:

  ```python
  class OcrDetection(BaseModel):
      text: str
      confidence: float
      quad: list[tuple[int, int]]  # 4 points (TL, TR, BR, BL), oriented

  class FrameOcrResult(BaseModel):
      fansub_frame_idx: int
      detections: list[OcrDetection]  # empty if no text detected
  ```

- Chunk size: 500 frames (flush + fsync per chunk). Resume: re-read JSONL, restart after the last persisted `fansub_frame_idx`. SIGINT: finish the current chunk write, exit cleanly.
- Sidecar: model, device, language. Cache invalidated if any of these change.

### Stage 7 — Group (per-quad tracking)

Replaces the per-frame grouping implied by ADR-0001. Groups detections, not frames, into events.

- Each (frame, quad) is a "detection." Stage 7 forms temporal trajectories of detections across consecutive ALIGNED frames.
- Trajectory continuation criterion frame `N` → frame `N+1`: text Levenshtein distance normalized by length `< 0.2` **AND** quad IoU `> 0.5`.
- Per-quad tracking guarantees that simultaneous top and bottom subtitles on the same frame become two distinct events (one trajectory per detection).
- ORPHAN/USER_SKIPPED frames in the middle of a trajectory do **not** break it (continuity judged on neighboring ALIGNED frames).
- A frame with **no OCR detection** in the middle of a trajectory **does** break it. Preserves artistic repetitions (e.g., two identical lines separated by a tiny silent gap).
- Aggregations per event:
  - `raw_ocr_texts`: list of N texts (one per ALIGNED member frame). **No canonical text decided here** — that role moves to stage 9.
  - `raw_ocr_confidences`: list of N confidences.
  - `quads`: median per vertex (TL/TR/BR/BL) across member frames.
  - `member_frame_indices`: list of ALIGNED frame indices that contributed.

Output: `07_group/events.json` (atomic):

```python
class SubtitleEvent(BaseModel):
    event_id: int
    fansub_frame_start: int
    fansub_frame_end: int
    raw_ocr_texts: list[str]
    raw_ocr_confidences: list[float]
    quads: list[tuple[int, int]]  # 4 vertices, median per coord
    member_frame_indices: list[int]

class GroupResult(BaseModel):
    fansub_total_frames: int
    events: list[SubtitleEvent]
    stats: dict
```

### Stage 8 — Color extraction (post-group, per-event)

For each event:

- For each member frame, **rectify the quad** via perspective warp (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`) into a canonical canvas of size `(W_canon, H_canon) = median(member widths/heights)`. Works uniformly for static subs, translation (`\move`), rotation, and perspective.
- Crop padding before warp: extend the quad by +10% of its diagonal in each direction so the outline (which spills beyond DBNet's text-tight quad) is captured.
- Stack rectified crops into `(N, H_canon, W_canon, 3)`, take **temporal median** pixel-wise → one canonical image per event. Median absorbs anti-aliased background bleed-through (varies per frame) while preserving the stable glyph.
- Apply the ADR algorithm to this canonical image:
  - Otsu **global** within the crop (the crop is already localized to a text region → distribution is bimodal at this scale).
  - Distance transform on the glyph mask. Stroke width = `p95` of the distance values on glyph pixels (`p95` is robust; `max` would be outlier-sensitive).
  - Erode the glyph mask by a circular kernel of radius `floor(0.4 * stroke_width_p95)`. Interior pixels = fill, eroded-out edge pixels = outline.
  - Color clustering per region: **mode in HSV with 16-bin quantization**, re-converted to RGB. Mode is robust against numerous but minoritarian anti-aliasing edge pixels; HSV binning groups perceptually-close hues regardless of value variation.
- `style_supported = False` if any of:
  - Interior pixel pool < 100 px (glyphs too thin or stroke-width estimation wrong).
  - Variance of H in interior pool > threshold (non-uniform fill: gradient, multi-color).
  - Variance of H in outline pool > threshold (non-uniform outline).
  - Ratio interior_pool_size / outline_pool_size aberrant (`< 0.1` or `> 10`).
- Output `08_color/colors.json` (atomic):

```python
class EventColors(BaseModel):
    event_id: int
    fill_color: tuple[int, int, int] | None     # None if unsupported
    outline_color: tuple[int, int, int] | None  # None if unsupported
    style_supported: bool
    stroke_width_px: float  # debug info

class ColorExtractionResult(BaseModel):
    events: list[EventColors]
    stats: dict
```

### Stage 9 — Per-event LLM cleanup (with reconciliation)

Two roles in one call: reconcile the N OCR variants per event into a single canonical text, and fix common OCR confusables (`rn`/`m`, `I`/`l`, accent recovery).

- **Pre-check consensus**: if all `raw_ocr_texts` of an event are strictly identical (`set(texts) | == 1`), use the text directly, skip LLM call. Common case for "clean" events.
- Otherwise → LLM call:
  - Client: Ollama via OpenAI-compatible HTTP (same pattern as existing `src/subtitles_ocr/vlm/client.py`).
  - Model: configurable via `--event-cleanup-model <name>` (default to be set empirically).
  - `response_format` = JSON schema strict: `{"text": "..."}` (`CleanedEvent` Pydantic model).
  - **No context** beyond the event itself (parallelizable 100%, clean role separation with stage 10).
- Retry: 2 retries with exponential backoff (1 s, then 3 s). If all 3 attempts fail (timeout, invalid JSON, schema mismatch) → **hard stop** with event_id identification. Resume picks up from the last persisted event after user diagnosis.
- Parallelism: `ThreadPoolExecutor` with `--event-cleanup-parallelism` (default 4). **Parallelism does not invalidate cache** (sidecar tracks only the model name); only `--event-cleanup-model` changes invalidate the chunk.
- Output `09_event_cleanup/cleaned.jsonl`, chunk size 100 events:

  ```json
  {"event_id": 0, "cleaned_text": "...", "skipped_llm": true|false}
  ```

  `skipped_llm` distinguishes consensus-derived texts from actual LLM-derived ones (debug utility).

### Stage 10 — Whole-document LLM cleanup

- Single LLM call over **all** events of the episode (no chunking with overlap, no rolling window).
- Prompt structure:
  - System: instructions for narrative coherence (consistent character names, ponctuation, optional synopsis application).
  - User: JSON `{"synopsis": <synopsis text or null>, "events": [{"id": int, "text": str}, ...]}`.
  - Strict response schema: same list, same IDs, same length. Validation post-call: mismatch → **hard fail**.
- If the assembled prompt exceeds the model's context window → hard fail with a message suggesting the user pick a larger model or revise content. No automatic chunking fallback (philosophy: fail loud rather than silently degrade output coherence).
- `--synopsis <path>`: free Markdown text loaded as-is into the prompt. No imposed schema (the LLM extracts linguistic cues from free text as well as from structured input).
- Client, retry policy, sidecar/cache invalidation rules: same as stage 9.
- CLI: `--doc-cleanup-model`, `--doc-cleanup-parallelism` (default 1: stage is naturally sequential, parallelism only relevant if implementing chunked fallback later).
- Output `10_doc_cleanup/cleaned_final.json` (atomic): list of events with final canonical text.

### Stage 11 — Export ASS

Library: `pysubs2` (mature, native ASS/SSA, handles encoding, timing format, escaping).

**Sub-step "style synthesis"** before writing the `.ass`:

1. Position classification per event from the centroid of the canonical quad:
   - `Bottom`: centroid in the lower third AND horizontally centered (±20% of width center).
   - `Top`: centroid in the upper third AND horizontally centered.
   - `Sign`: everything else.
2. Color clustering (perceptual, LAB + ΔE76):
   - Distance between two events: `max(ΔE76(fill_a, fill_b), ΔE76(outline_a, outline_b))`. Both colors must be close for two events to share a style.
   - Greedy clustering: process events in order; join an existing cluster if distance to its centroid `< --color-cluster-threshold` (default `10`); otherwise create a new cluster.
   - Canonical color of each cluster: mean in LAB, re-converted to RGB.
   - ΔE76 chosen over ΔE2000: at the threshold (~10, well above the JND of ~2.3), the corrections that ΔE2000 brings are negligible. ΔE76 is a one-line numpy operation; ΔE2000 would require scikit-image. Acceptable accuracy without the extra dependency.
3. Style naming: `<Position>-<ColorIndex>` where index 0 is reserved for the dominant color of each position (most frequent across the episode, typically `WhiteFill+BlackOutline`).
   - `Bottom-0`, `Bottom-1`, ...
   - `Top-0`, `Top-1`, ...
   - `Sign-0`, `Sign-1`, ...
4. Events with `style_supported = False` (stage 8): assigned to `<Position>-Default`, a separate style group inheriting standard white-fill/black-outline. Editor can redefine these styles freely in post-production without touching the events.

Inline tags per event:
- `Bottom-*`, `Top-*`: none (style fully defines the appearance).
- `Sign-*`: `{\pos(x,y)\frz(angle)}`. Omit `\frz` if `|angle| < 2°`. Angle = orientation of the canonical quad's `TL→TR` edge vs horizontal (ASS convention: clockwise positive).
- Colors: **never** inline. Always in the style. Editors can globally change a color by editing one style in Aegisub.

Timing per event covering frames `[start, end[` at `fps`:
- `start_time_ms = round(start * 1000 / fps)`
- `end_time_ms = round(end * 1000 / fps)`
- Formatted as `H:MM:SS.cc` (centiseconds) by pysubs2.

Multi-line text: `\n` in `cleaned_text` (introduced by stage 10) → `\N` (ASS hard line break).

`[Script Info]` block:

```
ScriptType: v4.00+
PlayResX: <fansub_width>
PlayResY: <fansub_height>
WrapStyle: 0
ScaledBorderAndShadow: yes
```

Default style (`<Position>-Default`): `Arial`, size 60, fill white, outline black, outline thickness 2, alignment 2 (bottom-centered). Conservative defaults that render acceptably at any resolution.

Encoding: UTF-8 without BOM (pysubs2 default; compatible Aegisub/MPV/VLC).

Output: `--out <path>.ass`, atomic write (`<path>.tmp` then rename). No sidecar (terminal stage, no cache).

---

## 4. Consolidated CLI

```
subtitles-ocr \
  --hardsub <fansub.avi> \
  --raw <bluray.mkv> \
  --out <output.ass> \
  [--language latin] \
  [--synopsis path/to/synopsis.txt] \
  [--workdir path/to/intermediates] \
  [--debug-images] \
  [--ar-strategy error|letterbox|crop] \
  [--hardsub-audio-track <idx>] \
  [--raw-audio-track <idx>] \
  [--hardsub-skip "HH:MM:SS-HH:MM:SS"] (repeatable) \
  [--raw-skip "HH:MM:SS-HH:MM:SS"] (repeatable) \
  [--ocr-device auto|cuda|rocm|cpu] \
  [--event-cleanup-model <ollama-name>] \
  [--event-cleanup-parallelism <int>] \
  [--doc-cleanup-model <ollama-name>] \
  [--doc-cleanup-parallelism <int>] \
  [--color-cluster-threshold <float>]
```

Single mode on this branch — no flag-based switching to a legacy pipeline (consistent with ADR-0001 §9).

---

## 5. Workdir layout

```
workdir/
  01_conform/        raw.mkv, raw.meta.json
  02_alignment/      hardsub_audio.wav, raw_audio.wav, *.meta.json, alignment.json
  03_diff/           (empty in prod, debug/ under --debug-images)
  04_mask/           frames/<idx>.png
  05_compose/        (empty in prod, frames/<idx>.png under --debug-images)
  06_ocr/            results.jsonl, results.meta.json
  07_group/          events.json, events.meta.json
  08_color/          colors.json, colors.meta.json
  09_event_cleanup/  cleaned.jsonl, cleaned.meta.json
  10_doc_cleanup/    cleaned_final.json, cleaned_final.meta.json
```

Convention: one numbered subdirectory per stage; `*.meta.json` sidecar carries inputs/config for cache invalidation; per-frame artifacts (mask, debug images) live in `frames/` subdirectories; slow-stage outputs (OCR, event cleanup) use JSONL append for crash-resilient resume.

---

## 6. Persistence and resume

- **Fast stages** (1, 2, 3+4 streaming, 5 streaming, 7, 8, 10, 11): atomic JSON or final artifact written at end of stage. Mid-stage interrupt re-runs the whole stage.
- **Slow stages** (6 OCR, 9 event cleanup): JSONL append per chunk (N=500 for OCR, N=100 for event cleanup). Resume reads the JSONL, restarts after the last persisted entry. SIGINT handler completes the current chunk write and exits cleanly.
- **Cache invalidation**: each stage's sidecar records the inputs and config that affect its output (source paths, mtimes, sizes, model names, language, ar-strategy, etc.). On re-run, sidecars are compared; mismatch → invalidate and recompute. Parallelism settings are runtime-only and do not invalidate caches.
- **No `--resume` flag**: resume is implicit. User deletes a stage's directory (or the whole workdir) to force recomputation.

---

## 7. Failure policy summary

- **Hard stops** (pipeline exits with explicit error):
  - Stage 1: ffmpeg error, AR mismatch under `--ar-strategy error`, codec/colorspace probing failure.
  - Stage 2: alignment ratio `> 30% orphan` after all post-processing.
  - Stage 6: `--ocr-device` explicit (cuda/rocm) fails to init or run on test frame.
  - Stage 9: LLM call fails after 2 retries on an event without consensus.
  - Stage 10: LLM call fails after 2 retries, or response schema mismatch (wrong number/IDs of events), or prompt exceeds model context.
- **Soft failures** (warnings logged, pipeline continues):
  - Stage 2: per-frame phash miss within an aligned segment (counted toward orphan ratio).
  - Stage 2: audio extraction failure when `--ocr-device auto` style fallback applies (here: 2a/2b skipped, 2c runs with warning).
  - Stage 6: `--ocr-device auto` failing on GPU and falling back to CPU.
  - Stage 8: `style_supported = False` events (fallback to default style at stage 11).

---

## 8. Out-of-scope confirmations

These were considered during this scoping session and explicitly declined:

- Pre-cropping the phash region to exclude the typical subtitle area: rejected — phash's robustness to local high-frequency changes is sufficient; pre-cropping doesn't apply for signs at arbitrary positions anyway.
- Per-channel diff at stage 3 (R, G, B independent then max): rejected — the cost of 3× compute does not justify mitigating the niche case of colored subs without outline on iso-luminant backgrounds (cf. stage 3 reasoning).
- Census transform / DoG for the diff: rejected in favor of LCN+gradient, which preserves magnitude (used by mask formation) and is more debuggable.
- DBSCAN for mask cluster formation: rejected — heavyweight versus the morphological pipeline, with no quality advantage at our scale.
- Persisting diff frames in cache: rejected — ~42 GB intractable; stages 3+4 stream in production.
- DTW for audio alignment: rejected in favor of hierarchical cross-correlation — DTW's strength (handling continuous non-linear drift) is unnecessary for our piecewise-constant offset structure, and it is more opaque to debug.
- Windowed bidirectional context at stage 9: rejected — parallelization advantage of no-context outweighs the marginal coherence gain (stage 10 handles global coherence).
- Chunked fallback at stage 10: rejected — hard fail on oversize prompts preferred (philosophy: clear failure over silently-degraded coherence).
- ΔE2000 instead of ΔE76 for color clustering: rejected — at the threshold (~10) the additional accuracy of ΔE2000 doesn't change cluster membership; avoids the scikit-image dependency.

---

## 9. Tunables left for prototyping

Extends ADR-0001 §17 with the new tunables introduced by this design:

- **Stage 2 (audio)**:
  - Coarse window initial size (~30–60 s baseline).
  - Subdivision floor (~1–2 s baseline).
  - `THRESH_LOW`, `THRESH_HIGH`, `THRESH_SNR` for the verdict function.
  - `MIN_MATCH_S` and `OFFSET_TOLERANCE_FRAMES` for the post-filter.
- **Stage 2 (refinement)**:
  - `THRESH_AGREE` Hamming distance (~10/64 baseline).
  - `THRESHOLD_DISAGREE` for the fallback trigger (~30% baseline).
- **Stage 2 (phash fallback)**:
  - `W_INITIAL`, `W_MIN`, `W_MAX`, `grow_step`, `shrink_step`, `THRESH_MATCH` (per ADR-0001 §17, unchanged).
- **Stage 3**: Gaussian σ for LCN, `std_floor`.
- **Stage 4**: smoothing σ, `T_high`, `T_low`, `AREA_MIN`, `AREA_MAX`.
- **Stage 8**: hue-variance thresholds for `style_supported`, interior-pool minimum size.
- **Stage 9**: choice of small text-only LLM via `--event-cleanup-model` (per ADR-0001 §17, unchanged).
- **Stage 10**: choice of larger text-only LLM via `--doc-cleanup-model` (per ADR-0001 §17, unchanged).
- **Stage 11**: `--color-cluster-threshold` default (10 proposed).

GPU support for PaddleOCR on ROCm (RX 7700XT) and MPS (M3 Pro) is also empirical — both targets may end up CPU-only in practice.

---

## 10. Decision log (key forks from this session)

| Fork | Decision | Why |
|---|---|---|
| Alignment modality | Hybrid audio-primary + phash refinement + phash-only fallback (vs ADR-0001's phash-only) | Audio is more precise (sub-frame) and faster when available; phash is robust fallback for incompatible audio. |
| Audio matching algorithm | Hierarchical batch cross-correlation in 3 passes | Native handling of multi-offset (mid-episode discontinuities); debuggable; the dichotomy elegantly localizes discontinuities to subdivision floor. |
| VAD implementation | `silero-vad` via ONNX | Modern, robust, no PyTorch dependency, lightweight (~2 MB). |
| Audio signature representation | Continuous VAD probabilities, z-score normalized per source | Compromise robust-to-mix-bias (binary's strength) and informative (continuous's strength). |
| Peak quality metric | `peak_height` + `peak/std` | Well-defined (unlike `peak/second_peak` which is ambiguous); catches degenerate cases including silence-matches-silence. |
| Tiny isolated matches | Post-Pass-3 filter reclassifies as orphan | Eliminates coincidence-matches, music-loop false positives, silence-silence artefacts before they pollute downstream. |
| Stage 7 ↔ Stage 8 ordering | Color extraction moves after group (per-event) | Multi-frame aggregation via quad-rectified median gives dramatically more stable color estimates. |
| Quad-rectified median for color | Used as the only path | The quad provides per-frame tracking → perspective warp handles immobile, translation, rotation, perspective uniformly. The earlier "hybrid only-if-stable" proposal was rejected after user pushback. |
| Stage 7 grouping granularity | Per-quad tracking, not per-frame | Same-frame top+bottom subs become distinct events; tracking trajectory semantics is correct. |
| Frame without OCR mid-event | Breaks the event (no tolerance) | Preserves intentional artistic micro-pauses (e.g., repeated lines for emphasis). |
| Text canonicalization | Delegated to stage 9 LLM (replaces character-wise consensus at stage 7) | Single LLM pass reconciles OCR variants AND fixes confusables, with full information; no information loss to a deterministic consensus first. |
| Stage 9 context strategy | No context (event in isolation) | Clean role separation: stage 9 local, stage 10 global; preserves parallelization. |
| LLM failure policy | Hard stop on LLM error after retries; consensus pre-check before any call | Robustness via fail-loud: never accept potentially-bad data; resume mechanism makes restart cheap. |
| Per-stage LLM configuration | `--<stage>-model`, `--<stage>-parallelism` (each LLM stage independent) | Future LLM stages reuse the same pattern; parallelism kept separate from cache-affecting config. |
| OCR model version | PP-OCRv5 server | Current default since the scoping date; no reason to lag. |
| `--ocr-device` semantics | Auto = warn-and-fallback; explicit = hard fail | Explicit user intent should not be silently overridden. |
| Style synthesis at stage 11 | Group into named styles by position+color cluster | Enables editor workflow (Aegisub modifications) — was the user's explicit requirement that pivoted from "all inline" architecture. |
| Color clustering metric | ΔE76 in LAB, default threshold 10 | Sufficient accuracy at threshold ~10 (well above JND); no scikit-image dependency. |
| ADR revision style | Standalone ADR-0002 with revision pointers; ADR-0001 stays intact | Preserves historical reasoning trail; respects ADR append-only convention. |

---

## 11. Conversation provenance

This ADR was produced by a `pick-my-brain` scoping session that systematically traversed the 11 stages in pipeline order, with one decision at a time and explicit recommendations. The user requested "tout, en profondeur, jusqu'à ce qu'on s'arrête" (option `c` of an initial scope question) and the session reached completion of all 11 stages.

Key inflection moments where the user pushed back productively on initial recommendations:

- **Q11** (audio matching algorithm): user pointed out that naive sliding cross-correlation handles discontinuities-within-a-window poorly; the dichotomy/hierarchical refinement was added in response.
- **Q12** (alignment failure policy): user proposed pre-declarable skip ranges via CLI, replacing the rigid "≥10s contiguous = hard stop" with a percentage-based threshold and explicit user override.
- **Q17** (peak quality metric): user noted that `peak/second_peak` was ambiguous (acknowledged as a self-criticism in the question); switched to `peak/std`.
- **Q22→Q34**: user observed that "tiny matched segments in the middle of large orphan regions" should be flagged as false matches by default; led to the post-Pass-3 filter.
- **Q36**: user challenged the rejection of median-frame approach for moving subs ("the quad tells us which pixels to choose"); the rectified-median approach replaced the hybrid recommendation, simplifying the design.
- **Q39 follow-up**: user replaced the proposed LLM-failure fallback (character-wise consensus) with a hard-stop policy, citing the principle "fail loud rather than accept bad data."
- **Q43 follow-up**: user pivoted the style architecture from "all inline tags" to "grouped styles for editor workflow," and prompted the separate question of whether top/bottom simultaneous subs are guaranteed to be distinct events (which exposed and corrected an under-specified piece of stage 7).
- **Q44**: user requested justification for "ΔE76 is 10× simpler than ΔE2000"; led to a corrected (more honest) recommendation acknowledging that with scikit-image both are equally easy and the real reason is sufficient accuracy at the chosen threshold.

These pushbacks materially improved the design and are explicitly logged in §10.
