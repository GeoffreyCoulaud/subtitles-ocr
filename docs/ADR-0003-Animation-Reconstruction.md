# ADR-0003: Animation Reconstruction (`\move` + `\fad`)

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: Designed, not yet implemented. Implementation deferred until the static MVP (ADR-0002) is validated against the KenIchi ground-truth `.ass` (cf. ADR-0001 §7).
Origin: Pick-my-brain follow-up session between user and Claude, building on ADR-0002.
Supersedes: nothing.
Revises: parts of ADR-0001 §3 (in-scope/out-of-scope styles) and ADR-0002 §2, §3, §5 (pipeline sequence, stage specs, workdir layout).

This ADR scopes the reconstruction of two ASS animation tags — `\fad(t_in, t_out)` and `\move(x1, y1, x2, y2)` — which were explicitly listed as out-of-scope in ADR-0001 §3. Empirical observation shows they degrade extraction quality even when ignored: fades truncate event boundaries and moves fragment events. This ADR introduces a new Stage 8 (Animation Analysis) that detects and reconstructs these tags additively, without invalidating the static-subtitle design of ADR-0002. ADR-0002's stages 8–11 are renumbered to 9–12 to make room for the insertion.

---

## 1. Revisions to ADR-0001 and ADR-0002

| Section | Status | This ADR |
|---|---|---|
| ADR-0001 §3 (out-of-scope: `\fad`, `\move`) | Partial removal | §2 — these tags become in-scope under restricted form |
| ADR-0002 §2 (pipeline sequence) | Extended + renumbered | §3 — new Stage 8 inserted; old Stages 8–11 shift to 9–12 |
| ADR-0002 §3 Stage 7 (`SubtitleEvent.quads`) | Extended | §4.1 — adds `quads_per_frame` (additive, non-breaking) |
| ADR-0002 §3 Stage 8 (color extraction, input) | Renumbered to Stage 9 + extended | §4.3 — color extraction excludes fade-frames from temporal median |
| ADR-0002 §3 Stage 9 (per-event LLM cleanup) | Renumbered to Stage 10 | §3 (unchanged behavior) |
| ADR-0002 §3 Stage 10 (whole-doc LLM cleanup) | Renumbered to Stage 11 | §3 (unchanged behavior) |
| ADR-0002 §3 Stage 11 (export ASS) | Renumbered to Stage 12 + extended | §4.4 — emits `\move` and `\fad` in addition to `\pos`, `\frz`, `\c`, `\3c` |
| ADR-0002 §5 (workdir layout) | Renumbered | §5 — old `08_color`..`10_doc_cleanup` shift to `09_color`..`11_doc_cleanup`; new `08_animation/` |

All other ADR-0002 sections remain unchanged and are not duplicated here.

---

## 2. Scope of animation tags

### In-scope (priority A — initial implementation target)

- `\fad(t_in_ms, t_out_ms)` — symmetric linear fade-in/out. Either side may be `0` (hard cut on that side).
- `\move(x1, y1, x2, y2)` — linear translation over the full event duration.

These two tags cover the great majority of dialogue subtitles (typical `\fad(200, 200)`) and signs (typical `\move` panning across the frame) in French anime fansubbing.

### In-scope (priority B — extensibility, design must not preclude)

- `\move(x1, y1, x2, y2, t1, t2)` — translation with temporal offsets (move triggers partway through event).
- `\fade(a1, a2, a3, t1, t2, t3, t4)` — non-linear alpha envelope with a plateau.

The Stage 8 design is structured to extend to these without architectural rework: priority A uses linear fits across the full event duration; priority B replaces the full-duration assumption with offset-aware fits.

### In-scope (priority C — distant future, bbox-affecting only)

- `\t(\frz...)` — continuous rotation.
- `\t(\fscx..., \fscy...)` — continuous scaling.
- `\t(\move...)` — compound animated translation.

Bounded explicitly to transformations that affect the bounding box (scale + rotate + translate). The pipeline's per-frame quad tracking observes exactly these three degrees of freedom.

### Out of scope (declined)

- Animated colors (`\t(\c...)`, `\t(\1c...)`, `\t(\3c...)`, `\t(\alpha...)`). The diff-based pipeline cannot reliably observe pure color animation independently from background motion.
- Partial fades (alpha that does not complete its 0→1 or 1→0 ramp before the event ends). The fade detection algorithm assumes ranges always run the full {0, 1}; partial fades are flagged and fall back to static.
- Nested `\t(...)` compositions, `\org` rotation-origin offsets, `\frx` / `\fry` 3D rotations, `\clip` / `\iclip` animations.
- Karaoke tags (`\k`, `\kf`, `\ko`) — already out of scope per ADR-0001 §3.

---

## 3. Revised pipeline sequence

```
 1. Spatial conform              (unchanged)
 2. Adaptive alignment           (unchanged)
 3. Diff                         (unchanged)
 4. Mask formation               (unchanged)
 5. Compose                      (unchanged)
 6. OCR                          (unchanged)
 7. Group                        (extended: stores quads_per_frame)
 8. Animation analysis           (NEW)
 9. Color extraction             (was 8 in ADR-0002; extended: excludes fade-frames)
10. Per-event LLM cleanup        (was 9 in ADR-0002; unchanged behavior)
11. Whole-document LLM cleanup   (was 10 in ADR-0002; unchanged behavior)
12. Export ASS                   (was 11 in ADR-0002; extended: emits \move and \fad)
```

Stage 8 is **additive**: disabling it yields the ADR-0002 pipeline behavior exactly (static-only output, fragmented `\move` events kept distinct, truncated boundaries on faded events). Useful property for A/B testing and debug.

---

## 4. Per-stage detailed spec

### 4.1 Stage 7 modifications

Adds one field to `SubtitleEvent`:

```python
class SubtitleEvent(BaseModel):
    event_id: int
    fansub_frame_start: int
    fansub_frame_end: int
    raw_ocr_texts: list[str]
    raw_ocr_confidences: list[float]
    quads_per_frame: dict[int, list[tuple[int, int]]]  # NEW — keyed by fansub_frame_idx
    quad_median: list[tuple[int, int]]                  # WAS named `quads` in ADR-0002
    member_frame_indices: list[int]
```

`quads_per_frame[K]` aligns with `member_frame_indices`: one quad (4 vertices, oriented TL/TR/BR/BL) per ALIGNED member frame. `quad_median` is the pre-existing median aggregation, retained as fallback for stages that don't need per-frame trajectory data.

Cost: O(N × 8 ints) per event. JSON-compact. Resume semantics unchanged (atomic JSON at stage end).

### 4.2 Stage 8 — Animation analysis (NEW)

Runs after Stage 7 has produced events. Two sub-stages, executed in order:

**Sub-stage A — `\move` detection (two levels)**

*Level A1 — intra-event slow move.* For each event from Stage 7:
- Fit linear regression on `quads_per_frame[K]` centers `(cx(K), cy(K))` vs `K`.
- If `total_displacement_px ≥ MIN_MOVE_DISPLACEMENT_PX` AND `R²(cx) ≥ 0.95` AND `R²(cy) ≥ 0.95` → enrich event with linear trajectory `motion = {type: "linear", start: (x1, y1), end: (x2, y2)}`.
- If `total_displacement_px ≥ MIN_MOVE_DISPLACEMENT_PX` but R² fails → flag β (`motion = {type: "nonlinear_flagged"}`).
- Otherwise → `motion = None` (truly static).

*Level A2 — inter-event fragmented move.* Across consecutive events (post-Level-A1):
- Candidate chain: events whose `fansub_frame_end` and next event's `fansub_frame_start` differ by ≤ `MOVE_GAP_TOLERANCE_MS / (1000/fps)` frames.
- Text continuity: pairwise Levenshtein distance < 0.2 on the highest-confidence `raw_ocr_texts` of each event.
- Trajectory continuity: combined `quads_per_frame` across all chain events fit a linear regression with `R²(cx) ≥ 0.95` AND `R²(cy) ≥ 0.95`.
- If all conditions hold → merge the chain into a single event; concatenate `quads_per_frame`, set `motion = {type: "linear", ...}`, regenerate `raw_ocr_texts` and `member_frame_indices`.
- If text + temporal continuity hold but trajectory R² fails → merge as static + flag β.

**Sub-stage B — `\fad` detection (per event, post-merge)**

For each event (whether static, linearly-moving, or β-flagged), measure fade-in (pre-event window) and fade-out (post-event window) independently:

1. **Search window**: frames in `[start - W, start)` and `(end, end + W]` where `W = ceil(FADE_SEARCH_WINDOW_MS / (1000/fps))`. Exclude frames already attributed to another Stage 7 event.

2. **Bbox of score**:
   - If `motion.type == "linear"`: extrapolate the trajectory linearly into the search window (`cx(K), cy(K)` derived from the fitted line; quad geometry kept identical to `start` quad). Bbox = bbox of the predicted quad at K.
   - If `motion.type == "nonlinear_flagged"`: **skip `\fad` detection** for this event. The trajectory is undefined; predicting bbox positions would be unreliable.
   - If `motion is None`: bbox = bbox of `quad_median`.

3. **Score function**: for each frame K in the search window,
   ```
   score(K) = mean(diff_intensity(K) within bbox) / mean(diff_intensity(start) within bbox)
   ```
   where `diff_intensity` is recomputed on-the-fly (LCN + Sobel gradient magnitude, same recipe as Stage 3) restricted to the bbox. Score ≈ 1 when sub is fully opaque, ≈ 0 when absent.

4. **Linear fit with anchor**:
   - Anchor point: `(start_frame, score=1)` for fade-in fit (resp. `(end_frame, score=1)` for fade-out fit). Justified by the no-partial-fade hypothesis.
   - Observations: frames K where `score(K) ∈ [0.05, 0.95]`.
   - Fit constrained linear regression through the anchor + observations.

5. **Decision per side**:
   - If `count(observations) < ceil(MIN_FADE_DURATION_MS / (1000/fps))` → `t = 0` (hard cut on this side; not enough points to fit reliably).
   - Elif `R²(fit) < 0.7` → `t = 0` + warning log (likely scene cut or background pollution, not a real fade).
   - Else → extrapolate the fit line to `score = 0`; compute `t_extrapolated_ms = |start_frame - fade_origin_frame| × (1000 / fps)`.
   - Belt-and-braces: if `t_extrapolated_ms < MIN_FADE_DURATION_MS` → `t = 0` (post-fit sanity check).
   - Belt-and-braces: if `t_extrapolated_ms > FADE_DURATION_CAP_MS` → `t = 0` + warning log (probable cut-scene pollution).
   - Else → `t_in_ms` (resp. `t_out_ms`) = `t_extrapolated_ms`. Extend event boundary: `fansub_frame_start ← fade_origin_frame` (resp. `fansub_frame_end ← fade_end_frame`).

6. **Consistency check**: if `t_in_ms + t_out_ms > duration_ms` after extension → partial fade detected → revert both to 0, mark event as static, log warning. Cohérent with no-partial-fade hypothesis.

**Output schema** (`08_animation/animation.json`, atomic):

```python
class AnimatedEvent(BaseModel):
    event_id: int
    fansub_frame_start: int      # post-extension (may be earlier than Stage 7 start)
    fansub_frame_end: int        # post-extension
    raw_ocr_texts: list[str]
    raw_ocr_confidences: list[float]
    quads_per_frame: dict[int, list[tuple[int, int]]]
    quad_median: list[tuple[int, int]]
    member_frame_indices: list[int]
    motion: dict | None          # {"type": "linear", "start": (x,y), "end": (x,y)}
                                 # or {"type": "nonlinear_flagged"}
                                 # or None (static)
    fade_in_ms: int              # 0 if no fade-in detected
    fade_out_ms: int             # 0 if no fade-out detected

class AnimationAnalysisResult(BaseModel):
    events: list[AnimatedEvent]
    stats: dict                  # counts: static, linear_move, flagged_nonlinear, fade_in_only, fade_out_only, full_fade
```

### 4.3 Stage 9 modifications (was Stage 8 in ADR-0002)

Color extraction excludes fade-frames from the temporal median:

- For each event with `fade_in_ms > 0`: exclude frames `[fansub_frame_start, fansub_frame_start + ceil(fade_in_ms / (1000/fps)))` from the stack.
- For each event with `fade_out_ms > 0`: exclude frames `(fansub_frame_end - ceil(fade_out_ms / (1000/fps)), fansub_frame_end]` from the stack.

Rationale: fade-frames contain alpha-blended background bleed-through, which biases fill/outline color estimates. Excluding them is the resolution of "mechanism 3" (color pollution) called out as a consequence of fixing mechanism 1 (missed fade frames).

Edge case: if remaining frames after exclusion < 3, fall back to `style_supported = False` rather than producing a brittle color estimate.

### 4.4 Stage 12 modifications (was Stage 11 in ADR-0002)

Per-event inline tag emission, in order: position + rotation + animation tags + colors-via-style. Colors remain in the style (never inline), per ADR-0002 §3 Stage 11.

For an event with `motion.type == "linear"`:
- Compute `\pos` from `start` of trajectory (overridden by `\move` at render time but required by ASS spec).
- Emit `\move(x1, y1, x2, y2)` from `motion.start` and `motion.end`.

For an event with `motion is None` or `motion.type == "nonlinear_flagged"`:
- Emit `\pos(cx, cy)` from `quad_median` centroid (standard ADR-0002 behavior).
- For `nonlinear_flagged` events, **prepend a comment** to the dialogue text: `{!sign: animation non reconstruite!}`. Visible to the editor in Aegisub, ignored by renderers. Makes the fallback explicit so post-production catches it.

For any event with `fade_in_ms > 0` OR `fade_out_ms > 0`:
- Emit `\fad(fade_in_ms, fade_out_ms)`. ASS supports asymmetric values (`\fad(0, 200)` and `\fad(200, 0)` are valid).
- If both are 0 → no `\fad` tag.

Combined example: `{\pos(320,240)\move(100,240,540,240)\fad(200,300)}Texte du panneau`.

Timing extension: an event's `fansub_frame_start` / `fansub_frame_end` after Stage 8 reflect the *post-extension* boundaries (including fade ramps). ASS timing uses these directly — the `\fad` durations are measured *within* the event's visible window, as per ASS semantics.

---

## 5. Workdir layout (revised from ADR-0002 §5)

```
workdir/
  01_conform/        raw.mkv, raw.meta.json
  02_alignment/      hardsub_audio.wav, raw_audio.wav, *.meta.json, alignment.json
  03_diff/           (empty in prod, debug/ under --debug-images)
  04_mask/           frames/<idx>.png
  05_compose/        (empty in prod, frames/<idx>.png under --debug-images)
  06_ocr/            results.jsonl, results.meta.json
  07_group/          events.json, events.meta.json
  08_animation/      animation.json, animation.meta.json          (NEW)
  09_color/          colors.json, colors.meta.json                (was 08_color)
  10_event_cleanup/  cleaned.jsonl, cleaned.meta.json             (was 09_event_cleanup)
  11_doc_cleanup/    cleaned_final.json, cleaned_final.meta.json  (was 10_doc_cleanup)
```

Stages 09–11 are functionally identical to ADR-0002's 08–10 (modulo the fade-frame exclusion in 09). Only directory names shift.

---

## 6. Persistence and resume

- Stage 8 is a **fast stage**: atomic JSON write at end-of-stage (`08_animation/animation.json`), same convention as Stages 7, 9, 11.
- Sidecar `animation.meta.json` records the tunables used (so re-run with different tunables invalidates the cache).
- Mid-stage interrupt → whole stage re-runs.
- The on-the-fly diff recomputation in sub-stage B is not cached: ~500 events × ~60 adjacent frames × bbox-local diff is negligible compared to Stage 3 full-frame diff. Re-runs accept this cost.

---

## 7. Failure policy summary

- **Hard stops**: none new. Stage 8 degrades gracefully (events that don't match any animation pattern fall back to static — same as ADR-0002 baseline).
- **Soft failures (warnings logged, pipeline continues)**:
  - β-flagged events (nonlinear `\move` chain merged as static with `{!sign...!}` comment).
  - Fade fits rejected (R² below threshold, duration outside [MIN, CAP]) — falls back to `t = 0`.
  - Partial fade detected (`t_in_ms + t_out_ms > duration_ms`) — both reverted to 0, event marked static.

---

## 8. Tunables (all temporal values in ms; frame counts derived in runtime via `fps`)

| Tunable | Default | Role | Stage |
|---|---|---|---|
| `MIN_MOVE_DISPLACEMENT_PX` | 8 px | Minimum centroid drift for an event to be classified as moving (vs static jitter) | 8.A1 |
| `MOVE_GAP_TOLERANCE_MS` | 200 | Max temporal gap between fragmented events to consider merging | 8.A2 |
| `MOVE_R2_THRESHOLD` | 0.95 | Minimum R² on `cx(t)` and `cy(t)` linear fits for "linear" classification | 8.A1, A2 |
| `MOVE_TEXT_LEVENSHTEIN_MAX` | 0.2 | Max normalized Levenshtein for text similarity in fragmented chains | 8.A2 |
| `FADE_SEARCH_WINDOW_MS` | 1250 | Width of pre/post-event search window | 8.B |
| `MIN_FADE_DURATION_MS` | 125 | Minimum detectable fade duration (sub-frame precision via linear extrapolation) | 8.B |
| `FADE_DURATION_CAP_MS` | 1000 | Hard cap above which fade is flagged as suspect (likely scene cut) | 8.B |
| `FADE_SCORE_FIT_RANGE` | [0.05, 0.95] | Score interval used as observations in the fade fit | 8.B |
| `FADE_FIT_R2_THRESHOLD` | 0.7 | Minimum R² to accept the fade fit (lower bar than `\move` because score is noisier) | 8.B |

All defaults are **placeholder baselines** to be tuned during prototyping against the KenIchi ground-truth (cf. ADR-0001 §7).

---

## 9. Decision log (key forks from this session)

| Fork | Decision | Why |
|---|---|---|
| Animation reconstruction scope | Priority A (`\fad` simple + `\move` linear full-duration); B/C extensible later | A covers 90%+ of dialogues and majority of signs in FR fansub anime; B/C are rarer and benefit from validating A first |
| Animated colors | Out of scope, declined permanently | Diff-based pipeline cannot observe pure color animation independently of background motion |
| Partial fades | Out of scope; flagged as static + warning | Simplifies fade detection algorithm dramatically (anchor at score=1); rare in practice |
| Architecture | New additive Stage 8 post-Stage 7 (α) over refactoring Stage 7 (β) | Lower error cost; works at event level with full context; disabling Stage 8 yields ADR-0002 behavior exactly |
| Pipeline renumbering | Old stages 8–11 shift to 9–12 to insert Stage 8 (Animation analysis) | Sequential integer numbering preserved; avoids ambiguous "Stage 7.5" half-numbers; workdir directory names follow the same shift |
| `\move` detection levels | Two-level (intra-event slow drift + inter-event fragmented chains) | Stage 7's IoU > 0.5 criterion catches some slow moves as a single static event with drifted quad median — needs explicit handling |
| Stage 7 `quads_per_frame` | Add per-frame quads alongside `quad_median` | Required for intra-event move detection in Stage 8; additive, non-breaking |
| Non-linear `\move` chains | β: merge as static with `{!sign: animation non reconstruite!}` comment | Avoids silent information loss; editor sees the flag in Aegisub and retouches manually |
| `\fad` detection method | Linear regression on diff-intensity score within bbox, anchored at score=1 | Robust to noise; sub-frame precision; explicitly relies on extrapolation since score=0 is never observed |
| `\fad` for moving subs | Bbox extrapolated along the linear trajectory | Consistent with ASS semantics (fade alpha rampes while sub is in motion); bbox must follow trajectory |
| `\fad` for β-flagged events | Not attempted | Trajectory undefined → bbox prediction unreliable; editor handles fade manually too |
| Color pollution by fade-frames | Stage 9 excludes fade-frames from temporal median explicitly | Resolves "mechanism 3" as feature once fade detection is in place |
| Tunable units | All temporal tunables in ms; frame counts derived at runtime | Multi-framerate portability (24/25/30/60 fps); cohérent avec ADR-0001 §16 multi-show architecture |
| Min fade duration | 125 ms (≈ 3 frames @ 24fps) | Sub-100ms fades unlikely in practice; threshold gates noise-driven false positives |
| Search window | 1250 ms (≈ 30 frames @ 24fps) | Covers typical fades (200–500ms dialogue, 500–1000ms signs) with margin |
| Hard cap | 1000 ms on `t_in` / `t_out` | Beyond this, fade likely conflated with scene cut or background change |
| Fade format | `\fad(t_in, t_out)` always with both params (0 on absent side); omitted if both = 0 | ASS canonical sémantique; `\fad(0, 200)` and `\fad(200, 0)` both valid |
| Implementation timing | Design only now (ADR-0003); implementation after MVP static validation | Capture the design while fresh; bind to empirical signal from ground-truth before coding |

---

## 10. Conversation provenance

This ADR was produced from a `pick-my-brain` follow-up session that traversed seven structured questions in order:

1. **Ambition level** (a/b/c — robust defensive / detect+flag / full reconstruction): user chose c with b as fallback, a as worst-case.
2. **Failure mechanisms in the current pipeline**: user confirmed mechanisms 1 (fade frames missed) and 2 (move breaks tracking); identified mechanism 3 (color pollution by fade-frames) as a derived consequence — rare in current pipeline because mechanism 1 already discards those frames, but redevenu un problème dès qu'on résout 1.
3. **Animation tag scope**: user prioritized A; open to B; for C explicitly excluded animated colors, kept only bbox-affecting transforms (scale + rotate + translate).
4. **Architecture placement**: user agreed with α (additive Stage 8 post-tracking) over β (refactor Stage 7 with motion-aware tracker).
5. **`\fad` detection method**: user validated the linear regression approach; added a critical constraint: anchor at `score=1` because `score=0` is never observed (window-bounded, frame zero may be out-of-window or before video start). Also confirmed the no-partial-fade hypothesis, which simplifies the algorithm.
6. **`\move` mergeability criteria**: user validated three axes (R² ≥ 0.95 for trajectory, ≤ 5-frame gap for fragmented chains, Levenshtein < 0.2 for text). For non-linear chains, chose β (merge as static + flag comment) over silent acceptance.
7. **Implementation timing**: user chose ADR-0003 now (design only) over deferring or annexing to ADR-0002.

Mid-session, three follow-up edge cases materially improved the design:

- **`\move` + `\fad` combination**: led to the rule that the score bbox follows the *extrapolated trajectory* of the linear `\move`, not a fixed start-frame quad. Also led to the rule that β-flagged events skip `\fad` detection (undefined trajectory).
- **Slow `\move` intra-event**: led to the realization that Stage 7's IoU > 0.5 criterion masks slow drifts inside a single event. Added `quads_per_frame` to Stage 7 (additive) and the two-level detection (intra + inter) to Stage 8.
- **Asymmetric fades (`\fad(200, 0)` and `\fad(0, 200)`)**: led to the explicit "minimum N points for fit" criterion to avoid spurious tiny fades on hard-cut sides; the user then reframed the minimum in **ms rather than frames** (`MIN_FADE_DURATION_MS = 125`) for multi-framerate portability — propagated to all temporal tunables in §8.

Post-design, the user requested renumbering the new stage from "7.5" to integer "8" and shifting ADR-0002's stages 8–11 to 9–12 (with corresponding workdir directory renames). This is reflected throughout §3, §4, §5, §6, §8.

These follow-ups are reflected in §4 and §9.
