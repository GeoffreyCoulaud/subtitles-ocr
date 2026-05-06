# ADR-0001: Diff-based Hardsub Extraction Pipeline

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: Scoped, not yet implemented.
Origin: Pick-my-brain scoping conversation between user and Claude.

This ADR captures both the final design and the reasoning behind each fork, so the work can be resumed in another session without losing context.

---

## 1. Problem

The existing pipeline on `main` extracts hardcoded subtitles from a single video using a VLM (Qwen 3 VL 4B) that prefilters frames, analyzes them, and groups them into events. It works but is slow, accuracy-bounded by the VLM, and weak at recovering position / rotation / color.

We want to explore a different approach when a clean "raw" companion video (no subs) is available:

1. Diff each fansub frame against its raw counterpart.
2. The locus of significant difference is the subtitle region.
3. Mask the rest of the frame, run a real OCR model on the result.
4. Recover position, rotation, fill color, outline color from the OCR quad + the masked glyph pixels.
5. LLM-clean the output text for OCR errors and continuity.

This branch is an exploration. It is not merged to `main` until proven against ground truth.

## 2. Inputs and test pair

The pipeline takes two videos per run: a hardsubbed video and a raw companion. Different masters are expected — the raw is usually a higher-quality release (Bluray) re-graded vs the fansub source. Exact-master matching is **not** assumed.

Concrete test pair (committed location: `~/Projects/subtitles-ocr-kenichi-data/`):

| Aspect | Fansub `.avi` | Raw Bluray `.mkv` |
|---|---|---|
| Codec | MPEG-4 | HEVC |
| Resolution | 640×480 | 1440×1080 |
| Bit depth | 8-bit | 10-bit |
| DAR | 4:3 | 4:3 |
| Framerate | 23.976fps | 23.976fps |
| Duration | 1499.83s (35960 frames) | 1475.20s |
| Subs (in file) | hardcoded | PGS soft-sub track present |

~24.6s of cumulative duration drift between the two — non-trivial editorial differences.

Subtitle language: French. Pipeline supports any non-Japanese script (Latin scripts at minimum); selected per run via `--language`.

## 3. Subtitle styling — what's in scope

In scope:
- Signs / title translations at any position and orientation
- Multi-position dialogue (top / bottom / mid)
- Outlined fonts, sometimes colored
- Standard two-color typography (single fill + single outline)

Out of scope (confirmed during scoping):
- OP/ED karaoke — pre-cut by user before pipeline runs
- In-episode karaoke / insert songs — don't occur in target content
- Gradient fills, shadow-fill-outline three-pass typesetting — pipeline flags as "unsupported style", skips color, falls back to default
- Animation tags (`\fad`, `\move`)
- Outline thickness, font weight, italic, font family

## 4. Pipeline stages

1. **Spatial conform** — scale/letterbox raw to fansub resolution.
2. **Adaptive phash alignment** — per-frame perceptual-hash matching against raw, with a search window that tracks a running offset, widens on miss, contracts on resumed matches.
3. **Diff** — local normalization + edge-based diff, to suppress regrade noise (global brightness/gamma/saturation shifts) while keeping the hard edges that define subtitle glyphs.
4. **Smoothing → mask** — cluster diff pixels into mask regions.
5. **Compose** — apply mask to fansub frame (mask in = keep fansub pixel, mask out = black).
6. **OCR** — PaddleOCR PP-OCRv4 **server** variant: DBNet detector → angle classifier → recognizer. Recognizer language picked by `--language` (default `latin`).
7. **Color extraction** — per OCR quad: in-quad Otsu binarization → glyph mask → adaptive erosion sized from stroke-width via distance transform → interior pixels = fill, edge pixels = outline. Flag "unsupported style" if the two clusters cannot be cleanly separated.
8. **Group** — collapse consecutive frames with matching OCR text + stable mask geometry into `SubtitleEvent` cues.
9. **Per-event LLM cleanup** — small text-only model fixes obvious OCR confusables (`rn`/`m`, `I`/`l`/`1`, accent recovery).
10. **Whole-document LLM cleanup** — larger text-only model with optional `--synopsis` context for global corrections (consistent character names, place names, story-level continuity).
11. **Export** — write `.ass` with `\pos`, `\frz` (rotation), `\c` (fill), `\3c` (outline).

## 5. Alignment failure policy

- Per-frame miss (no good phash match within window): WARNING log with timestamp and search-window bounds. Frame skipped, contributes nothing to subtitle extraction. Summary count at stage end.
- ≥10s contiguous unaligned span: hard stop. Pipeline exits with error. Rationale: alignment is load-bearing for the entire approach; sustained failure means the pair is not workable.

## 6. Output

- **Final artifact:** `.ass` file.
- **Intermediates:** per-stage JSON in `--workdir`. Slow stages (alignment, OCR) write per-N-frame JSONL chunks (N ≈ 500) for incremental crash recovery. Fast stages (group, cleanup, export) write atomic JSON.
- **Resume:** implicit when intermediates are valid. User deletes workdir, or points `--workdir` elsewhere, for a fresh start. No `--resume` flag.
- **Debug images** (per-frame masks + composed images) only emitted under `--debug-images`.
- **Recovered metadata per cue:** text, timing, position, rotation, fill color, outline color.

## 7. Validation strategy

Full-episode hand-annotated Aegisub `.ass` for KenIchi e01, produced by the user (~4–8h work, accelerated by pre-populating from the Bluray's PGS subtitle track using `pgsrip` or SubtitleEdit). Quantitative comparison of pipeline output vs ground truth on text + timing + position.

The ground-truth `.ass` is also a reusable test asset for the project beyond this branch.

"Proven enough to consider merging" is judged empirically against ground-truth match quality. No fixed threshold pre-committed.

## 8. Compute target

- CPU fallback required (must run, may be slow).
- GPU paths: Apple Silicon M3 Pro (MPS, 18GB shared memory) and AMD ROCm RX 7700XT (12GB VRAM).
- End-to-end budget: **< 2h for a 24-min episode** (~5× realtime, ~200ms/frame at every-frame processing).
- Frame strategy: every frame processed (no sampling). Premature optimization explicitly avoided.

## 9. CLI

```
subtitles-ocr \
  --hardsub <fansub.avi> \
  --raw <bluray.mkv> \
  --out <output.ass> \
  [--language latin] \
  [--synopsis path/to/synopsis.txt] \
  [--workdir path/to/intermediates] \
  [--debug-images]
```

Single mode on this branch — no flag-based switching to a legacy pipeline.

## 10. Code housekeeping on this branch

Delete now:
- `src/subtitles_ocr/pipeline/prefilter.py` and corresponding tests
- `src/subtitles_ocr/vlm/` (entire directory)
- `src/subtitles_ocr/pipeline/filter.py` (current edge-similarity grouping — replaced by the new grouping logic)
- Any other dead code that only served the VLM pipeline

Git history and `main` preserve everything. To revert a deletion: `git checkout main -- <path>`.

## 11. Out-of-domain inputs

No automatic detection or gating for inputs that violate scope (live-action footage, Japanese-script subtitles, non-anime). Trust the user, document scope in README. Soft-warning approaches were considered and rejected — they produce false positives on quiet/sparse-text episodes.

## 12. Tests / CI

- TDD per CLAUDE.md ("implementer subagents must invoke `superpowers:test-driven-development` before writing any production code").
- Mock PaddleOCR and ffmpeg (matches existing convention of mocking Ollama).
- Test fixtures generated in-test via numpy — no committed binary fixtures.
- No real-content end-to-end integration test in CI; the Aegisub-ground-truth validation pass against the KenIchi episode is the real end-to-end check.

## 13. Logging / progress

- tqdm progress bars per stage (already used in current codebase).
- Stage-transition INFO logs with timestamps and counts (e.g., `alignment complete: 35,832/35,960 frames matched, 128 misses`).
- Per-frame alignment misses → WARNING in log file, not stdout (would spam the terminal at every-frame scale). Summary count at stage end.
- `--debug` flag enables DEBUG-level traces. No `--verbose`.
- No emoji, no color noise.

## 14. Performance regression policy

Trust the budget + use existing log timestamps as ad-hoc record. No per-commit benchmark infrastructure. Validation against ground truth is the real success bar; speed is a secondary constraint with a hard 2h ceiling.

## 15. Failure recovery (Ctrl+C / crash / OOM)

- Per-N-frame JSONL append in slow stages (alignment, OCR) → resume seeks past the last persisted line.
- Atomic JSON in fast stages (group, cleanup, export) → mid-stage interrupt redoes the whole stage.
- SIGINT handler: finish the current chunk write, exit cleanly, no half-written files. Print a hint reminding the user that the workdir is recoverable.

## 16. Decision log (key forks and rationale)

| Fork | Decision | Why |
|---|---|---|
| Sample frames vs every frame | Every frame | Premature optimization avoided; cleanest temporal precision; brief signs not missed. |
| Pure VLM vs PaddleOCR vs hybrid | Pure stock PaddleOCR (PP-OCRv4 server) | Want oriented quads + good text; VLMs poor at precise positions; ParSeq considered as escape hatch but not phase 1. Server variant chosen over mobile because target machines are decent. |
| K-means color vs glyph-mask color | Glyph-mask via Otsu + adaptive erosion | Glyph-mask is materially better at fill/outline separation; the "more code" cost is small (~50 lines OpenCV). The simpler k-means was a hedge; pushed back on. |
| Vision LLM vs text-only LLM for cleanup | Text-only, two-stage | No need to re-examine glyphs once OCR is good; smaller / faster models possible. Two stages: per-event small fixes, final-pass larger with optional synopsis. |
| Synopsis source | User-provided CLI flag, optional | Avoids external API dependency, identifier-mapping, network requirement. Best-effort cleanup without it. |
| Alignment method | Adaptive phash search window | Robust > clever; tracks running offset rather than same-timecode. Frame-based fits the every-frame strategy. |
| Alignment failure | Per-frame log+skip; ≥10s contiguous = hard stop | Alignment is load-bearing for the entire approach; sustained failure means the pair isn't workable. Single missing frames shouldn't kill the run. |
| Regrade compensation | Local normalization + edge-based diff | Suppresses smooth global shifts; keeps hard subtitle edges. |
| Pipeline integration | Branch-only experiment, delete VLM code on this branch | Cleanest mental model; `main` and git history preserve old pipeline. |
| Validation | Full-episode Aegisub `.ass` ground truth | User willing to invest the time; produces a high-quality reusable test asset. |
| Multi-show vs single-show | Multi-show architecture, validated only on KenIchi | Anime + non-Japanese subs; no fansub-team-specific or aspect-ratio-specific hardcoding. |
| Resume granularity | Hybrid: small-checkpoints default + `--debug-images` flag | Disk-cheap default, debugging available on demand. |
| Test fixtures | numpy-generated in-test | Clean repo, deterministic, fast. |
| Performance regression policy | Trust the budget + log timestamps | No infrastructure for an unproven branch; revisit if perf bites. |

## 17. Still empirical (defer to prototyping)

These are tuning knobs whose values can only be set by running on real content and measuring:

- phash window initial size, growth rate on miss, shrink rate on resumed matches
- Mask post-processing kernels — Gaussian sigma, minimum cluster area, dilation/erosion
- Otsu and adaptive-erosion thresholds for color extraction
- Per-N-frame `N` for the slow-stage JSONL chunk size (initial guess: 500)
- "Stable mask geometry" tolerance for grouping consecutive frames into one cue
- Choice of specific small text-only LLM (per-event cleanup) and larger text-only LLM (final pass) — pick by testing against ground-truth quality

## 18. Conversation provenance

This ADR was produced from a structured pick-my-brain scoping conversation. Domains covered, in roughly the order they were explored:

1. Whether a concrete test pair existed (yes — KenIchi e01)
2. Same-master vs different-master pair (different — fansub vs Bluray re-grade)
3. Resolution / encoding / framerate / duration of the pair (probed via ffprobe; 25s drift discovered)
4. Subtitle styling complexity (signs, multi-position, outlined; OP/ED out)
5. Alignment strategy (phash + adaptive search window)
6. OCR model choice (PaddleOCR over VLMs; server variant)
7. Color extraction methodology (glyph-mask via Otsu + adaptive erosion)
8. LLM cleanup design (text-only, two-stage)
9. Pipeline integration (branch-only experiment)
10. Validation (full-episode Aegisub `.ass` ground truth)
11. Frame strategy (every frame)
12. Output format (`.ass` final, JSON intermediates with resume)
13. Compute envelope (< 2h, M3 Pro / RX 7700XT, CPU fallback)
14. Synopsis source (user-provided CLI flag, optional)
15. Multi-show scope (multi-show architecture, validated only on KenIchi)
16. In-episode karaoke (out of scope — doesn't occur)
17. Edge cases (cut-straddling, sign-overlap — both no-special-handling)
18. Failure recovery (per-N-frame JSONL in slow stages, SIGINT graceful exit)
19. Test/CI strategy (mock PaddleOCR + ffmpeg, in-test fixtures)
20. Existing VLM code (delete on this branch)
21. Performance regression policy (trust budget + log timestamps)

Several places during scoping the agent hedged with "ship a baseline first, escalate if needed" recommendations — for the recognizer (PaddleOCR-recognizer vs ParSeq) and for color extraction (k-means vs glyph mask). The user pushed back on these hedges and the agent committed to the better option directly. Future scoping in this project should default to surfacing an informed recommendation rather than enumerating options neutrally.
