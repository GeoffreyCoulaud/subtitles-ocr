# ADR-0008: Effective on-screen anchor scoring

Branch: `feat/subtitle-pixels-by-diff-with-raw` (to fork)
Status: Stub — context captured, design pending.
Revises: ADR-0006 §5.8 (Position sub-score).
Motivated by: ADR-0007 §3.4 (synthetic `\pos\fad` heuristic), §7 (open issues).

## 1. Context

ADR-0006's `position_pair_score` compares the inline `\pos` / `\move` of each
event. When neither side has an inline tag, the comparison is skipped (the
pair returns `None`). When one side has it and the other does not, the pair
scores `0.0` regardless of where the line actually renders on screen.

In ASS, the on-screen anchor of a line is determined by, in priority order:

1. Inline `\pos(x,y)` or `\move(...)` — explicit override.
2. Inline `\an<n>` — overrides only the alignment, not the margins.
3. The line's `Style`: `Alignment` + `MarginL` / `MarginR` / `MarginV`,
   combined with `[Script Info] PlayResX/Y` to produce a numeric anchor.

A fansub that positions all dialogue through a `Default` style (no inline
`\pos`) renders to a specific point on screen — bottom-centre with
`MarginV = 17` for KenIchi. The current scorer cannot see that point, so a
pipeline that places its output at the same point through its own style
gets a `None` position score on those events and a `0.0` score on every
event where ref happens to use an explicit `\pos`. The asymmetry forced
ADR-0007 to synthesise inline tags from text patterns
(`ALL-CAPS in mid-band` → `\pos(...)\fad(350,0)`), which is over-fitted to
KenIchi's fansub conventions and will not generalise.

The user-specified target for the revision is:

- The bulk of position points should come from **effective on-screen
  anchor** equality, derived from whichever combination of `\pos` / `\an` /
  style attributes / margins / `PlayResX/Y` actually produces the rendered
  point.
- An **"exact" axis** comparable to `text_exact` should reward clean
  reconstructions that use the same source of position as the ref (e.g. if
  ref uses an explicit `\pos`, output should also use one). Without this
  guard, anchoring every event to bottom-centre through the default style
  would maximise the effective-anchor sub-score even on events that are
  semantically top-positioned signs.
- For non-dialogue text (signs, annotations), effective anchor alone is
  not enough: the comparison must also reward matching the **kind** of
  positioning signal (explicit `\pos`, in-frame motion via `\move`,
  rotation via `\frz`), independently of the numeric position.

## 2. Decision

To be designed. The following directions are on the table and not yet
chosen:

### 2.1. Effective anchor as the primary numeric component

Compute, for each parsed event, an effective anchor:

```python
def effective_anchor(event, subs: SSAFile) -> tuple[float, float]:
    parsed = parse_event_text(event.text)
    if parsed.pos is not None:
        return parsed.pos
    if parsed.move is not None:
        # already handled per-endpoint in ADR-0006 §5.8 (\move case)
        ...
    style = subs.styles.get(event.style) or _ssa_default_style()
    play_res = _play_res(subs)
    alignment = parsed.alignment or _alignment_of(style)
    return _anchor_from_style(
        play_res,
        alignment,
        style.marginL,
        style.marginR,
        style.marginV,
    )
```

The pair score becomes a distance between effective anchors, always
defined (no `None` case for matched pairs).

### 2.2. "Exact" sub-score that rewards matching position source

A discrete component, on top of the numeric effective-anchor comparison,
that scores 1.0 when ref and output draw the anchor from the same kind of
source:

- both inline `\pos`
- both inline `\move`
- both pure-style (no inline)
- otherwise: 0.0

Weight to be chosen so that "match the numeric anchor through the style"
gets the majority of the position points but cannot equal the score of
"match the numeric anchor AND the source kind". This is the guard against
anchoring everything in the default style.

### 2.3. "Intent" component for non-dialogue overlays

When ref uses any inline `\pos` / `\move` / `\frz`, that's a fansubber
signalling "this is a sign / annotation / overlay, not dialogue". The
pipeline should be incentivised to detect that and emit corresponding
overrides on its side, even when the numeric anchor would coincide with a
style default. Possible shapes:

- A "has-override" binary component, scored the same way as italic / bold
  in the current styling.
- A separate "intent" sub-score with its own weight, that fires only when
  at least one side has an override.

### 2.4. Mechanical changes implied for the pipeline

Whatever the final scoring shape, the pipeline will need to:

- Emit ASS styles whose `Alignment` and `MarginV` reproduce ref's
  bottom-centre default. The current Default-style synthesis is close but
  uses `MarginV = 10` instead of ref's 17; we will need to either match a
  user-known default or learn one from the OCR's typical bottom-line y.
- Stop emitting synthetic `\pos\fad` on ALL-CAPS / time-window heuristics
  (ADR-0007 §3.4). Once the scorer no longer punishes "style-derived
  anchor vs ref-`\pos`-same-numeric-anchor" pairs, the heuristic loses
  its purpose and only contributes false positives.
- Provide a real signal for "this OCR'd event looks like an overlay"
  (rotated quad, mid-screen-vertical, short-text, etc.) so the pipeline
  can correctly emit `\pos` on those events. This is no longer
  KenIchi-shaped because the scorer only checks "did you emit an
  override at all" (intent) plus "is the numeric anchor close", not
  "did you guess the fansubber's fade timing".

## 3. Open questions

- How are the weights between numeric-anchor, exact-source, and intent
  partitioned within the position pillar? (ADR-0006 currently allocates
  weight 5 to position; this revision likely needs to redistribute or
  expand the pillar.)
- What is the unit / normalisation of the numeric anchor distance? ADR-0006
  uses `distance / diagonal` with a `d_max = 0.10` cliff. Does that
  threshold still make sense once style-derived anchors enter the picture
  (which can have rounding from `PlayResX/Y` / margins)?
- Does the "exact-source" component apply per-pair (1.0 / 0.0) or as a
  proportion across the whole document? Per-pair is simpler and stays
  aligned with the other ADR-0006 components.
- How does the revised scorer interact with `\fad` and `\frz`? They have no
  style fallback in ASS — inline-only. The current ADR-0006 logic already
  returns `None` when neither side has them, which is the desired
  behaviour. The revision is probably orthogonal to those two.
- Do we keep ADR-0007's wrap-merge (export-side) once anchor scoring is
  fixed? It is independent of the position revision and stays useful for
  text_plain / text_exact / precision regardless. (Likely yes.)

## 4. Migration plan

1. Land this ADR's design (concrete formulae, weights, anchor derivation).
2. Implement and unit-test `effective_anchor` and the new pair-score in
   `evaluation/position.py` (plus any new sub-score module).
3. Regenerate KenIchi's scoring report. Expected: position component
   improves materially without the ADR-0007 heuristics emitting any
   inline tags.
4. Retire ADR-0007 §3.4 (the synthetic `\pos\fad` heuristic), preserving
   only the OCR / wrap-merge / filter / event-cleanup work. Update
   ADR-0007 status to "partially superseded".
5. Land final ADR-0008 status: "Implemented".

## 5. Out of scope

- `\frz` and `\fad` scoring revision (kept inline-only as today).
- New input modalities (the scorer still consumes `.ass` files, not
  rendered video).
- A learned (ML) anchor / style predictor — explicitly deferred; the
  intent is to make the scorer right, not the pipeline cleverer.
