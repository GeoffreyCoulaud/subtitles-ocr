# ADR-0008: Effective on-screen anchor scoring

Branch: `feat/subtitle-pixels-by-diff-with-raw` (to fork)
Status: Designed — implementation pending.
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

The revision replaces the existing single `position` sub-score with a
three-axis decomposition inside the position pillar, chosen so that the
score reflects what a viewer actually sees (`(x, y)` + alignment direction)
and rewards missing-an-override harder than emitting-one-too-many.

## 2. Decision

The `position` pillar is rebuilt around three independent sub-scores:
`position`, `anchor`, `intent`. The pillar total weight rises from 5 to 10,
partitioned 6 / 3 / 1.

### 2.1. Effective position model

The on-screen position of an event is fully described by:

- `(x, y)` — the pixel coordinates of the alignment point.
- `alignment ∈ {1..9}` — which corner / edge of the rendered text box sits
  at `(x, y)`. Same `(x, y)` with different `alignment` produces visually
  different renders (the text extends in different directions from the
  anchor), so the two dimensions are treated as **independent axes**.

Both dimensions are derived per-event from a single resolver:

```python
def effective_anchor(event, subs: SSAFile) -> EffectiveAnchor:
    parsed = parse_event_text(event.text)
    style = subs.styles.get(event.style) or _ssa_default_style()
    play_res = _play_res(subs)
    alignment = parsed.alignment or _alignment_of(style)

    if parsed.pos is not None:
        return EffectiveAnchor(points=[parsed.pos], alignment=alignment, source="pos")
    if parsed.move is not None:
        return EffectiveAnchor(
            points=[parsed.move.start, parsed.move.end],
            alignment=alignment,
            source="move",
        )
    point = _anchor_from_style(
        play_res, alignment, style.marginL, style.marginR, style.marginV,
    )
    return EffectiveAnchor(points=[point], alignment=alignment, source="style")
```

`source` is recorded for the `intent` axis. The numeric formula for
`_anchor_from_style` follows the standard ASS rules (V-margin from the
bottom for alignments 1-3, from the top for 7-9, centred for 4-6;
H-margin symmetric on the left / right depending on alignment). For
implementation, the resolver lives in `evaluation/position.py` alongside
the pair-score functions.

### 2.2. Sub-score: `position` (weight 6)

Continuous, applies to **every** paired event. Compares the `(x, y)`
points of the two effective anchors.

Static event is treated as a degenerate `\move` whose start equals end.
For a pair, score per endpoint with the existing ADR-0006 cliff and
average:

```python
def position_pair_score(ref: EffectiveAnchor, out: EffectiveAnchor) -> float:
    ref_pts = ref.points if len(ref.points) == 2 else [ref.points[0], ref.points[0]]
    out_pts = out.points if len(out.points) == 2 else [out.points[0], out.points[0]]
    d_max = play_res_diagonal(...) * 0.10
    return mean(
        max(0.0, 1.0 - euclid(rp, op) / d_max)
        for rp, op in zip(ref_pts, out_pts, strict=True)
    )
```

Consequence: a static event mis-matched against a long `\move` is
penalised proportionally to the displacement (a small unmatched `\move`
costs little, a big unmatched `\move` costs a lot). The cliff and the
`d_max = diagonal × 0.10` normalisation are inherited from ADR-0006 §5.8.

### 2.3. Sub-score: `anchor` (weight 3)

Binary, applies to **every** paired event. Compares the resolved
`alignment` (1..9) of the two effective anchors:

```python
def anchor_pair_score(ref: EffectiveAnchor, out: EffectiveAnchor) -> float:
    return 1.0 if ref.alignment == out.alignment else 0.0
```

Source-agnostic: an `\an2` inline on ref and `Style.Alignment = 2` on
output both resolve to `alignment = 2` and match. A mismatch on this axis
does **not** zero out the `position` axis — the two are independent and
each carries its own weight.

### 2.4. Sub-score: `intent` (weight 1)

Binary, asymmetric. Fires **only when ref has an inline `\pos` or `\move`**
(i.e. ref's `source ∈ {"pos", "move"}`). On those pairs:

```python
def intent_pair_score(ref: EffectiveAnchor, out: EffectiveAnchor) -> float | None:
    if ref.source == "style":
        return None  # not applicable to this pair
    return 1.0 if out.source in {"pos", "move"} else 0.0
```

Pair score is `None` when ref is pure-style (no penalty, no contribution).
Adding a superfluous `\pos` on output where ref is pure-style is **not**
punished by `intent` (it is N/A on those pairs). This is the desired
asymmetry: losing an override (ref had information, output dropped it) is
penalised; adding an override (output adds a verbose-but-correct override
on dialogue) is not.

A symmetric "exact source" axis was considered (both `\pos`, both
`\move`, both style → 1.0, else 0.0) and rejected: with `anchor` already
split into its own axis, that symmetric scoring would punish the "add a
superfluous `\pos`" case as hard as the "drop ref's `\pos`" case,
contradicting the asymmetry above.

### 2.5. Aggregation

Each axis aggregates per the ADR-0006 §6 pattern:

- `position`, `anchor` — mean across **all** paired events. Always
  defined (resolver returns a value for every event).
- `intent` — mean across paired events where the per-pair score is not
  `None`. If **no** ref event has a `\pos` / `\move` in the whole
  document, `intent` is `None` for the document and its weight (1) is
  redistributed into the global denominator (standard ADR-0006 rescaling).

### 2.6. Weights

| Axis      | Weight | Fires on              |
|-----------|--------|------------------------|
| `position`| 6      | Every paired event     |
| `anchor`  | 3      | Every paired event     |
| `intent`  | 1      | Pairs where ref has `\pos`/`\move` |

Pillar total: 10 (was 5 in ADR-0006). Global denominator rises from 95 to
100 once `position` (old, weight 5) is replaced by the three new axes
(combined weight 10). No other ADR-0006 weight changes.

### 2.7. Mechanical changes implied for the pipeline

The pillar redesign makes the following pipeline-side changes possible
and desirable; they are out of scope for **this** ADR (the scoring change
lands first, then the pipeline catches up):

- Retire ADR-0007 §3.4 (synthetic `\pos\fad` from text patterns). With
  `anchor` + `position` no longer punishing "pure-style vs ref-`\pos`-at-
  same-anchor" pairs, the heuristic loses its only purpose and becomes a
  source of false positives.
- Tune the `Default` style synthesis (currently `MarginV = 10`; KenIchi's
  ref uses 17). Either match a known default per source or learn from
  the OCR's typical bottom-line y. This is what unlocks the bulk of the
  `position` score for dialogue.
- Provide a real overlay signal (rotated quad, mid-screen y, short text,
  etc.) so the pipeline can correctly emit `\pos` on signs. With the new
  scoring, the pipeline only needs to (a) emit `\pos` at all (intent) and
  (b) land near ref's `(x, y)` and `\an` (position + anchor) — no need
  to guess fade timings.

## 3. Resolved questions

The stub had open questions on weights, normalisation, per-pair vs
document, and `\fad`/`\frz` interaction. Resolutions:

- **Weights**: pillar 10 total, 6 / 3 / 1 partition (§2.6).
- **Numeric normalisation**: keep ADR-0006's `d / diagonal` with
  `d_max = 0.10` cliff. Rounding from `PlayResX/Y` / margins is small
  relative to a 10% diagonal cliff and not worth a separate threshold.
- **Per-pair vs document**: per-pair, matching all other ADR-0006
  components. `intent` aggregates only over the subset where ref has an
  override (§2.5).
- **`\fad` / `\frz`**: orthogonal to this revision. Their scoring stays
  inline-only as today (ADR-0006 §5.9 / §5.10). They do **not** trigger
  the `intent` axis — only `\pos` and `\move` do (§2.4).
- **Wrap-merge (ADR-0007 §3.5)**: kept. Independent of position scoring,
  still useful for `text_plain` / `text_exact` / `precision`.

## 4. Migration plan

1. Implement `effective_anchor`, `position_pair_score`,
   `anchor_pair_score`, `intent_pair_score` in
   `src/subtitles_ocr/evaluation/position.py` (TDD: failing tests
   first, then production code).
2. Wire the three new sub-scores into `Weights`, `score.py`,
   `report.py` (replacing the single `position` weight 5 with the three
   axes 6 / 3 / 1).
3. Update existing position tests (`tests/evaluation/test_position.py`,
   `test_score.py`, `test_report.py`) to assert the new shape.
4. Regenerate KenIchi's scoring report. Expected: position pillar
   improves materially through the style-derived anchor matching ref's
   pure-style events, with no pipeline-side `\pos\fad` heuristics needed.
5. Retire ADR-0007 §3.4 (synthetic `\pos\fad`) in a follow-up commit;
   update ADR-0007 status to "partially superseded".
6. Tune `Default` style `MarginV` (and any other style-margin defaults)
   in a follow-up commit, validated against the new score.
7. Land final ADR-0008 status: "Implemented".

## 5. Out of scope

- `\frz` and `\fad` scoring revision (kept inline-only as today, do not
  trigger `intent`).
- New input modalities (the scorer still consumes `.ass` files, not
  rendered video).
- A learned (ML) anchor / style predictor — explicitly deferred; the
  intent is to make the scorer right, not the pipeline cleverer.
- Pipeline-side overlay detection logic (signs / annotations). The
  scoring change unlocks it but does not specify it.
