# ADR-0009: Data-driven font size from OCR quad height

Branch: `feat/subtitle-pixels-by-diff-with-raw`
Status: Designed — implementation pending.
Revises: ADR-0002 §3 Stage 11 (Export — Style synthesis), ADR-0006 §5.7
(via ADR-0008's source-agnostic styling resolution).

## 1. Context

The export stage currently emits every synthesised Style with a constant
``fontsize = default_font_size`` (60 in `ExportConfig`). KenIchi's ref
uses very different sizes per Style:

| Style                  | fontsize |
|------------------------|----------|
| Default                | 34       |
| Default Top            | 34       |
| Default Top - Smaller  | 28       |
| Inner Monologue        | 34       |
| Translator's Note      | 30       |
| Forced                 | 20       |
| Forced discreet        | 33       |
| Episode Title          | 26       |
| Book Title             | 19       |
| Book Title - Big       | 28       |
| Sign                   | 64       |
| Sign - Black           | 50       |

After ADR-0008 extended styling to source-agnostic resolution, the
`font_size` component compares the resolved fontsize via the min/max
ratio. With ref Default = 34 and output Default = 60, every dialogue
pair contributes a `34/60 ≈ 0.567` on the font_size axis, dragging
`styling` down across 400 pairs.

The OCR pipeline already produces, for every event, a quad bounding the
detected glyphs. The quad height is roughly proportional to the rendered
font size (cap height + ascender + descender + outline/aa padding). We
can derive a per-event font size from that quad height and aggregate
across each synthesised Style.

## 2. Decision

### 2.1. Pipeline: derive font size from quad height per Style

For each `_PreparedEvent`, compute a candidate font size from the
event's `quad_median` height. Aggregate within each synthesised Style
group (Bottom-0, Sign-Default, etc.) to a single robust statistic — the
**median** of the contributing candidates — and set that as the Style's
`fontsize`. No inline `\fs` overrides on individual events; the Style
holds the size, ADR-0008's resolver picks it up.

Candidate font size derivation:

```python
def fontsize_from_quad(quad: list[tuple[int, int]]) -> float:
    ys = [y for _, y in quad]
    height = max(ys) - min(ys)
    return height * _OCR_QUAD_TO_FONTSIZE
```

`_OCR_QUAD_TO_FONTSIZE` is a single empirical constant. It models
"quad height vs ASS fontsize" for the OCR + renderer pair we run with.
Calibration is data-driven — measured by regressing observed quad
heights against known ref fontsizes on the validation corpus — and
**not** per-fansub. The same constant applies to every video; what
varies per video is the per-event quad heights, which the median
aggregation naturally tracks.

Default starting value: **0.75** (i.e. `fontsize ≈ height × 0.75`). It
will be tuned during implementation against the KenIchi calibration
sample; the constant is a *one-line* knob.

### 2.2. Pipeline: per-Style aggregation

`_synthesize_styles` returns a list of `style_assignments`. The same
function already groups events by `(position, color cluster)`. For each
group, compute `fontsize = median(fontsize_from_quad(ev.quad_median))`
across the events assigned to that style. Plug that into `_make_style`
and `_make_default_style`.

Bonus: each *position class* gets the size it deserves. Sign clusters
end up with the larger fontsize their visually-larger glyphs imply;
Bottom dialogue clusters end up with the smaller fontsize ref typically
uses.

### 2.3. Scoring: keep the existing min/max ratio

ADR-0008's source-agnostic styling already resolves `font_size` from
inline `\fs` *or* `Style.fontsize`. The current pair score
``min(a, b) / max(a, b)`` is already graceful, source-agnostic, and
robust. **No scoring change** is required for this ADR. A 5%
calibration error gives a `0.95` font_size component on the pair, which
is well within the noise floor of the other component contributions.

The pair score retains the same shape as ADR-0006 §5.7 / ADR-0008:

```python
def _ratio(a: float, b: float) -> float:
    if max(a, b) == 0.0:
        return 1.0
    return min(a, b) / max(a, b)
```

No tolerance band, no thresholding. The ratio's gracefulness *is* the
tolerance.

### 2.4. PlayResY scale invariance — deferred

In principle, `fontsize` is a script-space measurement scaled at render
time by ``screen_height / PlayResY``. Two scripts with the same visual
text size can disagree numerically if they use different `PlayResY`.
This matters only when output and ref `PlayResY` differ.

In the KenIchi case both sides use `PlayResY = 1080`, so direct
numerical comparison is correct. Cross-script scale invariance is a
**deferred** scoring refinement (out of scope of this ADR); when needed
it would normalise both sides by `fontsize / PlayResY` before the
ratio.

## 3. Open questions

- **Calibration constant** (`_OCR_QUAD_TO_FONTSIZE`): starting at 0.75
  from rough cap-height arithmetic. The acceptance criterion is the
  KenIchi `font_size` component median across Default-style dialogue
  pairs ≥ 0.95 after the change. Re-tune in the same commit if the
  starting value misses.
- **Per-style median vs per-event inline `\fs`**: per-style median is
  the chosen shape (§2.2). Inline `\fs` is rejected because it bloats
  the .ass and confers no scoring benefit (the styling resolver reads
  whichever source is present; both yield the same comparison).
- **Outliers in a cluster**: median absorbs outliers naturally. No
  explicit trimming required.
- **Empty clusters**: fall back to `config.default_font_size` (the
  current constant) when a synthesised style has zero contributing
  events — defensive, should not happen in practice because
  `_synthesize_styles` builds styles from the events themselves.

## 4. Migration plan

1. Add `fontsize_from_quad` helper in `pipeline/export.py` with the
   `_OCR_QUAD_TO_FONTSIZE` constant. TDD: a focused unit test that
   asserts `fontsize_from_quad([(0,0), (100,0), (100,40), (0,40)]) ==
   30.0` (height 40 × 0.75).
2. Plumb the helper through `_synthesize_styles` so each style group
   has a `fontsize` argument; compute it as the per-group median of
   candidates. TDD: a focused integration test that builds three events
   with different quad heights and asserts the synthesised Style's
   `fontsize` matches the expected median.
3. Update `_make_style` / `_make_default_style` to accept the computed
   fontsize and emit it; remove the `default_font_size` hard-coding
   from those paths (the config field stays as the *fallback* when no
   events contribute).
4. Re-export KenIchi, measure the new `styling` / `font_size`
   contribution. Expected lift: `styling` ≈ 0.90 → ≈ 0.95 on the full
   episode.
5. Re-tune `_OCR_QUAD_TO_FONTSIZE` once if §3 acceptance criterion
   isn't met on the first pass.
6. Land ADR-0009 status: "Implemented".

## 5. Out of scope

- Cross-script `PlayResY` scale invariance in the `font_size` axis
  (§2.4) — deferred until we see a real case where it bites.
- Font family detection (e.g. Arial vs Noto Sans).
- Per-glyph or per-line font size variation within an event.
- Bold / italic weight detection from glyph metrics.
