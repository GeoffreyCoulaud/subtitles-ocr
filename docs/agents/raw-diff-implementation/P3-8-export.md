# Agent P3.8 — ExportStage

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stage 11 (renuméroté Stage 12 par ADR-0003 §4.4 qui ajoute `\move` et `\fad`).

**Pré-requis** : P2 terminée.

## Tâche

Implémente `ExportStage.run` dans `pipeline/export.py`.

### Inputs
- `08_animation/animation.json` (motion + fades par event).
- `09_color/colors.json`.
- `11_doc_cleanup/cleaned_final.json`.

### Style synthesis
- **Position class** par centroïde du `quad_median` (règles ADR-0002 §3 Stage 11) :
  - `Bottom` : centroïde dans le tiers inférieur ET horizontalement centré (±20% du centre).
  - `Top` : centroïde dans le tiers supérieur ET horizontalement centré.
  - `Sign` : tout le reste.
- **Color clustering ΔE76 en LAB** :
  - Distance = `max(ΔE(fill_a, fill_b), ΔE(outline_a, outline_b))`.
  - Greedy avec `--color-cluster-threshold` (default 10.0).
  - Canonical color du cluster = mean en LAB, reconverti RGB.
- **Style naming** : `<Position>-<idx>`, idx=0 = dominante par position.
- Events `style_supported=False` → `<Position>-Default` (groupe séparé inheritant white-fill/black-outline).

### Tags inline par event (dans cet ordre : position + rotation + animation + couleurs-via-style)
- `Bottom-*`, `Top-*` : aucun tag inline.
- `Sign-*` : `{\pos(x,y)\frz(angle)}` (omet `\frz` si `|angle| < 2°`).
- `motion.type == "linear"` : ajoute `\move(x1,y1,x2,y2)`.
- `motion.type == "nonlinear_flagged"` : prepend `{!sign: animation non reconstruite!}` au texte.
- `fade_in_ms > 0` OU `fade_out_ms > 0` : ajoute `\fad(fade_in_ms, fade_out_ms)`.
- Couleurs JAMAIS inline (toujours dans le style).

### Timing
- `start_time_ms = round(start * 1000 / fps)`, idem end. Via `timing.frame_to_ms`.

### Multi-line
- `\n` dans `cleaned_text` → `\N` (ASS hard line break).

### `[Script Info]`
```
ScriptType: v4.00+
PlayResX: <fansub_width>
PlayResY: <fansub_height>
WrapStyle: 0
ScaledBorderAndShadow: yes
```

Default style (`<Position>-Default`) : Arial 60, fill white, outline black, outline 2, alignment 2.

Encoding UTF-8 sans BOM. Écriture atomique : `.tmp` + rename.

## Dépendance

```
uv add pysubs2
```

## Tests (`tests/pipeline/test_export.py`)

- 1 event Bottom avec fill blanc + outline noir → style `Bottom-0`, pas de tag inline.
- 1 event Sign → tag `{\pos(...)\frz(...)}`.
- 2 events Bottom avec mêmes couleurs → même style.
- 2 events Bottom avec couleurs distantes (ΔE > 10) → styles `Bottom-0` et `Bottom-1`.
- Event `motion.type=="linear"` → tag `\move(x1,y1,x2,y2)`.
- Event `nonlinear_flagged` → texte commence par `{!sign: animation non reconstruite!}`.
- Event `fade_in_ms=200, fade_out_ms=300` → tag `\fad(200,300)`.
- Event `fade_in_ms=0, fade_out_ms=0` → pas de tag `\fad`.
- Event `style_supported=False` → style `<Position>-Default`.
- Multi-line text → `\N` en sortie.
- L'`.ass` produit est parsable par `pysubs2.load`.
- Atomicité : mock `os.rename` qui lève → `out_path` n'existe pas.

## Périmètre strict

Tu ne touches PAS aux autres stages.
