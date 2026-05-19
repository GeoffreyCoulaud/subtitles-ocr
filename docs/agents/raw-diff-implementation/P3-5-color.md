# Agent P3.5 — ColorStage

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stage 8 (renuméroté Stage 9 par ADR-0003 §4.3, qui exclut les frames de fade).

**Note MVP** : `AnimationStage` stub met `fade_in_ms=0` et `fade_out_ms=0` partout, mais TON code DOIT déjà gérer l'exclusion (et l'edge case "< 3 frames restantes → `style_supported=False`").

**Pré-requis** : P2 terminée.

## Tâche

Implémente `ColorStage.run` dans `pipeline/color.py`.

- Lit `08_animation/animation.json` (passe-plat MVP).
- Pour chaque event :
  - Pour chaque member frame : crop fansub via `cv2.getPerspectiveTransform` + `cv2.warpPerspective` vers `(W_canon, H_canon) = median(member widths/heights)`.
  - Crop padding : +10% de la diagonale du quad dans chaque direction (capture l'outline qui spille au-delà du quad DBNet).
  - Exclut les fade-frames du début/fin selon `fade_in_ms`/`fade_out_ms` (conversion via `timing.ms_to_frame`).
  - Si remaining frames < 3 → `style_supported=False`, `fill_color=None`, `outline_color=None`.
  - Stack en `(N, H, W, 3)`, médiane temporelle pixel-wise.
  - Otsu global sur l'image canonique → glyph mask binaire.
  - Distance transform sur le glyph mask. `stroke_width = p95` des valeurs sur les pixels du glyph.
  - Erode mask par kernel circulaire radius `floor(0.4 * stroke_width_p95)`.
  - Interior pixels (mask érodé) = mode HSV 16-bin → `fill_color` (re-convertie RGB).
  - Outline pixels (mask original moins mask érodé) = mode HSV 16-bin → `outline_color`.
  - `style_supported = False` si :
    - Interior pixel pool < `config.interior_min_pixels` (default 100).
    - Variance H interior > `config.interior_hue_var_max`.
    - Variance H outline > `config.outline_hue_var_max`.
    - Ratio interior/outline aberrant (`< config.pool_ratio_min` ou `> config.pool_ratio_max`).
- Écrit `09_color/colors.json` (atomique) + sidecar.

## Tests (`tests/pipeline/test_color.py`)

Frames numpy synthétiques (créer un workdir avec fausses vidéos lisibles via OpenCV, ou mocker la lecture vidéo).

- Event avec glyph blanc + outline noir sur fond noir → fill blanc, outline noir.
- Event avec interior gradient (variance H haute) → `style_supported=False`.
- Event avec `fade_in_ms=200` à 24fps → 4 premières frames exclues du stack.
- Event member_frames=2 avec exclusion supplémentaire qui laisse < 3 frames → `style_supported=False`.
- Quad-rectified warp : un glyph tourné dans la frame source apparaît droit dans la canonical image.
- Resume : `09_color/colors.json` existant + sidecar valide → skip.

## Périmètre strict

Tu ne touches PAS aux autres stages.
