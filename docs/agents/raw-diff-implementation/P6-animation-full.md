# Agent P6 — AnimationStage reconstruction complète

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : MVP validé contre KenIchi ground truth (Phase 5).

**Référence** : ADR-0003 §4.2 (algorithme complet).

## Tâche

Remplace l'impl passe-plat de `pipeline/animation.py` par l'algorithme complet d'ADR-0003 §4.2.

### Sub-stage A1 — `\move` intra-event slow

Pour chaque event :
- Régression linéaire sur `quads_per_frame[K]` centroids `(cx(K), cy(K))` vs `K`.
- Si `total_displacement_px ≥ MIN_MOVE_DISPLACEMENT_PX` ET `R²(cx) ≥ 0.95` ET `R²(cy) ≥ 0.95` → `motion = {type: "linear", start: (x1, y1), end: (x2, y2)}`.
- Si déplacement ≥ seuil mais R² fail → `motion = {type: "nonlinear_flagged"}`.
- Sinon → `motion = None`.

### Sub-stage A2 — `\move` inter-event fragmenté

Across consecutive events (post-A1) :
- Candidate chain : `fansub_frame_end` → next `fansub_frame_start` ≤ `MOVE_GAP_TOLERANCE_MS / (1000/fps)` frames.
- Text continuity : Levenshtein < 0.2 sur les `raw_ocr_texts` les plus confiants.
- Trajectory continuity : combined `quads_per_frame` fit `R² ≥ 0.95` sur cx ET cy.
- Si toutes conditions OK → merge en un seul event ; concatène `quads_per_frame`, regenere `raw_ocr_texts` et `member_frame_indices`.
- Si text + temporal OK mais R² fail → merge static + flag β.

### Sub-stage B — `\fad` detection (post-merge)

Pour chaque event (static, linear move, ou flagged) :
1. Search window : `[start - W, start)` et `(end, end + W]`, `W = ceil(FADE_SEARCH_WINDOW_MS / (1000/fps))`. Exclure frames attribuées à un autre event.
2. Bbox du score :
   - `motion.type == "linear"` : extrapole la trajectoire dans la search window, bbox = bbox du quad prédit.
   - `motion.type == "nonlinear_flagged"` : **skip `\fad` detection**.
   - `motion is None` : bbox = bbox de `quad_median`.
3. Score : `score(K) = mean(diff_intensity(K) bbox) / mean(diff_intensity(start) bbox)` où `diff_intensity` est recomputé on-the-fly (LCN + Sobel restricted à bbox).
4. Linear fit ancré à `(start_frame, score=1)` (ou `(end_frame, score=1)` pour fade-out).
   - Observations : frames K où `score(K) ∈ [0.05, 0.95]`.
   - Fit contraint passant par l'anchor.
5. Decision per side :
   - `count(observations) < ceil(MIN_FADE_DURATION_MS / (1000/fps))` → `t = 0`.
   - `R²(fit) < 0.7` → `t = 0` + warning.
   - Sinon → extrapole à `score = 0`. `t_extrapolated_ms = |start_frame - fade_origin_frame| × (1000 / fps)`.
   - Belt-and-braces : `t < MIN_FADE_DURATION_MS` → 0 ; `t > FADE_DURATION_CAP_MS` → 0 + warning.
   - Sinon → `t_in_ms` / `t_out_ms` adopté. Étend `fansub_frame_start` / `_end`.
6. Consistency check : `t_in_ms + t_out_ms > duration_ms` → revert both to 0, mark static, warning.

### Tunables

`AnimationConfig` est déjà scaffoldé en P2 avec les defaults d'ADR-0003 §8.

### STAGE_VERSION

Passe `STAGE_VERSION` de 1 à 2 dans `pipeline/animation.py` : tous les caches `08_animation/` aval invalident automatiquement.

## Tests TDD complets (`tests/pipeline/test_animation.py`)

Un test par cas du tableau ADR-0003 §7 (et plus) :
- Static (motion=None, fade=0/0).
- Linear move intra-event (R² 0.99 sur trajectoire fittée).
- Linear move intra-event avec déplacement < seuil → static.
- Flagged nonlinear (R² < 0.95) → `motion.type == "nonlinear_flagged"`.
- Inter-event merge happy path (2 events fragmentés, mêmes texte + trajectoire fit).
- Inter-event merge bloqué par Levenshtein.
- Inter-event merge bloqué par gap > tolerance.
- Inter-event merge avec R² fail → flag β.
- Fade-in only (`t_in > 0`, `t_out = 0`).
- Fade-out only.
- Full fade.
- Fade fit R² < 0.7 → revert to 0.
- Fade `t_extrapolated < MIN` → revert.
- Fade `t_extrapolated > CAP` → revert + warning.
- Partial fade (`t_in + t_out > duration`) → both revert.
- `motion.type=="nonlinear_flagged"` → fade skipped (assert `fade_in_ms=0`, `fade_out_ms=0` même si vrais fades dans les données).
- `\fad` for moving subs : bbox extrapolated along trajectory (test critique).

Frames numpy synthétiques (créer une vidéo numpy avec un glyph qui fade-in puis se déplace).

## Périmètre strict

Tu ne touches PAS aux autres stages.
