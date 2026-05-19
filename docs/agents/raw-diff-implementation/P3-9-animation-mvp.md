# Agent P3.9 — AnimationStage MVP (passe-plat)

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0003 §3 (« disabling Stage 8 yields ADR-0002 behavior exactly »).

**Pré-requis** : P2 terminée.

## Tâche

Implémente `AnimationStage.run` dans `pipeline/animation.py` en mode **PASSE-PLAT** pour le MVP.

L'impl complète d'ADR-0003 §4.2 (intra/inter `\move` detection + `\fad` linear regression) est différée en **Phase 6**.

### Comportement passe-plat

- Lit `07_group/events.json` (`GroupResult`, contient des `SubtitleEvent`).
- Pour chaque `SubtitleEvent`, produit un `AnimatedEvent` avec :
  - `event_id`, `fansub_frame_start`, `fansub_frame_end`, `raw_ocr_texts`, `raw_ocr_confidences`, `quads_per_frame`, `quad_median`, `member_frame_indices` recopiés.
  - `motion = None`
  - `fade_in_ms = 0`
  - `fade_out_ms = 0`
- `stats = {"static": N, "linear_move": 0, "flagged_nonlinear": 0, "fade_in_only": 0, "fade_out_only": 0, "full_fade": 0}`.
- Écrit `08_animation/animation.json` (atomique) + sidecar.

## Tests (`tests/pipeline/test_animation_mvp.py`)

- 3 events in (avec `quads_per_frame` variés) → 3 `AnimatedEvent` out avec `motion=None`, `fade_in_ms=0`, `fade_out_ms=0`.
- `quads_per_frame` strictement préservé.
- Stats : `static=3`, autres = 0.
- Resume : sidecar valide → skip.

## Périmètre strict

Tu ne touches PAS aux autres stages.
