# Agent P1.A.2 — timing.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0004 §9.

Tâche : créer `src/subtitles_ocr/timing.py` avec les helpers de conversion frame↔ms.

## Contrat

- `frame_to_ms(frame_idx: int, fps: Fraction) -> int` (arrondi `int(round(...))`).
- `ms_to_frame(ms: int, fps: Fraction) -> int` (arrondi `int(round(...))`).
- `fps` est `fractions.Fraction`, JAMAIS `float`.

## Tests (`tests/test_timing.py`)

- Round-trip stable à 23.976 fps (`Fraction(24000, 1001)`) : pour TOUS les frame idx dans `range(0, 35000, 137)`, `ms_to_frame(frame_to_ms(i, fps), fps) == i` (assert sur TOUS, pas un seul échantillon).
- Round-trip stable à 24 fps exact (`Fraction(24, 1)`) sur le même intervalle.
- `frame_to_ms(0, fps) == 0`.
- Test documentant le footgun `float` : `Fraction(24000, 1001)` produit un résultat différent de `float(23.976)` sur frame 35000. Le test affirme que la version `Fraction` est correcte (drift bornée).

## Périmètre strict

Tu ne touches à AUCUN autre fichier.
