# Agent P4 — Câblage final + README

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : P3 complète (y compris P3.10).

## Tâche

### 1. `cli.py`

Remplace `build_stages() = []` par la liste réelle :
```python
[
    ConformStage(),
    AlignmentStage(),
    OcrStage(),
    GroupStage(),
    AnimationStage(),
    ColorStage(),
    EventCleanupStage(),
    DocCleanupStage(),
    ExportStage(),
]
```

Vérifie que `globals.debug_images` est bien propagé à `frame_processing` (via `OcrStage` qui en est l'owner).

Retire les flags transitoires `--fps-num`/`--fps-den` que P1.C.2 avait ajoutés : maintenant `ConformStage.probe` fournit ces valeurs pour `PipelineGlobals`. Le boot CLI devient :
1. argparse → flags.
2. `SubprocessFfmpegRunner().probe(hardsub_path)` → `VideoMetadata` → fps/w/h pour `PipelineGlobals`.
3. `run_pipeline`.

### 2. `README.md`

Met à jour :
- Documente la nouvelle CLI (tous les flags d'ADR-0002 §4).
- Documente le pipeline (11 stages MVP + 12 stages avec animation Phase 6).
- Documente le workdir layout (ADR-0003 §5).
- Documente la stratégie de resume.
- Supprime toute mention du pipeline VLM legacy.

### 3. Test e2e mocké

`tests/test_main_e2e.py` qui appelle `main()` avec des fakes (via injection — `run_pipeline(stages=...)` exposé par P1.C.2).

### 4. Vérification

`uv run python -m pytest` vert sur l'ensemble.

## Commit

`feat: wire diff-based hardsub extraction pipeline`.
