# Plan d'implémentation parallèle — guide de dispatch

Ce répertoire contient :
- `preamble.md` — règles communes à tous les agents (TDD, uv, Pydantic v2, no monkeypatch, …).
- `PXX-*.md` — un fichier par agent, contenu spécifique uniquement.

## Comment dispatcher un agent

L'agent doit lire **`preamble.md` + son fichier spécifique**. Deux options :

**Option A (recommandée)** : passe-lui les deux chemins dans le prompt :
```
Lis d'abord docs/superpowers/agent-prompts/preamble.md, puis suis les
instructions de docs/superpowers/agent-prompts/P3-4-group.md.
```

**Option B** : inline les deux fichiers dans le prompt (concaténation).

## Ordre de dispatch

### Phase 0 — Séquentiel, 1 agent
- [`P0-cleanup.md`](P0-cleanup.md)

### Phase 1.A — 6 agents en parallèle (aucune dépendance croisée)
- [`P1A-1-exceptions.md`](P1A-1-exceptions.md)
- [`P1A-2-timing.md`](P1A-2-timing.md)
- [`P1A-3-retry.md`](P1A-3-retry.md)
- [`P1A-4-llm-protocol.md`](P1A-4-llm-protocol.md)
- [`P1A-5-ocr-protocol.md`](P1A-5-ocr-protocol.md)
- [`P1A-6-ffmpeg-protocol.md`](P1A-6-ffmpeg-protocol.md)

### Phase 1.B — 3 agents en parallèle (dépendent de P1.A)
- [`P1B-1-meta.md`](P1B-1-meta.md)
- [`P1B-2-config.md`](P1B-2-config.md)
- [`P1B-3-io.md`](P1B-3-io.md)

### Phase 1.C — 2 agents en parallèle (dépendent de P1.B)
- [`P1C-1-ollama.md`](P1C-1-ollama.md)
- [`P1C-2-cli-skeleton.md`](P1C-2-cli-skeleton.md)

### Phase 2 — Séquentiel, 1 agent (dépend de P1)
- [`P2-scaffold.md`](P2-scaffold.md)

### Phase 3 — 9 agents stage en parallèle + 1 agent intégration séquentiel
- [`P3-1-conform.md`](P3-1-conform.md)
- [`P3-2-alignment.md`](P3-2-alignment.md)
- [`P3-3-frame-processing-ocr.md`](P3-3-frame-processing-ocr.md)
- [`P3-4-group.md`](P3-4-group.md)
- [`P3-5-color.md`](P3-5-color.md)
- [`P3-6-event-cleanup.md`](P3-6-event-cleanup.md)
- [`P3-7-doc-cleanup.md`](P3-7-doc-cleanup.md)
- [`P3-8-export.md`](P3-8-export.md)
- [`P3-9-animation-mvp.md`](P3-9-animation-mvp.md)
- [`P3-10-integration.md`](P3-10-integration.md) — attend la complétion des 9 précédents

### Phase 4 — Séquentiel, 1 agent
- [`P4-wire.md`](P4-wire.md)

### Phase 5 — Hors agents
Validation empirique manuelle sur KenIchi e01 contre ground-truth.

### Phase 6 — Séquentiel, 1 agent (différé)
- [`P6-animation-full.md`](P6-animation-full.md) — après validation MVP.

## Recommandations de dispatch

- **Isolation** : utiliser `Agent({ isolation: "worktree", ... })` pour chaque agent P1.x et P3.x pour éviter les collisions de fichiers sur la même branche.
- **Modèle** : `sonnet` suffit pour les agents P1.A (mécaniques) ; `opus` recommandé pour P1.C, P2 et P3 (algorithmes non triviaux).
- **Vérification entre vagues** : après chaque vague, lance `uv run python -m pytest` et inspecte les diffs avant la suivante.
- **Coordination de merge** : chaque prompt délimite explicitement ce que l'agent NE doit PAS toucher. Si deux agents doivent éditer `config.py` (P1.B.2, P2, certains P3.x), respecter la règle « chacun n'ajoute QUE ses propres champs ».

## Tableau récapitulatif du parallélisme

| Phase | Agents simultanés | Chemin critique |
|---|---|---|
| P0 | 1 | bloquant |
| P1.A | 6 | court |
| P1.B | 3 | moyen |
| P1.C | 2 | moyen |
| P2 | 1 | moyen |
| P3.1–P3.9 | 9 | long |
| P3.10 | 1 | court (post-P3) |
| P4 | 1 | court |
| P6 (différé) | 1 | moyen |
