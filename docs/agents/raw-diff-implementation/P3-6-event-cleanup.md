# Agent P3.6 — EventCleanupStage

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stage 9 (renuméroté Stage 10 par ADR-0003).

**Pré-requis** : P2 terminée.

## Tâche

Implémente `EventCleanupStage.run` dans `pipeline/event_cleanup.py`.

- Lit `08_animation/animation.json` (events avec `raw_ocr_texts`).
- Pour chaque event :
  - **Pré-check consensus** : si `len(set(event.raw_ocr_texts)) == 1` → `cleaned_text = event.raw_ocr_texts[0]`, `skipped_llm=True`, AUCUN appel LLM.
  - Sinon : appelle `self.llm.complete(prompt, CleanedEvent, model=config.model)`. Prompt minimal : présente les variantes OCR + demande la canonical text + fix confusables (`rn`/`m`, `I`/`l`/`1`, accent recovery).
- `ThreadPoolExecutor(max_workers=config.parallelism)` pour paralléliser les appels LLM.
- Écrit `10_event_cleanup/cleaned.jsonl` via `JsonlWriter` (`chunk_size=100`).
- Sur `LlmCallFailed` (après retries internes du client) → enveloppe en `LlmRetryExhausted(stage="10_event_cleanup", hint="Check Ollama logs and --event-cleanup-model availability.")`.
- La parallélisation n'invalide PAS le cache (champ annoté `NoCacheKey` déjà fait en P2).

## Tests (`tests/pipeline/test_event_cleanup.py`)

Utiliser `class FakeLlm` qui implémente `LlmClient` :
- Consensus (toutes variantes identiques) → 0 appel LLM, `skipped_llm=True`.
- Variantes divergentes → 1 appel LLM par event.
- `FakeLlm` qui lève `LlmCallFailed` → `LlmRetryExhausted` propagé.
- Resume : jsonl partiel (3 events traités sur 5) → reprend à event 4.
- Parallélisme : 4 events, `parallelism=4`, FakeLlm enregistre les appels (tous appelés exactement une fois, ordre quelconque).
- Sidecar contient `model` mais PAS `parallelism`.

## Périmètre strict

Tu ne touches PAS aux autres stages.
