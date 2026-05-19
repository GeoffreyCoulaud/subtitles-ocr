# Agent P3.7 — DocCleanupStage

> Lis d'abord [`preamble.md`](./preamble.md).

**Référence** : ADR-0002 §3 Stage 10 (renuméroté Stage 11 par ADR-0003).

**Pré-requis** : P2 terminée.

## Tâche

Implémente `DocCleanupStage.run` dans `pipeline/doc_cleanup.py`.

- Lit `10_event_cleanup/cleaned.jsonl`.
- Construit le prompt user : JSON `{"synopsis": <text|null>, "events": [{"id": int, "text": str}, ...]}` où `synopsis` est lu depuis `globals.synopsis_path` si défini (free Markdown chargé tel quel).
- Un seul appel `self.llm.complete(prompt, DocCleanupResult, model=config.model)`.
- Validation post-call : nombre d'events identique ET ordre des `event_id` identique à l'input. Sinon → `LlmResponseSchemaError`.
- Si l'appel lève une exception "context overflow"-like (à détecter via message ou code spécifique d'Ollama) → `LlmPromptTooLarge`. Pas de fallback chunké (cf. ADR-0002 §3 Stage 10).
- `LlmCallFailed` après retries → `LlmRetryExhausted(stage="11_doc_cleanup")`.
- Écrit `11_doc_cleanup/cleaned_final.json` (atomique) + sidecar.

## Tests (`tests/pipeline/test_doc_cleanup.py`)

Avec `FakeLlm` :
- Happy path : 5 events in → FakeLlm renvoie 5 events out avec mêmes ids.
- IDs mismatch → `LlmResponseSchemaError`.
- Count mismatch → `LlmResponseSchemaError`.
- `LlmCallFailed` → `LlmRetryExhausted`.
- Synopsis path None → prompt contient `"synopsis": null`.
- Synopsis path valide → contenu du fichier inclus dans le prompt.
- Resume : `cleaned_final.json` existant + sidecar valide → skip.

## Périmètre strict

Tu ne touches PAS aux autres stages.
