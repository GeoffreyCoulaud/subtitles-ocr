# Agent P1.C.1 — llm/ollama.py

> Lis d'abord [`preamble.md`](./preamble.md).

**Pré-requis** : P1.A.3 (`retry.py`), P1.A.4 (`llm/protocol.py`).

**Référence** : ADR-0004 §10.1, §11.2.

Tâche : implémenter le client Ollama compatible OpenAI dans `src/subtitles_ocr/llm/ollama.py`.

## Dépendance à ajouter

```
uv add httpx
```

## Contrat

- `DEFAULT_LLM_RETRY = RetryConfig(max_retries=2, backoff_seconds=(1.0, 3.0))`.
- `def _is_llm_retryable(e: Exception) -> bool` :
  - True pour `httpx.TimeoutException`, `httpx.NetworkError`, `httpx.HTTPStatusError` avec code `>= 500`, `json.JSONDecodeError`, `pydantic.ValidationError`.
  - False pour 4xx.
- `class OllamaLlmClient` (implémente `LlmClient`) :
  - `__init__(host="http://localhost:11434", retry=DEFAULT_LLM_RETRY, request_timeout_seconds=60.0, transport: httpx.BaseTransport | None = None)`. Le `transport` permet de tester avec `httpx.MockTransport`.
  - Utilise l'API OpenAI-compatible d'Ollama (`POST /v1/chat/completions` avec `response_format = {"type": "json_schema", "json_schema": {...}}` construit depuis `response_schema.model_json_schema()`).
  - `@retry_method(is_retryable=_is_llm_retryable)` sur `_complete_once`.
  - `complete()` appelle `_complete_once` ; sur exception retryable exhaustée, lève `LlmCallFailed(...)`. Sur exception non-retryable (ex. 4xx), propage telle quelle.

## Tests (`tests/test_ollama_llm.py`)

Utiliser `httpx.MockTransport` EXCLUSIVEMENT, JAMAIS monkeypatcher httpx.

- 200 + JSON valide → retourne le model parsé.
- 503 puis 200 → retourne après 1 retry (`retry=RetryConfig(2, (0.0, 0.0))`).
- 503 × 3 → `LlmCallFailed`.
- 400 → propage `httpx.HTTPStatusError` immédiatement (pas de retry, pas de `LlmCallFailed`).
- JSON malformé dans la réponse → retry, échoue après 3 → `LlmCallFailed`.
- Schéma Pydantic mismatch (ex. champ manquant) → retry → `LlmCallFailed`.
- Compteur de tentatives correct (compter les invocations du transport).

## Sortie

Met à jour `src/subtitles_ocr/llm/__init__.py` pour ré-exporter `OllamaLlmClient`.

## Périmètre strict

Tu ne touches qu'à `src/subtitles_ocr/llm/ollama.py`, son `__init__.py`, et `tests/test_ollama_llm.py`.
