# ADR-0005: Normalize Stage Refactor

Branch: `feat/raw-diff-implementation`
Status: Designed and being implemented.
Supersedes: ADR-0002 §3 Stage 10 (Whole-document LLM cleanup).
Revises: nothing else.

## 1. Context

ADR-0002 §3 Stage 10 specified a whole-document LLM cleanup pass: a single LLM call over all events of the episode, producing a strict-schema response with the same event_ids. The intent was to enforce narrative coherence — consistent character names, ponctuation, and optional synopsis application.

Validation on KenIchi S01E01 (539 events) showed this stage times out systematically:
- Output ≈ 6-7k tokens of structured JSON
- gemma3:1b-it-qat with JSON-schema grammar sampling at temperature 1: generation degenerates or fails to terminate
- Three retries × 300s = 900s wall time → `LlmRetryExhausted`

A larger text-only model would slow generation further (more tokens/sec but more parameters), and the published literature (arxiv 2601.19410, "Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing?") concludes that LLMs largely fail to exploit document-level context for contextual error correction. The marginal benefit is not worth the cost and fragility.

A deterministic fuzzy alternative (Levenshtein matching of OCR variants against a user-supplied glossary) was prototyped on paper and rejected:
- Without a user-supplied glossary of canonical names, the stage has no reliable signal to anchor.
- With one, the user cannot anticipate OCR errors in advance, so the glossary would never cover the real failure modes.
- The narrative-coherence goal genuinely requires contextual understanding the deterministic approach cannot provide.

## 2. Decision

The narrative-coherence goal is **abandoned**. The stage is repurposed as a **deterministic, LLM-free text normalization** step:

1. **Unicode NFKC** normalization (folds ligatures `ﬁ` → `fi`, fullwidth → halfwidth, NBSP → regular space, etc.).
2. **Invisible-character stripping**: ZWSP (U+200B), ZWNJ (U+200C), BOM (U+FEFF), WORD JOINER (U+2060). ZWJ (U+200D) is kept (potentially legitimate in combined glyphs / emoji sequences).
3. **Whitespace collapsing**: multiple spaces/tabs collapse to one; spaces around `\n` are trimmed; leading/trailing whitespace per event is stripped. Newlines internal to an event are preserved (they become `\N` at ASS export).
4. **OCR-noise pruning**: drop any event whose normalized text has `len < 2` or contains no alphabetic character (`isalpha()` Unicode-wide).

The stage is renamed `normalize` (workdir directory `11_normalize`, output `normalized.json`). The Pydantic models are renamed accordingly. The `skipped_llm` debug field is removed.

Dropped events have their `event_id` absent from the output. The export stage tolerates the gap (skips events whose id is not in the normalized map).

## 3. Configuration

The CLI flags `--synopsis`, `--doc-cleanup-model`, `--doc-cleanup-parallelism` are removed. The new stage has no user-facing parameters. The `NormalizeConfig` Pydantic model has no fields (placeholder for future extensions).

## 4. Out of scope

- **Language-dependent typographic normalization** (French NBSP before `?!;:`, `...` → `…`, guillemets `«»`, ASCII dashes → em-dash for dialogues): deferred. Will be added if a concrete need arises, guarded by `--language`.
- **Character-name canonicalization**: abandoned. Editor can fix in Aegisub if needed; we will not pretend the pipeline does it.

## 5. Migration

Existing workdirs with `11_doc_cleanup/` become orphan directories. No automatic migration — user deletes the workdir (or the obsolete subdirectory) to force a clean run, consistent with the "no `--resume` flag, delete to force recompute" policy of ADR-0002 §6.
