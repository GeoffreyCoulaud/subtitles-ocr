# Inference setup

The pipeline uses three models. The two vision models never coexist in VRAM — each step completes before the next model loads:

| Role | Model | VRAM |
|------|-------|------|
| Pre-filter | `llava:7b` | 4.7 GB |
| Analysis | `qwen3-vl:4b` | ~6 GB |
| Reconciliation | `gemma3:1b-it-qat` | 1.0 GB |

All three can be overridden with `--filter-model`, `--analyze-model`, and `--reconcile-model`.

The tool talks to any OpenAI-compatible inference server via `--inference-url` (default: `http://localhost:11434`).

## Local Ollama

```bash
ollama pull llava:7b
ollama pull qwen3-vl:4b
ollama pull gemma3:1b-it-qat
ollama serve
```

```bash
uv run subtitles-ocr episode01.mkv
```

## Remote Ollama

```bash
uv run subtitles-ocr episode01.mkv --inference-url http://gpu-server:11434
```

## Multiple machines with LiteLLM proxy

LiteLLM proxy runs as a standalone service (not part of this codebase). It routes each model to the right machines and uses **least-busy** load balancing: a faster machine finishes requests sooner, frees its slot, and naturally receives more traffic — no manual weights needed.

**Install and start the proxy:**

```bash
# The official docs omit --with pillow, but pillow is required at runtime
uv tool install 'litellm[proxy]' --with pillow
litellm --config litellm.yaml
```

**Example `litellm.yaml`:**

```yaml
router_settings:
  routing_strategy: least-busy

model_list:
  # llava:7b is large — only the machine with enough VRAM
  - model_name: llava:7b
    litellm_params:
      model: ollama/llava:7b
      api_base: http://big-machine:11434
      max_parallel_requests: 2

  # qwen3-vl:4b needs ~6 GiB — runs on most machines
  - model_name: qwen3-vl:4b
    litellm_params:
      model: ollama/qwen3-vl:4b
      api_base: http://big-machine:11434
      max_parallel_requests: 2
  - model_name: qwen3-vl:4b
    litellm_params:
      model: ollama/qwen3-vl:4b
      api_base: http://small-machine:11434
      max_parallel_requests: 1

  - model_name: gemma3:1b-it-qat
    litellm_params:
      model: ollama/gemma3:1b-it-qat
      api_base: http://big-machine:11434
      max_parallel_requests: 8
  - model_name: gemma3:1b-it-qat
    litellm_params:
      model: ollama/gemma3:1b-it-qat
      api_base: http://small-machine:11434
      max_parallel_requests: 4

  # catch-all: any custom model (--analyze-model, --filter-model, --reconcile-model)
  # not listed above is forwarded to the main machine as-is
  - model_name: "*"
    litellm_params:
      model: "ollama/*"
      api_base: http://big-machine:11434
```

**Point the tool at the proxy:**

```bash
uv run subtitles-ocr episode01.mkv --inference-url http://localhost:4000
```

**Auto-detect worker counts with `--litellm-config`:**

Pass the same config file to the tool and it will sum `max_parallel_requests` across all backends for each model, using the result as the number of parallel workers:

```bash
uv run subtitles-ocr episode01.mkv \
  --inference-url http://localhost:4000 \
  --litellm-config litellm.yaml
```

With the example above, this sets `--reconcile-workers 12` (8 + 4) and `--filter-workers 2` automatically. Explicit `--*-workers` flags always take priority over the auto-detected value.

Every backend for a model used by the pipeline must declare `max_parallel_requests` — the tool raises an error if any entry is missing it.

Each machine must have the relevant models pulled (`ollama pull <model>`) before the proxy starts routing to it.