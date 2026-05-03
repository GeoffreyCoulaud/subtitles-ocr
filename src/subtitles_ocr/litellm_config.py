from pathlib import Path

import yaml


def get_workers_from_litellm(config_path: Path, model_name: str) -> int:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"litellm config at '{config_path}' is empty or not a YAML mapping")
    matches = [
        entry for entry in data.get("model_list", [])
        if entry.get("model_name") == model_name
    ]
    if not matches:
        raise ValueError(
            f"No backends found for model '{model_name}' in litellm config"
        )
    total = 0
    for entry in matches:
        params = entry.get("litellm_params", {})
        if "max_parallel_requests" not in params:
            api_base = params.get("api_base", "<unknown>")
            raise ValueError(
                f"Backend at '{api_base}' for model '{model_name}' is missing max_parallel_requests"
            )
        value = params["max_parallel_requests"]
        if not isinstance(value, int) or value < 1:
            api_base = params.get("api_base", "<unknown>")
            raise ValueError(
                f"Backend at '{api_base}' for model '{model_name}' has invalid max_parallel_requests: {value!r} (must be a positive integer)"
            )
        total += value
    return total
