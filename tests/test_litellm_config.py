import pytest
import yaml
from pathlib import Path
from subtitles_ocr.litellm_config import get_workers_from_litellm


def _write_config(tmp_path: Path, model_list: list) -> Path:
    config = tmp_path / "litellm.yaml"
    config.write_text(yaml.dump({"model_list": model_list}), encoding="utf-8")
    return config


def test_single_backend(tmp_path):
    config = _write_config(tmp_path, [
        {
            "model_name": "llava:7b",
            "litellm_params": {
                "model": "ollama/llava:7b",
                "api_base": "http://localhost:11434",
                "max_parallel_requests": 4,
            },
        },
    ])
    assert get_workers_from_litellm(config, "llava:7b") == 4


def test_sum_multiple_backends(tmp_path):
    config = _write_config(tmp_path, [
        {
            "model_name": "gemma3:1b-it-qat",
            "litellm_params": {
                "model": "ollama/gemma3:1b-it-qat",
                "api_base": "http://localhost:11434",
                "max_parallel_requests": 10,
            },
        },
        {
            "model_name": "gemma3:1b-it-qat",
            "litellm_params": {
                "model": "ollama/gemma3:1b-it-qat",
                "api_base": "http://192.168.1.61:11434",
                "max_parallel_requests": 8,
            },
        },
        {
            "model_name": "gemma3:1b-it-qat",
            "litellm_params": {
                "model": "ollama/gemma3:1b-it-qat",
                "api_base": "http://192.168.1.36:11434",
                "max_parallel_requests": 16,
            },
        },
    ])
    assert get_workers_from_litellm(config, "gemma3:1b-it-qat") == 34


def test_model_not_found(tmp_path):
    config = _write_config(tmp_path, [
        {
            "model_name": "llava:7b",
            "litellm_params": {
                "model": "ollama/llava:7b",
                "api_base": "http://localhost:11434",
                "max_parallel_requests": 4,
            },
        },
    ])
    with pytest.raises(ValueError, match="No backends found for model 'unknown:1b'"):
        get_workers_from_litellm(config, "unknown:1b")


def test_missing_max_parallel_requests_on_any_backend(tmp_path):
    config = _write_config(tmp_path, [
        {
            "model_name": "llava:7b",
            "litellm_params": {
                "model": "ollama/llava:7b",
                "api_base": "http://localhost:11434",
                "max_parallel_requests": 4,
            },
        },
        {
            "model_name": "llava:7b",
            "litellm_params": {
                "model": "ollama/llava:7b",
                "api_base": "http://192.168.1.61:11434",
                # max_parallel_requests intentionally omitted
            },
        },
    ])
    with pytest.raises(ValueError, match="missing max_parallel_requests"):
        get_workers_from_litellm(config, "llava:7b")


def test_wildcard_not_counted(tmp_path):
    config = _write_config(tmp_path, [
        {
            "model_name": "*",
            "litellm_params": {
                "model": "ollama/*",
                "api_base": "http://localhost:11434",
                "max_parallel_requests": 1,
            },
        },
    ])
    with pytest.raises(ValueError, match="No backends found for model 'gemma3:1b-it-qat'"):
        get_workers_from_litellm(config, "gemma3:1b-it-qat")


def test_empty_config_raises(tmp_path):
    config = tmp_path / "litellm.yaml"
    config.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty or not a YAML mapping"):
        get_workers_from_litellm(config, "llava:7b")


def test_max_parallel_requests_must_be_integer(tmp_path):
    config = _write_config(tmp_path, [
        {
            "model_name": "llava:7b",
            "litellm_params": {
                "model": "ollama/llava:7b",
                "api_base": "http://localhost:11434",
                "max_parallel_requests": "four",
            },
        },
    ])
    with pytest.raises(ValueError, match="invalid max_parallel_requests"):
        get_workers_from_litellm(config, "llava:7b")


def test_max_parallel_requests_must_be_positive(tmp_path):
    config = _write_config(tmp_path, [
        {
            "model_name": "llava:7b",
            "litellm_params": {
                "model": "ollama/llava:7b",
                "api_base": "http://localhost:11434",
                "max_parallel_requests": -1,
            },
        },
    ])
    with pytest.raises(ValueError, match="invalid max_parallel_requests"):
        get_workers_from_litellm(config, "llava:7b")
