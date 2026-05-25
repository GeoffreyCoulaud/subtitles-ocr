import json

import httpx
import pytest
from pydantic import BaseModel

from subtitles_ocr.llm import LlmCallFailed, OllamaLlmClient
from subtitles_ocr.retry import RetryConfig


class _Reply(BaseModel):
    answer: str
    n: int


def _json_body(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _ok_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=_json_body(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": content}},
                ],
            }
        ),
        headers={"content-type": "application/json"},
    )


def _make_client(
    handler,
    *,
    max_retries: int = 2,
    backoff: tuple[float, ...] = (0.0, 0.0),
) -> tuple[OllamaLlmClient, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    client = OllamaLlmClient(
        retry=RetryConfig(max_retries=max_retries, backoff_seconds=backoff),
        transport=transport,
    )
    return client, captured


# ---------------------------------------------------------------------------
# 200 OK → parsed model
# ---------------------------------------------------------------------------


def test_complete_returns_parsed_model_on_200(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = json.dumps({"answer": "hello", "n": 3})

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(body)

    client, captured = _make_client(handler)
    result = client.complete("prompt", _Reply, model="qwen2.5")

    assert isinstance(result, _Reply)
    assert result.answer == "hello"
    assert result.n == 3
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# Request shape: POST /v1/chat/completions with response_format json_schema
# ---------------------------------------------------------------------------


def test_request_targets_openai_compat_endpoint(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = json.dumps({"answer": "x", "n": 0})

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(body)

    client, captured = _make_client(handler)
    client.complete("the prompt", _Reply, model="qwen2.5")

    req = captured[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/chat/completions"
    payload = json.loads(req.content)
    assert payload["model"] == "qwen2.5"
    assert payload["messages"][-1]["content"] == "the prompt"
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == _Reply.model_json_schema()
    assert rf["json_schema"]["strict"] is True


# ---------------------------------------------------------------------------
# 503 then 200 → succeeds after one retry
# ---------------------------------------------------------------------------


def test_503_then_200_succeeds_after_one_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = json.dumps({"answer": "ok", "n": 1})
    seq: list[httpx.Response] = [
        httpx.Response(503, content=b"unavailable"),
        _ok_response(body),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return seq.pop(0)

    client, captured = _make_client(handler)
    result = client.complete("p", _Reply, model="m")

    assert result.answer == "ok"
    assert len(captured) == 2


# ---------------------------------------------------------------------------
# 503 × 3 → LlmCallFailed
# ---------------------------------------------------------------------------


def test_503_thrice_raises_llm_call_failed(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"unavailable")

    client, captured = _make_client(handler)
    with pytest.raises(LlmCallFailed):
        client.complete("p", _Reply, model="m")

    # 1 initial + 2 retries = 3 calls
    assert len(captured) == 3


def test_llm_call_failed_chains_original_exception(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"unavailable")

    client, _ = _make_client(handler)
    with pytest.raises(LlmCallFailed) as excinfo:
        client.complete("p", _Reply, model="m")

    assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)


# ---------------------------------------------------------------------------
# 400 → propagates httpx.HTTPStatusError immediately, no retry, no LlmCallFailed
# ---------------------------------------------------------------------------


def test_400_propagates_immediately(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b"bad request")

    client, captured = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.complete("p", _Reply, model="m")

    assert len(captured) == 1


# ---------------------------------------------------------------------------
# Malformed JSON in response content → retry → exhaustion → LlmCallFailed
# ---------------------------------------------------------------------------


def test_malformed_json_triggers_retry_then_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response("not json at all{{")

    client, captured = _make_client(handler)
    with pytest.raises(LlmCallFailed):
        client.complete("p", _Reply, model="m")

    assert len(captured) == 3


# ---------------------------------------------------------------------------
# Pydantic schema mismatch (missing required field) → retry → LlmCallFailed
# ---------------------------------------------------------------------------


def test_schema_mismatch_triggers_retry_then_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = json.dumps({"answer": "missing n"})

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(body)

    client, captured = _make_client(handler)
    with pytest.raises(LlmCallFailed):
        client.complete("p", _Reply, model="m")

    assert len(captured) == 3


# ---------------------------------------------------------------------------
# Defaults: DEFAULT_LLM_RETRY = (max_retries=2, backoff=(1.0, 3.0))
# ---------------------------------------------------------------------------


def test_default_retry_constant():
    from subtitles_ocr.llm.ollama import DEFAULT_LLM_RETRY

    assert DEFAULT_LLM_RETRY.max_retries == 2
    assert DEFAULT_LLM_RETRY.backoff_seconds == (1.0, 3.0)


# ---------------------------------------------------------------------------
# _is_llm_retryable: classification truth table
# ---------------------------------------------------------------------------


def test_is_llm_retryable_classification():
    from subtitles_ocr.llm.ollama import _is_llm_retryable

    req = httpx.Request("POST", "http://x")

    assert _is_llm_retryable(httpx.TimeoutException("t")) is True
    assert _is_llm_retryable(httpx.ConnectError("c")) is True
    assert _is_llm_retryable(
        httpx.HTTPStatusError(
            "500",
            request=req,
            response=httpx.Response(500, request=req),
        )
    ) is True
    assert _is_llm_retryable(
        httpx.HTTPStatusError(
            "503",
            request=req,
            response=httpx.Response(503, request=req),
        )
    ) is True
    assert _is_llm_retryable(
        httpx.HTTPStatusError(
            "400",
            request=req,
            response=httpx.Response(400, request=req),
        )
    ) is False
    assert _is_llm_retryable(
        httpx.HTTPStatusError(
            "404",
            request=req,
            response=httpx.Response(404, request=req),
        )
    ) is False
    assert _is_llm_retryable(json.JSONDecodeError("x", "y", 0)) is True
    assert _is_llm_retryable(ValueError("z")) is False
