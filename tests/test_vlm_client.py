# tests/test_vlm_client.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from subtitles_ocr.vlm.client import OllamaClient


def _make_response(content: str | None) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def test_analyze_passes_image_and_prompt():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_response("[]")
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        with patch.object(Path, "read_bytes", return_value=b"image_data"):
            client = OllamaClient(model="test-model")
            result = client.analyze(Path("frame.jpg"), "my prompt")

    assert result == "[]"
    call_args = mock_openai.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "test-model"
    content_parts = call_args.kwargs["messages"][0]["content"]
    assert content_parts[0]["text"] == "my prompt"
    assert content_parts[1]["type"] == "image_url"


def test_analyze_returns_raw_string():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_response('[{"text": "Bonjour"}]')
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        with patch.object(Path, "read_bytes", return_value=b"image_data"):
            client = OllamaClient(model="test-model")
            result = client.analyze(Path("frame.jpg"), "prompt")
    assert result == '[{"text": "Bonjour"}]'


def test_analyze_raises_runtime_error_on_none_content():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_response(None)
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        with patch.object(Path, "read_bytes", return_value=b"image_data"):
            client = OllamaClient(model="test-model")
            with pytest.raises(RuntimeError, match="no text content"):
                client.analyze(Path("frame.jpg"), "prompt")


def test_analyze_propagates_oserror_from_read_bytes():
    mock_openai = MagicMock()
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        with patch.object(Path, "read_bytes", side_effect=OSError("no such file")):
            client = OllamaClient(model="test-model")
            with pytest.raises(OSError):
                client.analyze(Path("missing.jpg"), "prompt")


def test_analyze_propagates_openai_exceptions():
    from openai import APIConnectionError
    import httpx
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("GET", "http://localhost")
    )
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        with patch.object(Path, "read_bytes", return_value=b"image_data"):
            client = OllamaClient(model="test-model")
            with pytest.raises(APIConnectionError):
                client.analyze(Path("frame.jpg"), "prompt")


def test_chat_returns_text_response():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_response("Bonjour tout le monde")
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        client = OllamaClient(model="test-model")
        result = client.chat("prompt text", system="system text")
    assert result == "Bonjour tout le monde"
    messages = mock_openai.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "system text"}
    assert messages[1] == {"role": "user", "content": "prompt text"}


def test_chat_raises_runtime_error_on_empty_content():
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _make_response(None)
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        client = OllamaClient(model="test-model")
        with pytest.raises(RuntimeError, match="no text content"):
            client.chat("prompt", system="system")


def test_chat_propagates_openai_exceptions():
    from openai import RateLimitError
    import httpx
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = RateLimitError(
        message="rate limited",
        response=httpx.Response(429, request=httpx.Request("POST", "http://localhost")),
        body=None,
    )
    with patch("subtitles_ocr.vlm.client.OpenAI", return_value=mock_openai):
        client = OllamaClient(model="test-model")
        with pytest.raises(RateLimitError):
            client.chat("prompt", system="system")
