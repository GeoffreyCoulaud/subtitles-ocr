import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from subtitles_ocr.vlm.client import OllamaClient


def test_analyze_passes_image_and_prompt():
    mock_response = MagicMock()
    mock_response.message.content = '[]'
    with patch("subtitles_ocr.vlm.client.ollama.chat", return_value=mock_response) as mock_chat:
        with patch.object(Path, "read_bytes", return_value=b"image_data"):
            client = OllamaClient(model="test-model")
            result = client.analyze(Path("frame.jpg"), "my prompt")

    assert result == "[]"
    call_args = mock_chat.call_args
    assert call_args.kwargs["model"] == "test-model"
    messages = call_args.kwargs["messages"]
    assert messages[0]["content"] == "my prompt"
    assert "images" in messages[0]


def test_analyze_returns_raw_string():
    mock_response = MagicMock()
    mock_response.message.content = '[{"text": "Bonjour"}]'
    with patch("subtitles_ocr.vlm.client.ollama.chat", return_value=mock_response):
        with patch.object(Path, "read_bytes", return_value=b"image_data"):
            client = OllamaClient(model="test-model")
            result = client.analyze(Path("frame.jpg"), "prompt")
    assert result == '[{"text": "Bonjour"}]'


def test_analyze_raises_on_none_content():
    mock_response = MagicMock()
    mock_response.message.content = None
    with patch("subtitles_ocr.vlm.client.ollama.chat", return_value=mock_response):
        with patch.object(Path, "read_bytes", return_value=b"image_data"):
            client = OllamaClient(model="test-model")
            with pytest.raises(RuntimeError, match="no text content"):
                client.analyze(Path("frame.jpg"), "prompt")


def test_chat_returns_text_response():
    mock_response = MagicMock()
    mock_response.message.content = "Bonjour tout le monde"
    with patch("subtitles_ocr.vlm.client.ollama.chat", return_value=mock_response) as mock_chat:
        client = OllamaClient(model="test-model")
        result = client.chat("prompt text", system="system text")
    assert result == "Bonjour tout le monde"
    messages = mock_chat.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "system text"}
    assert messages[1] == {"role": "user", "content": "prompt text"}


def test_chat_retries_on_failure_then_succeeds():
    ok_response = MagicMock()
    ok_response.message.content = "success"
    with patch("subtitles_ocr.vlm.client.ollama.chat",
               side_effect=[Exception("timeout"), ok_response]) as mock_chat:
        client = OllamaClient(model="test-model")
        result = client.chat("prompt", system="system", retries=3)
    assert result == "success"
    assert mock_chat.call_count == 2


def test_chat_raises_after_all_retries_exhausted():
    with patch("subtitles_ocr.vlm.client.ollama.chat", side_effect=Exception("always fails")) as mock_chat:
        client = OllamaClient(model="test-model")
        with pytest.raises(RuntimeError, match="always fails"):
            client.chat("prompt", system="system", retries=3)
    assert mock_chat.call_count == 3


def test_chat_raises_on_empty_content():
    mock_response = MagicMock()
    mock_response.message.content = None
    with patch("subtitles_ocr.vlm.client.ollama.chat",
               side_effect=[mock_response, mock_response]) as mock_chat:
        client = OllamaClient(model="test-model")
        with pytest.raises(RuntimeError, match="no text content"):
            client.chat("prompt", system="system", retries=2)
    assert mock_chat.call_count == 2
