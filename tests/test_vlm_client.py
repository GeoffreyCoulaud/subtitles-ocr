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
