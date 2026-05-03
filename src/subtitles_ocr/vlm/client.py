import base64
import logging
from pathlib import Path

from openai import OpenAI

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self._client = OpenAI(base_url=f"{host}/v1", api_key="ollama")

    def analyze(self, image_path: Path, prompt: str, json_mode: bool = False) -> str:
        try:
            image_data = image_path.read_bytes()
        except OSError as e:
            raise RuntimeError(f"Cannot read image {image_path}: {e}") from e
        b64 = base64.b64encode(image_data).decode()
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama VLM call failed ({self.model}): {e}") from e
        content = response.choices[0].message.content
        if not content:
            log.debug("Empty response from %s — full response: %r", self.model, response)
            raise RuntimeError(f"Ollama returned no text content ({self.model})")
        return content

    def chat(self, prompt: str, system: str, retries: int = 3) -> str:
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content
                if not content:
                    log.debug("Empty response from %s — full response: %r", self.model, response)
                    raise RuntimeError(f"Ollama returned no text content ({self.model})")
                return content
            except Exception as e:
                last_error = e
                log.debug("chat attempt %d/%d failed (%s): %s", attempt + 1, retries, self.model, e)
        raise RuntimeError(
            f"Ollama chat failed after {retries} attempts ({self.model}): {last_error}"
        ) from last_error
