import logging
import ollama
from pathlib import Path

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str):
        self.model = model

    def analyze(self, image_path: Path, prompt: str, options: dict | None = None) -> str:
        try:
            image_data = image_path.read_bytes()
        except OSError as e:
            raise RuntimeError(f"Cannot read image {image_path}: {e}") from e
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }],
                options=options,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama VLM call failed ({self.model}): {e}") from e
        content = response.message.content
        if not content:
            log.debug("Empty response from %s — full response: %r", self.model, response)
            raise RuntimeError(f"Ollama returned no text content ({self.model})")
        return content

    def chat(self, prompt: str, system: str, retries: int = 3) -> str:
        last_error: Exception | None = None
        for attempt in range(max(1, retries)):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.message.content
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
