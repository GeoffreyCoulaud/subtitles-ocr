import ollama
from pathlib import Path


class OllamaClient:
    def __init__(self, model: str):
        self.model = model

    def analyze(self, image_path: Path, prompt: str) -> str:
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
            )
        except Exception as e:
            raise RuntimeError(f"Ollama VLM call failed ({self.model}): {e}") from e
        content = response.message.content
        if content is None:
            raise RuntimeError(f"Ollama returned no text content ({self.model})")
        return content
