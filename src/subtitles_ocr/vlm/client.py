import ollama
from pathlib import Path


class OllamaClient:
    def __init__(self, model: str):
        self.model = model

    def analyze(self, image_path: Path, prompt: str) -> str:
        image_data = image_path.read_bytes()
        response = ollama.chat(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_data],
            }],
        )
        return response["message"]["content"]
