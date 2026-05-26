"""Default production implementations of `VadModel` and `AudioLoader`.

`SileroVadModel` — loads silero-vad ONNX (bundled in the silero_vad package)
directly via onnxruntime. Avoids the package's Python wrapper which imports
torchaudio and pulls a fragile CUDA runtime dependency we don't need for
inference.

`WavAudioLoader` — reads WAV → mono float32 in `[-1, 1]` via `scipy.io.wavfile`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class WavAudioLoader:
    def load(self, path) -> tuple[np.ndarray, int]:
        from scipy.io import wavfile

        sr, data = wavfile.read(str(path))
        arr = np.asarray(data)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.dtype == np.int16:
            samples = arr.astype(np.float32) / 32768.0
        elif arr.dtype == np.int32:
            samples = arr.astype(np.float32) / 2147483648.0
        elif arr.dtype == np.uint8:
            samples = (arr.astype(np.float32) - 128.0) / 128.0
        else:
            samples = arr.astype(np.float32)
        return samples, int(sr)


class SileroVadModel:
    _CHUNK = 512  # silero-vad native chunk at 16 kHz
    _CONTEXT = 64  # silero v5 prepends the last 64 samples of the previous chunk
    _STATE_SHAPE = (2, 1, 128)

    def __init__(self) -> None:
        import onnxruntime as ort

        # Use the bundled 16kHz model directly; bypass silero_vad's Python wrapper
        # which forces a torchaudio import + CUDA runtime libs we don't need.
        model_path = (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / ".venv"
            / "lib"
            / "python3.12"
            / "site-packages"
            / "silero_vad"
            / "data"
            / "silero_vad.onnx"
        )
        if not model_path.exists():
            # fallback to whatever onnx is bundled
            import silero_vad as _sv  # noqa: F401 — only to fail with clean msg if pkg absent

            raise FileNotFoundError(f"silero_vad ONNX not found at {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def compute_probs(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(round(samples.size * ratio))
            xp = np.linspace(0.0, samples.size, num=samples.size, endpoint=False)
            x = np.linspace(0.0, samples.size, num=new_len, endpoint=False)
            samples = np.interp(x, xp, samples).astype(np.float32)
            sample_rate = 16000

        samples = np.asarray(samples, dtype=np.float32)
        n_chunks = samples.size // self._CHUNK
        if n_chunks == 0:
            return np.zeros(0, dtype=np.float32)

        out = np.empty(n_chunks, dtype=np.float32)
        state = np.zeros(self._STATE_SHAPE, dtype=np.float32)
        context = np.zeros((1, self._CONTEXT), dtype=np.float32)
        sr_arr = np.array(sample_rate, dtype=np.int64)
        for i in range(n_chunks):
            chunk = samples[i * self._CHUNK : (i + 1) * self._CHUNK].reshape(1, -1)
            inp = np.concatenate([context, chunk], axis=1).astype(np.float32)
            result = self._session.run(
                None,
                {"input": inp, "state": state, "sr": sr_arr},
            )
            out[i] = float(result[0][0, 0])
            state = result[1]
            context = inp[:, -self._CONTEXT:]
        return out
