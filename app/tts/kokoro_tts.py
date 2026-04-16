from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download


class KokoroTTS:
    def __init__(self, model_dir: str, repo_id: str, voice: str = "af_heart", lang_code: str = "a", speed: float = 1.0) -> None:
        self.model_dir = Path(model_dir)
        self.repo_id = repo_id
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self._pipeline = None
        self._ready = False
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._ready and self._pipeline is not None:
            return

        with self._lock:
            if self._ready and self._pipeline is not None:
                return

            self.model_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.model_dir.parent))

            config_file = self.model_dir / "config.json"
            weights_file = self.model_dir / "kokoro-v1_0.pth"
            voice_file = self.model_dir / "voices" / f"{self.voice}.pt"

            if not (config_file.exists() and weights_file.exists() and voice_file.exists()):
                snapshot_download(
                    repo_id=self.repo_id,
                    local_dir=str(self.model_dir),
                )

            if self._pipeline is None:
                from kokoro import KPipeline
                self._pipeline = KPipeline(lang_code=self.lang_code)
            self._ready = True

    def synthesize_pcm16(self, text: str) -> bytes:
        text = (text or "").strip().replace('*','')
        if not text:
            return b""
        if not self._ready or self._pipeline is None:
            self.ensure_ready()
        assert self._pipeline is not None

        chunks: list[np.ndarray] = []
        generator = self._pipeline(text, voice=self.voice, speed=self.speed)
        for _, _, audio in generator:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.size:
                chunks.append(arr)

        if not chunks:
            return b""

        merged = np.concatenate(chunks)
        merged = np.clip(merged, -1.0, 1.0)
        pcm16 = (merged * 32767.0).astype(np.int16)
        return pcm16.tobytes()
