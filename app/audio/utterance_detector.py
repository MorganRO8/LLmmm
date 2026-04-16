from __future__ import annotations

import time
from collections import deque

import numpy as np


class UtteranceDetector:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        vad_threshold: float = 900.0,
        min_speech_ms: int = 300,
        silence_ms: int = 700,
        max_utterance_ms: int = 10000,
        preroll_ms: int = 250,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad_threshold = vad_threshold
        self.min_speech_ms = min_speech_ms
        self.silence_ms = silence_ms
        self.max_utterance_ms = max_utterance_ms
        self.preroll_ms = preroll_ms

        self._recording = False
        self._buffer = bytearray()
        self._speech_started_at = 0.0
        self._last_voiced_at = 0.0
        self._preroll: deque[bytes] = deque()
        self._preroll_bytes = 0
        self._max_preroll_bytes = int((sample_rate * channels * 2) * (preroll_ms / 1000.0))

    @staticmethod
    def _rms_int16(audio_bytes: bytes) -> float:
        if not audio_bytes:
            return 0.0
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio))))

    def reset(self) -> None:
        self._recording = False
        self._buffer.clear()
        self._speech_started_at = 0.0
        self._last_voiced_at = 0.0
        self._preroll.clear()
        self._preroll_bytes = 0

    def _append_preroll(self, audio_bytes: bytes) -> None:
        self._preroll.append(audio_bytes)
        self._preroll_bytes += len(audio_bytes)
        while self._preroll_bytes > self._max_preroll_bytes and self._preroll:
            removed = self._preroll.popleft()
            self._preroll_bytes -= len(removed)

    def process_chunk(self, audio_bytes: bytes, blocked: bool = False) -> bytes | None:
        now = time.monotonic()
        if blocked:
            self.reset()
            return None

        rms = self._rms_int16(audio_bytes)
        voiced = rms >= self.vad_threshold
        self._append_preroll(audio_bytes)

        if not self._recording:
            if voiced:
                self._recording = True
                self._speech_started_at = now
                self._last_voiced_at = now
                self._buffer = bytearray().join(self._preroll)
                self._buffer.extend(audio_bytes)
            return None

        self._buffer.extend(audio_bytes)
        if voiced:
            self._last_voiced_at = now

        duration_ms = (now - self._speech_started_at) * 1000
        silence_ms = (now - self._last_voiced_at) * 1000

        if duration_ms >= self.max_utterance_ms:
            utterance = bytes(self._buffer)
            self.reset()
            return utterance

        if silence_ms >= self.silence_ms:
            if duration_ms >= self.min_speech_ms:
                utterance = bytes(self._buffer)
                self.reset()
                return utterance
            self.reset()

        return None
