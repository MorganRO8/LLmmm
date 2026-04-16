from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd


class PlaybackController:
    """
    Handles streaming mono PCM16 audio playback with immediate interrupt support.

    Important behavior:
    - Incoming audio chunks are appended into a continuous byte buffer.
    - The callback consumes exactly the number of bytes needed for each audio frame.
    - Any remainder stays in the buffer for the next callback.
    - Playback waits for a short prebuffer before starting to reduce jitter.
    """

    def __init__(
        self,
        samplerate: int = 24000,
        channels: int = 1,
        blocksize: int | None = None,
        prebuffer_ms: int = 120,
    ) -> None:
        self.sample_rate = samplerate
        self.channels = channels
        self.blocksize = 0 if blocksize in (None, 0) else blocksize
        self.prebuffer_ms = prebuffer_ms

        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()

        self._buffer = bytearray()
        self._playing = False
        self._started_playback = False

        self._recent_output_rms = 0.0
        self._last_output_ts = 0.0

    def _min_prebuffer_bytes(self) -> int:
        samples = int(self.sample_rate * (self.prebuffer_ms / 1000.0)) * self.channels
        return max(0, samples * 2)  # int16 = 2 bytes

    def start(self) -> None:
        if self._stream is not None:
            return

        def callback(outdata, frames, time_info, status) -> None:
            needed_samples = frames * self.channels
            needed_bytes = needed_samples * 2

            with self._lock:
                if self._stop_flag.is_set():
                    outdata.fill(0)
                    return

                if not self._started_playback:
                    if len(self._buffer) >= self._min_prebuffer_bytes():
                        self._started_playback = True
                    else:
                        outdata.fill(0)
                        self._playing = bool(self._buffer)
                        return

                if len(self._buffer) == 0:
                    outdata.fill(0)
                    self._playing = False
                    self._started_playback = False
                    self._recent_output_rms = 0.0
                    return

                chunk = bytes(self._buffer[:needed_bytes])
                del self._buffer[: min(needed_bytes, len(self._buffer))]

                audio = np.frombuffer(chunk, dtype=np.int16)
                if audio.size < needed_samples:
                    padded = np.zeros(needed_samples, dtype=np.int16)
                    padded[: audio.size] = audio
                    audio = padded

                outdata[:] = audio.reshape(frames, self.channels)
                self._playing = True

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.blocksize,
            callback=callback,
        )
        self._stream.start()

    def play_chunk(self, pcm16_bytes: bytes) -> None:
        if not pcm16_bytes:
            return

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

        with self._lock:
            if self._stop_flag.is_set():
                return
            self._buffer.extend(pcm16_bytes)
            self._playing = True
            self._recent_output_rms = rms
            self._last_output_ts = time.monotonic()

    def interrupt(self) -> None:
        """
        Immediately stop playback and clear buffered audio.
        """
        with self._lock:
            self._stop_flag.set()
            self._buffer.clear()
            self._playing = False
            self._started_playback = False
            self._recent_output_rms = 0.0
            self._stop_flag.clear()

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._buffer.clear()
            self._started_playback = False
            self._recent_output_rms = 0.0

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def is_playing(self) -> bool:
        return self._playing

    def get_recent_output_rms(self) -> float:
        return self._recent_output_rms

    def get_last_output_ts(self) -> float:
        return self._last_output_ts
