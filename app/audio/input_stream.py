from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioInputStream:
    def __init__(self, samplerate: int, channels: int, blocksize: int, callback: Callable[[bytes], None], queue_maxsize: int = 64) -> None:
        self._samplerate = samplerate
        self._channels = channels
        self._blocksize = blocksize
        self._callback = callback
        self._queue_maxsize = max(8, queue_maxsize)
        self._stream: sd.RawInputStream | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=self._queue_maxsize)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stop_event.clear()
        self._worker = threading.Thread(target=self._drain_queue, name="audio-input-worker", daemon=True)
        self._worker.start()

        def _on_audio(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                logger.warning("Audio input status: %s", status)
            try:
                self._queue.put_nowait(bytes(indata))
            except queue.Full:
                logger.warning("Audio input queue full; dropping input chunk")

        self._stream = sd.RawInputStream(
            samplerate=self._samplerate,
            channels=self._channels,
            dtype="int16",
            blocksize=self._blocksize,
            callback=_on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        self._drain_remaining()

    def _drain_queue(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                self._callback(item)
            except Exception:
                logger.exception("Audio input callback failed")

    def _drain_remaining(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
