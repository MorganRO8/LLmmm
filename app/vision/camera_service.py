from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from app.vision.frame_buffer import BufferedFrame, FrameBuffer


class CameraService:
    def __init__(
        self,
        camera_index: int,
        preview_width: int,
        preview_height: int,
        context_width: int,
        context_height: int,
        context_sample_interval_ms: int,
        context_buffer_maxlen: int,
    ) -> None:
        self._camera_index = camera_index
        self._preview_width = preview_width
        self._preview_height = preview_height
        self._context_width = context_width
        self._context_height = context_height
        self._context_sample_interval_s = context_sample_interval_ms / 1000.0
        self._frame_buffer = FrameBuffer(maxlen=context_buffer_maxlen)
        self._lock = threading.Lock()
        self._latest_preview: np.ndarray | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._last_context_frame: np.ndarray | None = None
        self._last_sample_ts = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="camera-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def get_latest_preview_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_preview is None else self._latest_preview.copy()

    def get_context_frames(self, count: int = 3, max_age_seconds: float = 12.0) -> list[BufferedFrame]:
        return self._frame_buffer.get_recent(count=count, max_age_seconds=max_age_seconds)

    def _run(self) -> None:
        capture = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._preview_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._preview_height)
        self._capture = capture
        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest_preview = frame.copy()

            now = time.time()
            if now - self._last_sample_ts >= self._context_sample_interval_s:
                self._last_sample_ts = now
                context = cv2.resize(frame, (self._context_width, self._context_height), interpolation=cv2.INTER_AREA)
                motion = self._estimate_motion(context)
                self._frame_buffer.add_frame(context, motion_score=motion)
                self._last_context_frame = context

        capture.release()
        self._capture = None

    def _estimate_motion(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_context_frame is None:
            return 0.0
        prev_gray = cv2.cvtColor(self._last_context_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        return float(diff.mean())
