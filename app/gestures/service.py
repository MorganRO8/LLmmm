from __future__ import annotations

import threading
import time
from collections.abc import Callable

from app.gestures.detector import MultiGestureDetector
from app.gestures.mediapipe_adapter import GestureTuning
from app.vision.camera_service import CameraService


class GestureService:
    def __init__(
        self,
        camera_service: CameraService,
        confidence_threshold: float,
        hold_ms: int,
        cooldown_ms: int,
        poll_interval_ms: int,
        on_gesture: Callable[[str, float, int], None],
        hand_model_path: str = "assets/models/hand_landmarker.task",
        face_model_path: str = "assets/models/face_detector.tflite",
        palm_threshold: float = 0.78,
        mouth_threshold: float = 0.76,
        thumbs_up_threshold: float = 0.82,
        pinky_threshold: float = 0.78,
        fist_threshold: float = 0.82,
        option_threshold: float = 0.80,
        tuning: GestureTuning | None = None,
    ) -> None:
        self._camera_service = camera_service
        self._detector = MultiGestureDetector(
            confidence_threshold,
            hold_ms,
            cooldown_ms,
            hand_model_path=hand_model_path,
            face_model_path=face_model_path,
            palm_threshold=palm_threshold,
            mouth_threshold=mouth_threshold,
            thumbs_up_threshold=thumbs_up_threshold,
            pinky_threshold=pinky_threshold,
            fist_threshold=fist_threshold,
            option_threshold=option_threshold,
            tuning=tuning,
        )
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._on_gesture = on_gesture
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._enabled = True

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="gesture-service", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._detector.close()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._camera_service.get_latest_preview_frame()
            if frame is not None and self._enabled:
                events = self._detector.process(frame)
                for event in events:
                    self._on_gesture(event.name, event.confidence, event.held_for_ms)
            time.sleep(self._poll_interval_s)
