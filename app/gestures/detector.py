from __future__ import annotations

import time

from app.gestures.hold_gate import GestureCandidate, GestureEvent, HoldGate
from app.gestures.mediapipe_adapter import GestureTuning, MediapipeGestureAnalyzer


class MultiGestureDetector:
    def __init__(
        self,
        confidence_threshold: float,
        hold_ms: int,
        cooldown_ms: int,
        hand_model_path: str = "assets/models/hand_landmarker.task",
        face_model_path: str = "assets/models/face_detector.tflite",
        palm_threshold: float = 0.78,
        mouth_threshold: float = 0.6,
        thumbs_up_threshold: float = 0.98,
        pinky_threshold: float = 0.78,
        fist_threshold: float = 0.82,
        option_threshold: float = 0.80,
        tuning: GestureTuning | None = None,
    ) -> None:
        self._analyzer = MediapipeGestureAnalyzer(
            hand_model_path=hand_model_path,
            face_model_path=face_model_path,
            tuning=tuning,
        )
        self._gates = {
            "raised_palm_interrupt": HoldGate(
                target_name="raised_palm_interrupt",
                confidence_threshold=max(confidence_threshold, palm_threshold),
                hold_ms=min(3000, hold_ms),
                cooldown_ms=cooldown_ms,
            ),
            "mouth_cover_toggle_speech": HoldGate(
                target_name="mouth_cover_toggle_speech",
                confidence_threshold=mouth_threshold,
                hold_ms=min(1500, hold_ms),
                cooldown_ms=cooldown_ms,
            ),
            "thumbs_up_next_step": HoldGate("thumbs_up_next_step", thumbs_up_threshold, min(3000, hold_ms), 1200),
            "pinky_up_previous_step": HoldGate("pinky_up_previous_step", pinky_threshold, min(1200, hold_ms), 1200),
            "fist_repeat_step": HoldGate("fist_repeat_step", fist_threshold, min(1200, hold_ms), 1200),
            "option_choice_1": HoldGate("option_choice_1", option_threshold, min(1200, hold_ms), 1200),
            "option_choice_2": HoldGate("option_choice_2", option_threshold, min(1200, hold_ms), 1200),
            "option_choice_3": HoldGate("option_choice_3", option_threshold, min(1200, hold_ms), 1200),
        }

    def process(self, frame) -> list[GestureEvent]:  # noqa: ANN001
        timestamp_s = time.monotonic()
        candidates = [
            GestureCandidate(name=item.name, confidence=item.confidence, timestamp=timestamp_s)
            for item in self._analyzer.analyze(frame, int(timestamp_s * 1000))
        ]
        events: list[GestureEvent] = []
        for gate in self._gates.values():
            events.extend(gate.update(candidates, timestamp=timestamp_s))
        return events

    def close(self) -> None:
        self._analyzer.close()
