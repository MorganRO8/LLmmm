from __future__ import annotations

import time

from app.gestures.hold_gate import GestureCandidate, GestureEvent, HoldGate
from app.gestures.mediapipe_adapter import MediapipeHandAnalyzer


class RaisedPalmDetector:
    def __init__(self, confidence_threshold: float, hold_ms: int, cooldown_ms: int) -> None:
        self._analyzer = MediapipeHandAnalyzer()
        self._gate = HoldGate(
            target_name="raised_palm_interrupt",
            confidence_threshold=confidence_threshold,
            hold_ms=hold_ms,
            cooldown_ms=cooldown_ms,
        )

    def process(self, frame) -> list[GestureEvent]:  # noqa: ANN001
        timestamp_ms = int(time.monotonic() * 1000)
        result = self._analyzer.analyze_raised_palm(frame, timestamp_ms)
        candidates = []
        if result.hand_present:
            candidates.append(GestureCandidate(name="raised_palm_interrupt", confidence=result.confidence, timestamp=timestamp_ms))
        return self._gate.update(candidates, timestamp=timestamp_ms)

    def close(self) -> None:
        self._analyzer.close()
