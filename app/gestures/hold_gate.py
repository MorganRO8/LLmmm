from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GestureCandidate:
    name: str
    confidence: float
    timestamp: float


@dataclass(slots=True)
class GestureEvent:
    name: str
    confidence: float
    held_for_ms: int
    timestamp: float


class HoldGate:
    def __init__(self, target_name: str, confidence_threshold: float, hold_ms: int, cooldown_ms: int) -> None:
        self._target_name = target_name
        self._confidence_threshold = confidence_threshold
        self._hold_ms = hold_ms
        self._cooldown_ms = cooldown_ms
        self._armed_since: float | None = None
        self._cooldown_until: float = 0.0
        self._triggered_while_present = False

    def update(self, candidates: list[GestureCandidate], timestamp: float) -> list[GestureEvent]:
        if timestamp < self._cooldown_until:
            return []

        match = next((c for c in candidates if c.name == self._target_name and c.confidence >= self._confidence_threshold), None)
        if match is None:
            self._armed_since = None
            self._triggered_while_present = False
            return []

        if self._triggered_while_present:
            return []

        if self._armed_since is None:
            self._armed_since = timestamp
            return []

        held_for_ms = int((timestamp - self._armed_since) * 1000)
        if held_for_ms < self._hold_ms:
            return []

        self._triggered_while_present = True
        self._cooldown_until = timestamp + (self._cooldown_ms / 1000.0)
        return [GestureEvent(name=match.name, confidence=match.confidence, held_for_ms=held_for_ms, timestamp=timestamp)]
