from app.gestures.hold_gate import GestureCandidate, HoldGate


def test_hold_gate_requires_full_hold_duration() -> None:
    gate = HoldGate("raised_palm_interrupt", confidence_threshold=0.98, hold_ms=3000, cooldown_ms=4000)
    assert gate.update([GestureCandidate("raised_palm_interrupt", 0.99, 0.0)], timestamp=0.0) == []
    assert gate.update([GestureCandidate("raised_palm_interrupt", 0.99, 1.5)], timestamp=1.5) == []
    events = gate.update([GestureCandidate("raised_palm_interrupt", 0.99, 3.1)], timestamp=3.1)
    assert len(events) == 1
    assert events[0].held_for_ms >= 3000
