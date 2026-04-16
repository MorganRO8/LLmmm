from __future__ import annotations

import numpy as np

from app.vision.frame_buffer import FrameBuffer


def test_frame_buffer_keeps_maxlen() -> None:
    buffer = FrameBuffer(maxlen=2)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    buffer.add_frame(frame)
    buffer.add_frame(frame)
    buffer.add_frame(frame)
    assert len(buffer) == 2


def test_frame_buffer_get_recent() -> None:
    buffer = FrameBuffer(maxlen=5)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    for _ in range(4):
        buffer.add_frame(frame)
    recent = buffer.get_recent(count=2, max_age_seconds=100)
    assert len(recent) == 2
