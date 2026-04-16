from __future__ import annotations

import time
import uuid
from collections import deque

import cv2
import numpy as np
from pydantic import BaseModel


class BufferedFrame(BaseModel):
    frame_id: str
    timestamp: float
    width: int
    height: int
    jpeg_bytes: bytes
    motion_score: float | None = None


class FrameBuffer:
    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._frames: deque[BufferedFrame] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    def add_frame(self, frame: np.ndarray, motion_score: float | None = None, jpeg_quality: int = 70) -> BufferedFrame:
        if frame.ndim != 3:
            raise ValueError("expected a color image with 3 dimensions")

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        ok, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame")

        height, width = frame.shape[:2]
        buffered = BufferedFrame(
            frame_id=uuid.uuid4().hex,
            timestamp=time.time(),
            width=width,
            height=height,
            jpeg_bytes=encoded.tobytes(),
            motion_score=motion_score,
        )
        self._frames.append(buffered)
        return buffered

    def latest(self) -> BufferedFrame | None:
        return self._frames[-1] if self._frames else None

    def get_recent(self, count: int = 3, max_age_seconds: float = 12.0) -> list[BufferedFrame]:
        now = time.time()
        frames = [f for f in self._frames if now - f.timestamp <= max_age_seconds]
        if len(frames) <= count:
            return list(frames)

        if count == 1:
            return [frames[-1]]

        indices = np.linspace(0, len(frames) - 1, num=count, dtype=int)
        return [frames[i] for i in indices]
