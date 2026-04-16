from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass(slots=True)
class PalmDetection:
    confidence: float
    hand_present: bool


class MediapipeHandAnalyzer:
    def __init__(
        self,
        model_path: str | Path = "assets/models/hand_landmarker.task",
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        model_path = str(model_path)

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def analyze_raised_palm(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> PalmDetection:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return PalmDetection(confidence=0.0, hand_present=False)

        lm = result.hand_landmarks[0]

        wrist = lm[0]
        tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
        pips = [lm[3], lm[6], lm[10], lm[14], lm[18]]

        fingers_extended = 0
        scores: list[float] = []

        # Index, middle, ring, pinky
        for tip, pip in zip(tips[1:], pips[1:]):
            vertical = max(0.0, pip.y - tip.y)
            scores.append(min(1.0, vertical * 6.0))
            if tip.y < pip.y:
                fingers_extended += 1

        # Thumb openness relative to wrist
        thumb = tips[0]
        thumb_open = abs(thumb.x - wrist.x)
        scores.append(min(1.0, thumb_open * 4.0))

        confidence = sum(scores) / len(scores)
        if fingers_extended >= 3:
            confidence = max(confidence, 0.8)

        return PalmDetection(confidence=float(min(1.0, confidence)), hand_present=True)

    def close(self) -> None:
        self._landmarker.close()