from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import warnings

import cv2
try:
    import mediapipe as mp
except ModuleNotFoundError:  # pragma: no cover - test env may not have mediapipe
    mp = None
import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GestureCandidateResult:
    name: str
    confidence: float


@dataclass(slots=True)
class GestureTuning:
    mouth_box_expand: float = 0.02
    mouth_distance_factor: float = 0.20
    mouth_min_overlap_ratio: float = 0.03
    pinky_min_extension: float = 0.05
    pinky_min_separation: float = 0.025
    fist_max_avg_tip_distance_ratio: float = 0.58
    fist_max_bbox_height_ratio: float = 1.50
    debug_logging: bool = False


@dataclass(slots=True)
class _Point:
    x: float
    y: float


class MediapipeGestureAnalyzer:
    def __init__(
        self,
        hand_model_path: str | Path = "assets/models/hand_landmarker.task",
        face_model_path: str | Path = "assets/models/face_detector.tflite",
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.65,
        min_hand_presence_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
        tuning: GestureTuning | None = None,
    ) -> None:
        if mp is None:
            raise RuntimeError("mediapipe is required for gesture analysis at runtime")
        if not hasattr(mp, "tasks"):
            raise RuntimeError("This gesture stack requires a MediaPipe build with the Tasks API available")

        self._tuning = tuning or GestureTuning()
        self._hand_model_path = Path(hand_model_path)
        self._face_model_path = self._resolve_face_model_path(face_model_path)
        self._hands = self._create_hand_landmarker(
            model_path=self._hand_model_path,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._face = self._create_face_detector(
            model_path=self._face_model_path,
            min_detection_confidence=min_detection_confidence,
        )

    def _resolve_face_model_path(self, configured_path: str | Path) -> Path | None:
        raw_path = Path(configured_path)
        candidates: list[Path] = []

        if str(configured_path).strip():
            candidates.append(raw_path)
            if raw_path.suffix:
                candidates.append(raw_path.with_suffix('.task'))
                candidates.append(raw_path.with_suffix('.tflite'))
            else:
                candidates.append(raw_path.with_suffix('.task'))
                candidates.append(raw_path.with_suffix('.tflite'))

        assets_dir = raw_path.parent if raw_path.parent != Path('.') else Path('assets/models')
        common_names = [
            'face_detector.task',
            'face_detector.tflite',
            'blaze_face_short_range.tflite',
            'blaze_face_full_range.tflite',
            'blaze_face_short_range.task',
            'blaze_face_full_range.task',
            'blaze_face_sparse_full_range.tflite',
        ]
        candidates.extend(assets_dir / name for name in common_names)

        seen: set[Path] = set()
        for candidate in candidates:
            candidate = candidate.expanduser()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists() and candidate.is_file():
                logger.info('Using MediaPipe face model at %s', candidate)
                return candidate

        searched = ', '.join(str(p) for p in list(seen)[:8])
        warnings.warn(
            (
                f"MediaPipe face model not found. Looked for: {searched}. "
                'Mouth-cover gesture will be disabled until a compatible face detector model file is added.'
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def _create_hand_landmarker(
        self,
        model_path: Path,
        max_num_hands: int,
        min_detection_confidence: float,
        min_hand_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> Any:
        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe hand model not found at '{model_path}'. "
                "Place the Hand Landmarker task file there or set a different path in code/config."
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _create_face_detector(self, model_path: Path | None, min_detection_confidence: float) -> Any | None:
        if model_path is None:
            return None
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=min_detection_confidence,
        )
        return mp.tasks.vision.FaceDetector.create_from_options(options)

    def analyze(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[GestureCandidateResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self._hands.detect_for_video(mp_image, timestamp_ms)
        face_result = self._face.detect_for_video(mp_image, timestamp_ms) if self._face is not None else None

        hands = self._extract_hands(hand_result)
        if not hands:
            return []
        face = self._extract_face(face_result, frame_bgr.shape[1], frame_bgr.shape[0])

        option_candidates = self._classify_option_gestures(hands)
        if option_candidates:
            self._debug('option', option_candidates)
            return option_candidates

        mouth_candidates = self._classify_mouth_cover_gestures(hands, face)
        if mouth_candidates:
            self._debug('mouth', mouth_candidates)
            return mouth_candidates

        nav_candidates = self._classify_navigation_gestures(hands)
        if nav_candidates:
            self._debug('navigation', nav_candidates)
            return nav_candidates

        palm_candidates = self._classify_palm_gestures(hands)
        self._debug('palm', palm_candidates)
        return palm_candidates

    def _extract_hands(self, result: Any) -> list[dict[str, Any]]:
        landmarks_list = getattr(result, 'hand_landmarks', None) or []
        handedness_list = getattr(result, 'handedness', None) or []
        hands: list[dict[str, Any]] = []
        for idx, lm in enumerate(landmarks_list):
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            handedness = 'Right'
            if idx < len(handedness_list) and handedness_list[idx]:
                first = handedness_list[idx][0]
                handedness = getattr(first, 'category_name', None) or getattr(first, 'display_name', None) or handedness
            hands.append(
                {
                    'index': idx,
                    'landmarks': lm,
                    'handedness': handedness,
                    'bbox': (min(xs), min(ys), max(xs), max(ys)),
                    'wrist': self._point_from_landmark(lm[0]),
                    'palm_center': _Point(x=float((lm[0].x + lm[5].x + lm[9].x + lm[17].x) / 4.0), y=float((lm[0].y + lm[5].y + lm[9].y + lm[17].y) / 4.0)),
                }
            )
        return hands

    def _extract_face(self, result: Any, frame_width: int, frame_height: int) -> dict[str, Any] | None:
        if result is None:
            return None
        detections = getattr(result, 'detections', None) or []
        if not detections:
            return None
        detection = detections[0]
        bbox = detection.bounding_box
        x0 = float(bbox.origin_x) / max(frame_width, 1)
        y0 = float(bbox.origin_y) / max(frame_height, 1)
        x1 = float(bbox.origin_x + bbox.width) / max(frame_width, 1)
        y1 = float(bbox.origin_y + bbox.height) / max(frame_height, 1)
        keypoints = list(getattr(detection, 'keypoints', None) or [])
        if len(keypoints) < 4:
            return None
        mapped = {
            'right_eye': self._point_from_landmark(keypoints[0]),
            'left_eye': self._point_from_landmark(keypoints[1]),
            'nose_tip': self._point_from_landmark(keypoints[2]),
            'mouth_center': self._point_from_landmark(keypoints[3]),
        }
        return {
            'bbox': (x0, y0, x1, y1),
            'width': float(x1 - x0),
            'height': float(y1 - y0),
            'center': _Point(x=(x0 + x1) / 2.0, y=(y0 + y1) / 2.0),
            'keypoints': mapped,
        }

    def _classify_option_gestures(self, hands: list[dict[str, Any]]) -> list[GestureCandidateResult]:
        candidates: list[GestureCandidateResult] = []
        for hand in hands:
            count, pattern, finger_score = self._count_extended_fingers(hand['landmarks'], hand['handedness'])
            if pattern == (1, 0, 0, 0):
                candidates.append(GestureCandidateResult('option_choice_1', float(min(0.99, 0.82 + 0.15 * finger_score))))
            elif pattern == (1, 1, 0, 0):
                candidates.append(GestureCandidateResult('option_choice_2', float(min(0.99, 0.82 + 0.15 * finger_score))))
            elif pattern == (1, 1, 1, 0):
                candidates.append(GestureCandidateResult('option_choice_3', float(min(0.99, 0.82 + 0.15 * finger_score))))
        return candidates

    def _classify_mouth_cover_gestures(self, hands: list[dict[str, Any]], face: dict[str, Any] | None) -> list[GestureCandidateResult]:
        if face is None:
            return []
        kp = face['keypoints']
        mouth = kp['mouth_center']
        face_w = max(face['width'], 1e-6)
        candidates: list[GestureCandidateResult] = []
        for hand in hands:
            bbox = hand['bbox']
            palm = hand['palm_center']
            overlap = self._box_overlap_ratio(bbox, face['bbox'])
            if overlap < self._tuning.mouth_min_overlap_ratio:
                continue
            contains_mouth = self._point_in_expanded_box(mouth, bbox, expand=self._tuning.mouth_box_expand)
            palm_near = self._distance(palm, mouth) < face_w * self._tuning.mouth_distance_factor
            hand_below_eyes = palm.y >= min(kp['left_eye'].y, kp['right_eye'].y)
            if not hand_below_eyes:
                continue
            if contains_mouth or palm_near:
                conf = 0.76
                conf += min(0.12, overlap * 0.8)
                if contains_mouth:
                    conf += 0.10
                if palm_near:
                    conf += 0.08
                candidates.append(GestureCandidateResult('mouth_cover_toggle_speech', float(min(0.99, conf))))
        return candidates

    def _classify_navigation_gestures(self, hands: list[dict[str, Any]]) -> list[GestureCandidateResult]:
        candidates: list[GestureCandidateResult] = []
        for hand in hands:
            lm = hand['landmarks']
            handedness = hand['handedness']
            bbox = hand['bbox']
            hand_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1e-6)
            wrist = lm[0]
            count, _, finger_score = self._count_extended_fingers(lm, handedness)

            thumb_conf = self._thumbs_up_confidence(lm, handedness, hand_size)
            if thumb_conf > 0.0:
                candidates.append(GestureCandidateResult('thumbs_up_next_step', float(min(0.99, thumb_conf + 0.08 * finger_score))))
                continue

            pinky_conf = self._pinky_up_confidence(lm, hand_size)
            if pinky_conf > 0.0:
                candidates.append(GestureCandidateResult('pinky_up_previous_step', float(min(0.99, pinky_conf + 0.05 * finger_score))))
                continue

            fist_conf = self._closed_fist_confidence(lm, wrist, hand_size, count)
            if fist_conf > 0.0:
                candidates.append(GestureCandidateResult('fist_repeat_step', float(min(0.99, fist_conf))))
                continue
        return candidates

    def _classify_palm_gestures(self, hands: list[dict[str, Any]]) -> list[GestureCandidateResult]:
        candidates: list[GestureCandidateResult] = []
        for hand in hands:
            count, _, finger_score = self._count_extended_fingers(hand['landmarks'], hand['handedness'])
            bbox = hand['bbox']
            hand_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if count >= 4:
                confidence = min(0.99, 0.76 + 0.18 * finger_score + 0.10 * hand_size)
                candidates.append(GestureCandidateResult('raised_palm_interrupt', float(confidence)))
        return candidates

    def _thumbs_up_confidence(self, landmarks: list[Any], handedness: str, hand_size: float) -> float:
        index_ext, middle_ext, ring_ext, pinky_ext = self._extended_pattern(landmarks)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        thumb_vertical = wrist.y - thumb_tip.y
        thumb_chain = thumb_tip.y < thumb_ip.y < thumb_mcp.y < wrist.y
        others_folded = sum([index_ext, middle_ext, ring_ext, pinky_ext]) == 0
        lateral = abs(thumb_tip.x - thumb_ip.x)
        if handedness.lower() == 'right':
            away = thumb_tip.x >= thumb_ip.x - 0.02
        else:
            away = thumb_tip.x <= thumb_ip.x + 0.02
        if not (thumb_chain and others_folded and away):
            return 0.0
        return float(min(0.99, 0.78 + max(0.0, thumb_vertical) * 0.6 + lateral * 0.5 + min(0.08, hand_size * 0.2)))

    def _pinky_up_confidence(self, landmarks: list[Any], hand_size: float) -> float:
        index_ext, middle_ext, ring_ext, pinky_ext = self._extended_pattern(landmarks)
        if pinky_ext != 1 or any(v == 1 for v in (index_ext, middle_ext, ring_ext)):
            return 0.0
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        ring_tip = landmarks[16]
        pinky_extension = pinky_pip.y - pinky_tip.y
        pinky_separation = abs(pinky_tip.x - ring_tip.x)
        thumb_tip = landmarks[4]
        thumb_index_gap = self._distance(self._point_from_landmark(thumb_tip), self._point_from_landmark(landmarks[5]))
        if pinky_extension < self._tuning.pinky_min_extension:
            return 0.0
        if pinky_separation < self._tuning.pinky_min_separation:
            return 0.0
        conf = 0.76 + min(0.14, pinky_extension * 1.5) + min(0.08, pinky_separation * 1.2)
        if thumb_index_gap < hand_size * 0.32:
            conf += 0.04
        return float(min(0.99, conf))

    def _closed_fist_confidence(self, landmarks: list[Any], wrist: Any, hand_size: float, extended_count: int) -> float:
        index_ext, middle_ext, ring_ext, pinky_ext = self._extended_pattern(landmarks)
        if any(v == 1 for v in (index_ext, middle_ext, ring_ext, pinky_ext)) or extended_count > 1:
            return 0.0
        palm = _Point(x=float((wrist.x + landmarks[5].x + landmarks[9].x + landmarks[17].x) / 4.0), y=float((wrist.y + landmarks[5].y + landmarks[9].y + landmarks[17].y) / 4.0))
        fingertip_ids = [8, 12, 16, 20]
        avg_tip_distance = sum(self._distance(self._point_from_landmark(landmarks[i]), palm) for i in fingertip_ids) / len(fingertip_ids)
        bbox_h = max(p.y for p in landmarks) - min(p.y for p in landmarks)
        bbox_w = max(p.x for p in landmarks) - min(p.x for p in landmarks)
        thumb_tip = self._point_from_landmark(landmarks[4])
        thumb_target = self._point_from_landmark(landmarks[6])
        thumb_distance = self._distance(thumb_tip, thumb_target)
        compact = avg_tip_distance / max(hand_size, 1e-6)
        if compact > self._tuning.fist_max_avg_tip_distance_ratio:
            return 0.0
        if bbox_h / max(bbox_w, 1e-6) > self._tuning.fist_max_bbox_height_ratio:
            return 0.0
        conf = 0.78 + max(0.0, (self._tuning.fist_max_avg_tip_distance_ratio - compact)) * 0.5
        if thumb_distance < hand_size * 0.36:
            conf += 0.06
        return float(min(0.99, conf))

    @staticmethod
    def _point_from_landmark(landmark: Any) -> _Point:
        return _Point(x=float(landmark.x), y=float(landmark.y))

    @staticmethod
    def _distance(a: _Point, b: _Point) -> float:
        return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)

    @staticmethod
    def _extended_pattern(landmarks: list[Any]) -> tuple[int, int, int, int]:
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        extended: list[int] = []
        for tip_id, pip_id in zip(tip_ids, pip_ids):
            tip = landmarks[tip_id]
            pip = landmarks[pip_id]
            extended.append(1 if (pip.y - tip.y) > 0.025 else 0)
        return tuple(extended)  # type: ignore[return-value]

    @staticmethod
    def _count_extended_fingers(landmarks: list[Any], handedness: str) -> tuple[int, tuple[int, int, int, int], float]:
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        extended: list[int] = []
        scores: list[float] = []
        for tip_id, pip_id in zip(tip_ids, pip_ids):
            tip = landmarks[tip_id]
            pip = landmarks[pip_id]
            delta = pip.y - tip.y
            score = max(0.0, min(1.0, delta * 7.0))
            scores.append(score)
            extended.append(1 if delta > 0.025 else 0)

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        if handedness.lower() == 'right':
            thumb_extended = int((thumb_tip.x - thumb_ip.x) > 0.02 and (thumb_ip.x - thumb_mcp.x) > -0.01)
        else:
            thumb_extended = int((thumb_ip.x - thumb_tip.x) > 0.02 and (thumb_mcp.x - thumb_ip.x) > -0.01)
        count = int(sum(extended) + thumb_extended)
        finger_score = float(sum(scores) / max(len(scores), 1))
        return count, tuple(extended), finger_score

    @staticmethod
    def _point_in_expanded_box(point: _Point, bbox: tuple[float, float, float, float], expand: float) -> bool:
        x0, y0, x1, y1 = bbox
        return (x0 - expand) <= point.x <= (x1 + expand) and (y0 - expand) <= point.y <= (y1 + expand)

    @staticmethod
    def _box_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        x0 = max(a[0], b[0])
        y0 = max(a[1], b[1])
        x1 = min(a[2], b[2])
        y1 = min(a[3], b[3])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
        area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
        return float(inter / min(area_a, area_b))

    def _debug(self, category: str, candidates: list[GestureCandidateResult]) -> None:
        if not self._tuning.debug_logging or not candidates:
            return
        logger.info('Gesture candidates (%s): %s', category, ', '.join(f'{c.name}={c.confidence:.2f}' for c in candidates))

    def close(self) -> None:
        self._hands.close()
        if self._face is not None:
            self._face.close()
