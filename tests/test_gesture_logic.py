from types import SimpleNamespace

from app.core.models import Recipe, RecipeBranch, RecipeStep
from app.core.state import AppStateStore
from app.gestures.mediapipe_adapter import MediapipeGestureAnalyzer, GestureTuning
from app.realtime.prompts import build_system_prompt


def _make_landmarks(index=True, middle=False, ring=False, pinky=False, thumb=False):
    pts = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
    pts[0] = SimpleNamespace(x=0.5, y=0.9)
    pts[2] = SimpleNamespace(x=0.40, y=0.70)
    pts[3] = SimpleNamespace(x=0.42, y=0.68)
    pts[4] = SimpleNamespace(x=0.52 if thumb else 0.43, y=0.66)
    for tip_id, pip_id, is_up in [(8, 6, index), (12, 10, middle), (16, 14, ring), (20, 18, pinky)]:
        pts[pip_id] = SimpleNamespace(x=0.5, y=0.6)
        pts[tip_id] = SimpleNamespace(x=0.5, y=0.4 if is_up else 0.7)
    return pts


def _make_thumbs_up_landmarks():
    pts = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
    pts[0] = SimpleNamespace(x=0.52, y=0.90)
    pts[2] = SimpleNamespace(x=0.48, y=0.78)
    pts[3] = SimpleNamespace(x=0.50, y=0.62)
    pts[4] = SimpleNamespace(x=0.53, y=0.40)
    for pip_id, tip_id in [(6, 8), (10, 12), (14, 16), (18, 20)]:
        pts[pip_id] = SimpleNamespace(x=0.56, y=0.60)
        pts[tip_id] = SimpleNamespace(x=0.57, y=0.72)
    return pts


def _make_pinky_up_landmarks():
    pts = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
    pts[0] = SimpleNamespace(x=0.5, y=0.90)
    pts[2] = SimpleNamespace(x=0.46, y=0.73)
    pts[3] = SimpleNamespace(x=0.48, y=0.74)
    pts[4] = SimpleNamespace(x=0.49, y=0.75)
    for pip_id, tip_id, x in [(6, 8, 0.46), (10, 12, 0.50), (14, 16, 0.54)]:
        pts[pip_id] = SimpleNamespace(x=x, y=0.64)
        pts[tip_id] = SimpleNamespace(x=x, y=0.74)
    pts[18] = SimpleNamespace(x=0.62, y=0.66)
    pts[20] = SimpleNamespace(x=0.68, y=0.46)
    pts[5] = SimpleNamespace(x=0.44, y=0.70)
    return pts


def _make_fist_landmarks():
    pts = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
    pts[0] = SimpleNamespace(x=0.50, y=0.90)
    pts[5] = SimpleNamespace(x=0.44, y=0.74)
    pts[9] = SimpleNamespace(x=0.50, y=0.72)
    pts[13] = SimpleNamespace(x=0.56, y=0.74)
    pts[17] = SimpleNamespace(x=0.60, y=0.80)
    pts[2] = SimpleNamespace(x=0.46, y=0.72)
    pts[3] = SimpleNamespace(x=0.47, y=0.73)
    pts[4] = SimpleNamespace(x=0.47, y=0.71)
    for pip_id, tip_id, x in [(6, 8, 0.45), (10, 12, 0.49), (14, 16, 0.53), (18, 20, 0.57)]:
        pts[pip_id] = SimpleNamespace(x=x, y=0.68)
        pts[tip_id] = SimpleNamespace(x=x, y=0.73)
    return pts


def test_finger_count_detects_numbered_options() -> None:
    count, pattern, _ = MediapipeGestureAnalyzer._count_extended_fingers(_make_landmarks(index=True), "Right")
    assert count == 1
    assert pattern == (1, 0, 0, 0)

    count, pattern, _ = MediapipeGestureAnalyzer._count_extended_fingers(
        _make_landmarks(index=True, middle=True), "Right"
    )
    assert count == 2
    assert pattern == (1, 1, 0, 0)

    count, pattern, _ = MediapipeGestureAnalyzer._count_extended_fingers(
        _make_landmarks(index=True, middle=True, ring=True), "Right"
    )
    assert count == 3
    assert pattern == (1, 1, 1, 0)


def test_hand_navigation_pose_helpers() -> None:
    analyzer = MediapipeGestureAnalyzer.__new__(MediapipeGestureAnalyzer)
    analyzer._tuning = GestureTuning()
    thumb_conf = analyzer._thumbs_up_confidence(_make_thumbs_up_landmarks(), "Right", 0.5)
    pinky_conf = analyzer._pinky_up_confidence(_make_pinky_up_landmarks(), 0.5)
    fist_conf = analyzer._closed_fist_confidence(_make_fist_landmarks(), _make_fist_landmarks()[0], 0.5, 0)
    assert thumb_conf > 0.75
    assert pinky_conf > 0.75
    assert fist_conf > 0.75


def test_mouth_cover_prefers_face_overlap() -> None:
    analyzer = MediapipeGestureAnalyzer.__new__(MediapipeGestureAnalyzer)
    analyzer._tuning = GestureTuning()
    hand = {
        "bbox": (0.42, 0.44, 0.60, 0.66),
        "palm_center": SimpleNamespace(x=0.51, y=0.56),
    }
    face = {
        "bbox": (0.30, 0.20, 0.70, 0.80),
        "width": 0.40,
        "height": 0.60,
        "keypoints": {
            "left_eye": SimpleNamespace(x=0.44, y=0.36),
            "right_eye": SimpleNamespace(x=0.56, y=0.36),
            "nose_tip": SimpleNamespace(x=0.50, y=0.48),
            "mouth_center": SimpleNamespace(x=0.50, y=0.56),
        },
    }
    candidates = analyzer._classify_mouth_cover_gestures([hand], face)
    assert candidates
    assert candidates[0].name == "mouth_cover_toggle_speech"
    assert candidates[0].confidence >= 0.80


def test_previous_step_moves_back_one_step() -> None:
    recipe = Recipe(
        recipe_id="r1",
        name="Test",
        description="Test recipe",
        steps=[
            RecipeStep(step_id="s1", title="Start", instruction="Do the first thing"),
            RecipeStep(step_id="s2", title="Middle", instruction="Do the second thing"),
        ],
    )
    store = AppStateStore(recipe)
    store.advance_step()
    result = store.previous_step()
    assert result["moved_back"] is True
    assert result["current_step"]["step_id"] == "s1"


def test_prompt_requires_numbered_branch_options() -> None:
    recipe = Recipe(
        recipe_id="r1",
        name="Test",
        description="Test recipe",
        steps=[RecipeStep(step_id="s1", title="Start", instruction="Do the thing")],
        branches=[RecipeBranch(branch_id="soft", label="Soft", step_ids=["s1"])],
    )
    prompt = build_system_prompt(recipe)
    assert "number them explicitly as 1, 2, 3" in prompt
    assert "Keep branch options to at most three choices" in prompt
    assert "synthetic gesture messages for next step, previous step, repeat step" in prompt
