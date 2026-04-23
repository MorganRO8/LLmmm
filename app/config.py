from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.constants import (
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DEFAULT_MODEL,
    DEFAULT_RECIPE_PATH,
)

load_dotenv()


class AppConfig(BaseModel):
    model_name: str = Field(default_factory=lambda: os.getenv("COOKING_ASSISTANT_MODEL", DEFAULT_MODEL))
    transcription_model: str = Field(default_factory=lambda: os.getenv("COOKING_ASSISTANT_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"))
    recipe_path: str = Field(default_factory=lambda: os.getenv("COOKING_ASSISTANT_RECIPE", DEFAULT_RECIPE_PATH))
    camera_index: int = int(os.getenv("COOKING_ASSISTANT_CAMERA_INDEX", "0"))
    log_level: str = os.getenv("COOKING_ASSISTANT_LOG_LEVEL", "INFO")

    preview_width: int = 640
    preview_height: int = 480
    context_frame_width: int = 320
    context_frame_height: int = 180
    context_sample_interval_ms: int = 1500
    context_buffer_seconds: int = 18

    gesture_confidence_threshold: float = 0.98
    gesture_hold_ms: int = 3000
    gesture_cooldown_ms: int = 4000
    gesture_poll_interval_ms: int = 100

    timer_warn_at_seconds: int = 10
    timer_notify_at_seconds: int = 0

    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    audio_channels: int = DEFAULT_AUDIO_CHANNELS
    audio_blocksize: int = 2400
    audio_input_queue_maxsize: int = 64

    # Local voice activity detection for turn-based speech capture.
    speech_enabled: bool = True
    local_vad_threshold: float = 800.0
    local_vad_min_speech_ms: int = 450
    local_vad_silence_ms: int = 1500
    local_vad_max_utterance_ms: int = 10000

    # Demo stability: do not accept spoken interruption while assistant TTS is active.
    allow_voice_interrupt: bool = False
    post_tts_mic_block_ms: int = 600
    timer_alert_mic_block_ms: int = 1800

    # Kokoro local TTS
    kokoro_repo_id: str = os.getenv("COOKING_ASSISTANT_KOKORO_REPO", "hexgrad/Kokoro-82M")
    kokoro_model_dir: str = os.getenv("COOKING_ASSISTANT_KOKORO_DIR", "assets/models/Kokoro-82M")
    kokoro_voice: str = os.getenv("COOKING_ASSISTANT_KOKORO_VOICE", "af_heart")
    kokoro_lang_code: str = os.getenv("COOKING_ASSISTANT_KOKORO_LANG", "a")
    kokoro_speed: float = float(os.getenv("COOKING_ASSISTANT_KOKORO_SPEED", "1.0"))

    auto_connect_backend: bool = True
    image_capture_cooldown_seconds: float = 6.0

    gesture_hand_model_path: str = os.getenv("COOKING_ASSISTANT_HAND_MODEL", "assets/models/hand_landmarker.task")
    gesture_face_model_path: str = os.getenv("COOKING_ASSISTANT_FACE_MODEL", "assets/models/face_detector.tflite")
    gesture_palm_threshold: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_PALM_THRESHOLD", "0.78"))
    gesture_mouth_threshold: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_MOUTH_THRESHOLD", "0.6"))
    gesture_thumbs_up_threshold: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_THUMBS_UP_THRESHOLD", "0.95"))
    gesture_pinky_threshold: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_PINKY_THRESHOLD", "0.78"))
    gesture_fist_threshold: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_FIST_THRESHOLD", "0.82"))
    gesture_option_threshold: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_OPTION_THRESHOLD", "0.80"))
    gesture_mouth_box_expand: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_MOUTH_BOX_EXPAND", "0.02"))
    gesture_mouth_distance_factor: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_MOUTH_DISTANCE_FACTOR", "0.20"))
    gesture_mouth_min_overlap_ratio: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_MOUTH_MIN_OVERLAP_RATIO", "0.03"))
    gesture_pinky_min_extension: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_PINKY_MIN_EXTENSION", "0.05"))
    gesture_pinky_min_separation: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_PINKY_MIN_SEPARATION", "0.025"))
    gesture_fist_max_avg_tip_distance_ratio: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_FIST_MAX_AVG_TIP_DISTANCE_RATIO", "0.58"))
    gesture_fist_max_bbox_height_ratio: float = float(os.getenv("COOKING_ASSISTANT_GESTURE_FIST_MAX_BBOX_HEIGHT_RATIO", "1.50"))
    gesture_debug_logging: bool = os.getenv("COOKING_ASSISTANT_GESTURE_DEBUG_LOGGING", "false").strip().lower() in {"1", "true", "yes", "on"}

    @property
    def context_buffer_maxlen(self) -> int:
        maxlen = int(self.context_buffer_seconds * 1000 / max(self.context_sample_interval_ms, 1))
        return max(4, maxlen)

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def recipe_abspath(self) -> Path:
        return Path(self.recipe_path)
