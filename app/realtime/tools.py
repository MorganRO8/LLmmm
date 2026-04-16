from __future__ import annotations

import base64
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.config import AppConfig
from app.core.state import AppStateStore
from app.timers.manager import TimerManager
from app.vision.camera_service import CameraService


class StartTimerArgs(BaseModel):
    label: str
    seconds: int = Field(gt=0)
    warn_at_seconds: int = Field(default=10, ge=0)
    notify_at_seconds: int = Field(default=0, ge=0)


class CaptureContextFramesArgs(BaseModel):
    reason: str
    count: int = Field(default=3, ge=1, le=4)
    max_age_seconds: float = Field(default=12.0, gt=0.0, le=30.0)


class SetFeatureFlagArgs(BaseModel):
    feature_name: str
    enabled: bool


class ToolExecutor:
    def __init__(self, state_store: AppStateStore, timer_manager: TimerManager, camera_service: CameraService, config: AppConfig | None = None) -> None:
        self._state_store = state_store
        self._timer_manager = timer_manager
        self._camera_service = camera_service
        self._config = config or AppConfig()

    async def execute(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        try:
            if tool_name == "get_recipe_state":
                return {"ok": True, "result": self._state_store.snapshot()}

            if tool_name == "advance_step":
                return {"ok": True, "result": self._state_store.advance_step()}

            if tool_name == "repeat_step":
                return {"ok": True, "result": self._state_store.repeat_step()}

            if tool_name == "start_timer":
                payload = StartTimerArgs.model_validate(arguments)
                timer = await self._timer_manager.create_timer(
                    label=payload.label,
                    seconds=payload.seconds,
                    warn_at_seconds=payload.warn_at_seconds,
                    notify_at_seconds=payload.notify_at_seconds,
                )
                return {"ok": True, "result": timer.model_dump()}

            if tool_name == "list_timers":
                timers = await self._timer_manager.list_timers()
                return {"ok": True, "result": [timer.model_dump() for timer in timers]}

            if tool_name == "cancel_timer":
                timer_id = str(arguments.get("timer_id", ""))
                if not timer_id:
                    return {"ok": False, "error": "missing_timer_id"}
                cancelled = await self._timer_manager.cancel_timer(timer_id)
                return {"ok": True, "result": {"cancelled": cancelled, "timer_id": timer_id}}

            if tool_name == "capture_context_frames":
                payload = CaptureContextFramesArgs.model_validate(arguments)
                frames = self._camera_service.get_context_frames(
                    count=payload.count,
                    max_age_seconds=payload.max_age_seconds,
                )
                self._state_store.set_last_image_capture_ts(frames[-1].timestamp if frames else None)
                return {
                    "ok": True,
                    "result": {
                        "reason": payload.reason,
                        "frames": [
                            {
                                "frame_id": frame.frame_id,
                                "timestamp": frame.timestamp,
                                "width": frame.width,
                                "height": frame.height,
                                "jpeg_bytes_base64": base64.b64encode(frame.jpeg_bytes).decode("ascii"),
                                "motion_score": frame.motion_score,
                            }
                            for frame in frames
                        ],
                    },
                }

            if tool_name == "set_feature_flag":
                payload = SetFeatureFlagArgs.model_validate(arguments)
                features = self._state_store.state.features
                if not hasattr(features, payload.feature_name):
                    return {"ok": False, "error": "unknown_feature", "feature_name": payload.feature_name}
                setattr(features, payload.feature_name, payload.enabled)
                return {"ok": True, "result": features.model_dump()}

            return {"ok": False, "error": "unknown_tool", "tool_name": tool_name}
        except ValidationError as exc:
            return {"ok": False, "error": "validation_error", "details": exc.errors()}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_recipe_state",
        "description": "Return the current recipe step, next step, timers, and enabled features.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "advance_step",
        "description": "Advance the recipe to the next step when the user completes the current step.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "repeat_step",
        "description": "Return the current recipe step without changing state.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "start_timer",
        "description": "Start a local timer with a label and duration in seconds. notify_at_seconds controls when a local alert sound should play; use 0 for the end of the timer.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "seconds": {"type": "integer", "minimum": 1},
                "warn_at_seconds": {"type": "integer", "minimum": 0},
                "notify_at_seconds": {"type": "integer", "minimum": 0},
            },
            "required": ["label", "seconds"],
        },
    },
    {
        "type": "function",
        "name": "list_timers",
        "description": "List currently active local timers.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "cancel_timer",
        "description": "Cancel a timer by its timer_id.",
        "parameters": {
            "type": "object",
            "properties": {"timer_id": {"type": "string"}},
            "required": ["timer_id"],
        },
    },
    {
        "type": "function",
        "name": "capture_context_frames",
        "description": "Retrieve recent buffered low-resolution frames when visual context would help answer the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 4},
                "max_age_seconds": {"type": "number", "minimum": 0.1, "maximum": 30.0},
            },
            "required": ["reason"],
        },
    },
    {
        "type": "function",
        "name": "set_feature_flag",
        "description": "Enable or disable speech, gesture, vision, or tts features.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature_name": {"type": "string"},
                "enabled": {"type": "boolean"},
            },
            "required": ["feature_name", "enabled"],
        },
    },
]
