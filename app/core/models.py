from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecipeStep(BaseModel):
    step_id: str
    title: str
    instruction: str
    tips: list[str] = Field(default_factory=list)
    visual_check_suggested: bool = False
    default_timer_seconds: int | None = None


class RecipeBranch(BaseModel):
    branch_id: str
    label: str
    step_ids: list[str]


class Recipe(BaseModel):
    recipe_id: str
    name: str
    description: str
    steps: list[RecipeStep]
    branches: list[RecipeBranch] = Field(default_factory=list)


class ActiveTimer(BaseModel):
    timer_id: str
    label: str
    total_seconds: int
    remaining_seconds: int
    warn_at_seconds: int = 10
    notify_at_seconds: int = 0
    is_running: bool = True
    started_at: float
    ends_at: float
    warning_emitted: bool = False
    notify_emitted: bool = False


class FeatureFlags(BaseModel):
    speech_enabled: bool = True
    gesture_enabled: bool = True
    vision_enabled: bool = True
    tts_enabled: bool = True


class RealtimeStatus(BaseModel):
    connected: bool = False
    session_id: str | None = None
    model_name: str | None = None
    assistant_speaking: bool = False
    user_speaking: bool = False


class RecipeRuntimeState(BaseModel):
    recipe_id: str
    current_step_index: int = 0
    selected_branch_id: str | None = None
    completed_step_ids: list[str] = Field(default_factory=list)


class TranscriptEntry(BaseModel):
    speaker: str
    text: str
    timestamp: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppState(BaseModel):
    recipe: Recipe
    recipe_state: RecipeRuntimeState
    timers: list[ActiveTimer] = Field(default_factory=list)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    realtime: RealtimeStatus = Field(default_factory=RealtimeStatus)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    last_image_capture_ts: float | None = None
