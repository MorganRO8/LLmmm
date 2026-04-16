from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    timestamp: float


class GestureRecognizedEvent(BaseEvent):
    gesture_name: str
    confidence: float
    held_for_ms: int


class InterruptRequestedEvent(BaseEvent):
    source: Literal["speech", "gesture", "ui"]


class TimerWarningEvent(BaseEvent):
    timer_id: str
    label: str
    remaining_seconds: int


class TimerNotifyEvent(BaseEvent):
    timer_id: str
    label: str
    remaining_seconds: int


class TimerExpiredEvent(BaseEvent):
    timer_id: str
    label: str


class ToolCallRequestedEvent(BaseEvent):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultReadyEvent(BaseEvent):
    tool_name: str
    result: dict[str, Any]


class TranscriptAppendedEvent(BaseEvent):
    speaker: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RealtimeStatusChangedEvent(BaseEvent):
    connected: bool
    session_id: str | None = None
    model_name: str | None = None


class AssistantSpeakingEvent(BaseEvent):
    speaking: bool


class UserSpeakingEvent(BaseEvent):
    speaking: bool
