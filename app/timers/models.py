from __future__ import annotations

from pydantic import BaseModel


class TimerCreateRequest(BaseModel):
    label: str
    seconds: int
    warn_at_seconds: int = 10
    notify_at_seconds: int = 0
