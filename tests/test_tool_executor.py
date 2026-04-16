from __future__ import annotations

import asyncio
import numpy as np
import pytest

from app.config import AppConfig
from app.core.bus import EventBus
from app.core.recipe import RecipeLoader
from app.core.state import AppStateStore
from app.realtime.tools import ToolExecutor
from app.timers.manager import TimerManager


class DummyCameraService:
    def __init__(self):
        from app.vision.frame_buffer import FrameBuffer
        self._buffer = FrameBuffer(maxlen=4)
        self._buffer.add_frame(np.zeros((64, 64, 3), dtype=np.uint8), motion_score=0.0)

    def get_context_frames(self, count: int = 3, max_age_seconds: float = 12.0):
        return self._buffer.get_recent(count=count, max_age_seconds=max_age_seconds)


@pytest.mark.asyncio
async def test_tool_executor_start_timer() -> None:
    recipe = RecipeLoader.load("recipes/scrambled_eggs.json")
    state_store = AppStateStore(recipe)
    timer_manager = TimerManager(state_store.state, EventBus())
    await timer_manager.start()
    tool_executor = ToolExecutor(state_store, timer_manager, DummyCameraService(), AppConfig())

    result = await tool_executor.execute(
        "start_timer",
        {"label": "whisk", "seconds": 2, "warn_at_seconds": 1, "notify_at_seconds": 0},
    )
    await timer_manager.stop()

    assert result["ok"] is True
    assert result["result"]["label"] == "whisk"
    assert result["result"]["notify_at_seconds"] == 0
