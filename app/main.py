from __future__ import annotations

import asyncio

import numpy as np

from app.config import AppConfig
from app.core.bus import EventBus
from app.core.recipe import RecipeLoader
from app.core.state import AppStateStore
from app.logging_setup import configure_logging, get_logger
from app.realtime.tools import ToolExecutor
from app.timers.manager import TimerManager
from app.vision.frame_buffer import FrameBuffer


class DummyCameraService:
    def __init__(self, frame_buffer: FrameBuffer) -> None:
        self._frame_buffer = frame_buffer

    def get_context_frames(self, count: int = 3, max_age_seconds: float = 12.0):
        return self._frame_buffer.get_recent(count=count, max_age_seconds=max_age_seconds)


async def async_main() -> None:
    config = AppConfig()
    configure_logging(config.log_level)
    logger = get_logger(__name__)

    recipe = RecipeLoader.load(config.recipe_path)
    state_store = AppStateStore(recipe)
    event_bus = EventBus()
    timer_manager = TimerManager(state_store.state, event_bus)
    frame_buffer = FrameBuffer(maxlen=12)
    tool_executor = ToolExecutor(state_store, timer_manager, DummyCameraService(frame_buffer))

    await timer_manager.start()

    logger.info("Loaded recipe: %s", recipe.name)
    logger.info("Current step: %s", state_store.get_current_step().instruction)

    for idx in range(3):
        frame = np.zeros((config.context_frame_height, config.context_frame_width, 3), dtype=np.uint8)
        frame[:, :, idx % 3] = 150
        frame_buffer.add_frame(frame=frame, motion_score=float(idx) / 10.0)

    timer_result = await tool_executor.execute(
        "start_timer",
        {"label": "demo stir timer", "seconds": 3, "warn_at_seconds": 1},
    )
    state_result = await tool_executor.execute("get_recipe_state")
    frame_result = await tool_executor.execute(
        "capture_context_frames",
        {"reason": "smoke_test", "count": 2, "max_age_seconds": 30.0},
    )

    logger.info("Timer result: %s", timer_result)
    logger.info("State result current step: %s", state_result["result"]["current_step"]["title"])
    logger.info("Captured %s buffered frames", len(frame_result["result"]["frames"]))

    await asyncio.sleep(3.5)
    await timer_manager.stop()
    logger.info("Foundation smoke test completed.")


if __name__ == "__main__":
    asyncio.run(async_main())
