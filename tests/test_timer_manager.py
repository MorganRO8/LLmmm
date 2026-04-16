from __future__ import annotations

import asyncio

import pytest

from app.core.bus import EventBus
from app.core.events import TimerExpiredEvent, TimerNotifyEvent, TimerWarningEvent
from app.core.models import AppState, Recipe, RecipeRuntimeState
from app.timers.manager import TimerManager


@pytest.mark.asyncio
async def test_timer_expires_and_emits_events() -> None:
    recipe = Recipe(recipe_id="r", name="n", description="d", steps=[])
    state = AppState(recipe=recipe, recipe_state=RecipeRuntimeState(recipe_id="r"))
    bus = EventBus()
    manager = TimerManager(state, bus)

    seen: list[str] = []

    async def on_warning(event: TimerWarningEvent) -> None:
        seen.append(f"warn:{event.remaining_seconds}")

    async def on_notify(event: TimerNotifyEvent) -> None:
        seen.append(f"notify:{event.remaining_seconds}")

    async def on_expired(event: TimerExpiredEvent) -> None:
        seen.append(f"expired:{event.label}")

    bus.subscribe(TimerWarningEvent, on_warning)
    bus.subscribe(TimerNotifyEvent, on_notify)
    bus.subscribe(TimerExpiredEvent, on_expired)

    await manager.start()
    timer = await manager.create_timer(label="x", seconds=1, warn_at_seconds=1, notify_at_seconds=0)
    assert timer.remaining_seconds == 1
    await asyncio.sleep(1.4)
    timers = await manager.list_timers()
    await manager.stop()

    assert timers == []
    assert any(item.startswith("notify:") for item in seen)
    assert "expired:x" in seen
