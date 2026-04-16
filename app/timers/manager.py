from __future__ import annotations

import asyncio
import time
import uuid

from app.core.bus import EventBus
from app.core.events import TimerExpiredEvent, TimerNotifyEvent, TimerWarningEvent
from app.core.models import ActiveTimer, AppState


class TimerManager:
    def __init__(self, state: AppState, event_bus: EventBus) -> None:
        self._state = state
        self._event_bus = event_bus
        self._tick_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._tick_task and not self._tick_task.done():
            return
        self._stop_event.clear()
        self._tick_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._tick_task:
            await self._tick_task

    async def create_timer(self, label: str, seconds: int, warn_at_seconds: int = 10, notify_at_seconds: int = 0) -> ActiveTimer:
        now = time.time()
        warn_at_seconds = max(0, min(warn_at_seconds, seconds))
        notify_at_seconds = max(0, min(notify_at_seconds, seconds))
        timer = ActiveTimer(
            timer_id=uuid.uuid4().hex,
            label=label,
            total_seconds=seconds,
            remaining_seconds=seconds,
            warn_at_seconds=warn_at_seconds,
            notify_at_seconds=notify_at_seconds,
            started_at=now,
            ends_at=now + seconds,
        )
        self._state.timers.append(timer)
        return timer

    async def cancel_timer(self, timer_id: str) -> bool:
        for idx, timer in enumerate(self._state.timers):
            if timer.timer_id == timer_id:
                del self._state.timers[idx]
                return True
        return False

    async def list_timers(self) -> list[ActiveTimer]:
        return list(self._state.timers)

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            expired_ids: list[str] = []
            for timer in self._state.timers:
                if not timer.is_running:
                    continue
                remaining = max(0, int(round(timer.ends_at - now)))
                timer.remaining_seconds = remaining

                if not timer.warning_emitted and remaining <= timer.warn_at_seconds and remaining > 0:
                    timer.warning_emitted = True
                    await self._event_bus.publish(
                        TimerWarningEvent(
                            timestamp=now,
                            timer_id=timer.timer_id,
                            label=timer.label,
                            remaining_seconds=remaining,
                        )
                    )

                if not timer.notify_emitted and remaining <= timer.notify_at_seconds:
                    timer.notify_emitted = True
                    await self._event_bus.publish(
                        TimerNotifyEvent(
                            timestamp=now,
                            timer_id=timer.timer_id,
                            label=timer.label,
                            remaining_seconds=remaining,
                        )
                    )

                if remaining <= 0:
                    expired_ids.append(timer.timer_id)
                    await self._event_bus.publish(
                        TimerExpiredEvent(
                            timestamp=now,
                            timer_id=timer.timer_id,
                            label=timer.label,
                        )
                    )

            if expired_ids:
                self._state.timers[:] = [t for t in self._state.timers if t.timer_id not in expired_ids]

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass
