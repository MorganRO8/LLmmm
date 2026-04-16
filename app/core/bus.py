from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

from app.core.events import BaseEvent

EventT = TypeVar("EventT", bound=BaseEvent)
Subscriber = Callable[[EventT], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[type[BaseEvent], list[Subscriber[BaseEvent]]] = defaultdict(list)

    def subscribe(self, event_type: type[EventT], handler: Subscriber[EventT]) -> None:
        self._subscribers[event_type].append(cast(Subscriber[BaseEvent], handler))

    async def publish(self, event: BaseEvent) -> None:
        handlers = list(self._subscribers.get(type(event), []))
        if not handlers:
            return
        await asyncio.gather(*(handler(event) for handler in handlers))
