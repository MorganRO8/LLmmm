from __future__ import annotations

import threading
import time
from typing import Any

from app.core.models import AppState, Recipe, RecipeRuntimeState, TranscriptEntry


class AppStateStore:
    def __init__(self, recipe: Recipe) -> None:
        self._lock = threading.RLock()
        self._state = AppState(
            recipe=recipe,
            recipe_state=RecipeRuntimeState(recipe_id=recipe.recipe_id),
        )

    @property
    def state(self) -> AppState:
        return self._state

    def get_current_step(self):
        with self._lock:
            return self._state.recipe.steps[self._state.recipe_state.current_step_index]

    def get_next_step(self):
        with self._lock:
            next_index = self._state.recipe_state.current_step_index + 1
            if next_index >= len(self._state.recipe.steps):
                return None
            return self._state.recipe.steps[next_index]

    def advance_step(self) -> dict[str, Any]:
        with self._lock:
            current = self.get_current_step()
            if current.step_id not in self._state.recipe_state.completed_step_ids:
                self._state.recipe_state.completed_step_ids.append(current.step_id)

            if self._state.recipe_state.current_step_index < len(self._state.recipe.steps) - 1:
                self._state.recipe_state.current_step_index += 1
                advanced = True
            else:
                advanced = False

            return {
                "advanced": advanced,
                "current_step": self.get_current_step().model_dump(),
                "next_step": self.get_next_step().model_dump() if self.get_next_step() else None,
                "completed_step_ids": list(self._state.recipe_state.completed_step_ids),
            }

    def previous_step(self) -> dict[str, Any]:
        with self._lock:
            moved_back = False
            if self._state.recipe_state.current_step_index > 0:
                self._state.recipe_state.current_step_index -= 1
                moved_back = True
            return {
                "moved_back": moved_back,
                "current_step": self.get_current_step().model_dump(),
                "next_step": self.get_next_step().model_dump() if self.get_next_step() else None,
                "completed_step_ids": list(self._state.recipe_state.completed_step_ids),
            }

    def repeat_step(self) -> dict[str, Any]:
        with self._lock:
            return {"current_step": self.get_current_step().model_dump()}

    def add_transcript(self, speaker: str, text: str, **metadata: Any) -> TranscriptEntry:
        with self._lock:
            entry = TranscriptEntry(
                speaker=speaker,
                text=text,
                timestamp=time.time(),
                metadata=metadata,
            )
            self._state.transcript.append(entry)
            return entry

    def update_realtime_status(self, *, connected: bool | None = None, session_id: str | None = None, model_name: str | None = None, assistant_speaking: bool | None = None, user_speaking: bool | None = None) -> None:
        with self._lock:
            if connected is not None:
                self._state.realtime.connected = connected
            if session_id is not None:
                self._state.realtime.session_id = session_id
            if model_name is not None:
                self._state.realtime.model_name = model_name
            if assistant_speaking is not None:
                self._state.realtime.assistant_speaking = assistant_speaking
            if user_speaking is not None:
                self._state.realtime.user_speaking = user_speaking

    def set_last_image_capture_ts(self, timestamp: float | None) -> None:
        with self._lock:
            self._state.last_image_capture_ts = timestamp

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            current_step = self.get_current_step()
            next_step = self.get_next_step()
            return {
                "recipe": self._state.recipe.model_dump(),
                "recipe_state": self._state.recipe_state.model_dump(),
                "current_step": current_step.model_dump(),
                "next_step": next_step.model_dump() if next_step else None,
                "timers": [timer.model_dump() for timer in self._state.timers],
                "features": self._state.features.model_dump(),
                "realtime": self._state.realtime.model_dump(),
                "transcript": [entry.model_dump() for entry in self._state.transcript],
                "last_image_capture_ts": self._state.last_image_capture_ts,
            }
