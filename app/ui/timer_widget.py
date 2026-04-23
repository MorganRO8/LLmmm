from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class TimerWidget(QWidget):
    def __init__(self, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self.hero_label = QLabel("")
        self.hero_label.setObjectName("TimerHeroLabel")
        self.hero_label.setAlignment(Qt.AlignCenter)
        self.hero_label.hide()

        self.time_label = QLabel("-:-:-")
        self.time_label.setObjectName("TimerHeroTime")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setProperty("active", False)

        self._layout.addWidget(self.hero_label)
        self._layout.addWidget(self.time_label)

        self._last_active_label = ""
        self._last_active_time = "-:-:-"
        self._idle_reset_deadline = 0.0

    def _format_time(self, total_seconds: int) -> str:
        total_seconds = max(0, int(total_seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    def _set_active_state(self, active: bool) -> None:
        self.time_label.setProperty("active", active)
        self.time_label.style().unpolish(self.time_label)
        self.time_label.style().polish(self.time_label)
        self.time_label.update()

    def _show_idle(self) -> None:
        self.hero_label.hide()
        self.hero_label.setText("")
        self.time_label.setText("-:-:-")
        self._set_active_state(False)

    def update_timers(self, timers: list[dict]) -> None:
        now = time.monotonic()
        if not timers:
            if self._idle_reset_deadline and now < self._idle_reset_deadline:
                self.hero_label.setText(self._last_active_label or "Done")
                self.hero_label.show()
                self.time_label.setText("0:00")
                self._set_active_state(True)
                return
            self._show_idle()
            return

        sorted_timers = sorted(timers, key=lambda timer: int(timer.get("remaining_seconds", 0)))
        primary = sorted_timers[0]
        remaining = int(primary.get("remaining_seconds", 0))
        self._last_active_label = primary.get("label", "Timer")
        self._last_active_time = self._format_time(remaining)
        self._idle_reset_deadline = now + 2.5

        self.hero_label.setText(self._last_active_label)
        self.hero_label.show()
        self.time_label.setText(self._last_active_time)
        self._set_active_state(True)
