from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class TimerWidget(QWidget):
    def __init__(self, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self._label = QLabel("No active timers")
        layout = QVBoxLayout(self)
        layout.addWidget(self._label)

    def update_timers(self, timers: list[dict]) -> None:
        if not timers:
            self._label.setText("No active timers")
            return

        lines = [
            f"{timer['label']}: {timer['remaining_seconds']}s"
            for timer in timers
        ]
        self._label.setText("\n".join(lines))