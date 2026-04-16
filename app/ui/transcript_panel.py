from __future__ import annotations

from PySide6.QtWidgets import QTextEdit


class TranscriptPanel(QTextEdit):
    def __init__(self, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.setReadOnly(True)

    def append_line(self, speaker: str, text: str) -> None:
        self.append(f"<b>{speaker}:</b> {text}")
