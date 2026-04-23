from __future__ import annotations

import re
from html import escape

from PySide6.QtWidgets import QSizePolicy, QTextEdit

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


class TranscriptPanel(QTextEdit):
    def __init__(self, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Your conversation will appear here.")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    def append_message(self, speaker: str, text: str) -> None:
        safe_speaker = escape(speaker.title())
        safe_text = self._format_message_html(text)
        bubble = (
            "<div style='margin:8px 0;padding:12px 14px;border-radius:14px;"
            f"background:{'#fff0db' if speaker == 'user' else '#fffaf5'};"
            f"border:1px solid {'#efcc9f' if speaker == 'user' else '#ead9c8'};'>"
            f"<strong>{safe_speaker}:</strong> {safe_text}"
            "</div>"
        )
        self.append(bubble)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        
    def _format_message_html(self, text: str) -> str:
        text = re.sub(
            r"(?m)(^|\n)\*\*(Assistant|User)\*\*(?!:)\s*",
            r"\1**\2:** ",
            text,
        )
        text = re.sub(
            r"(?m)(^|\n)(Assistant|User)(?!:)\s*",
            r"\1**\2:** ",
            text,
        )
        escaped = escape(text).replace("\n", "<br>")
        return _BOLD_RE.sub(r"<strong>\1</strong>", escaped)