from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class CardFrame(QFrame):
    def __init__(self, title: str | None = None, accent: bool = False, parent=None, compact: bool = False):  # noqa: ANN001
        super().__init__(parent)
        self.setObjectName("AccentCard" if accent else "Card")
        self._layout = QVBoxLayout(self)
        margin = 12 if compact else 18
        spacing = 8 if compact else 12
        self._layout.setContentsMargins(margin, margin, margin, margin)
        self._layout.setSpacing(spacing)
        if title:
            header = QLabel(title)
            header.setStyleSheet(f"font-size: {12 if compact else 14}pt; font-weight: 700;")
            self._layout.addWidget(header)

    @property
    def body(self) -> QVBoxLayout:
        return self._layout


class StatusPill(QFrame):
    def __init__(self, title: str, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.setObjectName("StatusPill")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("PillTitle")
        self.value_label = QLabel("—")
        self.value_label.setObjectName("PillValue")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str, *, active: bool = False, warning: bool = False) -> None:
        self.value_label.setText(value)
        self.setProperty("active", active)
        self.setProperty("warning", warning)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class StatusBadge(QFrame):
    def __init__(self, text: str, parent=None, compact: bool = False):  # noqa: ANN001
        super().__init__(parent)
        self.setObjectName("StatusBadge")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10 if compact else 12, 8 if compact else 10, 10 if compact else 12, 8 if compact else 10)
        layout.setSpacing(6 if compact else 8)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if compact:
            label.setStyleSheet("font-size: 9pt; font-weight: 600;")
        layout.addWidget(label)


class RecipeOptionCard(QFrame):
    clicked = Signal(str)

    def __init__(self, recipe_path: str, title: str, description: str, meta: str, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.recipe_path = recipe_path
        self.setObjectName("RecipeOption")
        self.setProperty("selected", False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14pt; font-weight: 700;")
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setObjectName("subtle")
        meta_label = QLabel(meta)
        meta_label.setStyleSheet("font-size: 9pt; color: #946f53; font-weight: 600;")

        select_button = QPushButton("Choose recipe")
        select_button.setProperty("secondary", "true")
        select_button.clicked.connect(lambda: self.clicked.emit(self.recipe_path))

        layout.addWidget(title_label)
        layout.addWidget(desc_label)
        layout.addWidget(meta_label)
        layout.addStretch(1)
        layout.addWidget(select_button, alignment=Qt.AlignRight)

    def mousePressEvent(self, event):  # noqa: ANN001, N802
        self.clicked.emit(self.recipe_path)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
