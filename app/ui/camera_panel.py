from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraPanel(QWidget):
    def __init__(self, parent=None):  # noqa: ANN001
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 420)
        self.image_label.setStyleSheet(
            "background:#2d241d;color:#f8ead9;border-radius:18px;border:1px solid #5b4b3e;"
        )
        self.image_label.setText("Camera warming up…")

        layout.addWidget(self.image_label)

    def update_frame(self, rgb_frame) -> None:  # noqa: ANN001
        if rgb_frame is None:
            return
        height, width, channels = rgb_frame.shape
        image = QImage(rgb_frame.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
