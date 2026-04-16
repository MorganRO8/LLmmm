from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QFormLayout, QVBoxLayout, QPushButton


class SettingsDialog(QDialog):
    def __init__(self, snapshot: dict, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.speech_checkbox = QCheckBox("Speech enabled")
        self.gesture_checkbox = QCheckBox("Gesture enabled")
        self.vision_checkbox = QCheckBox("Vision enabled")
        self.tts_checkbox = QCheckBox("TTS enabled")

        features = snapshot.get("features", {})
        self.speech_checkbox.setChecked(features.get("speech_enabled", True))
        self.gesture_checkbox.setChecked(features.get("gesture_enabled", True))
        self.vision_checkbox.setChecked(features.get("vision_enabled", True))
        self.tts_checkbox.setChecked(features.get("tts_enabled", True))

        form = QFormLayout()
        form.addRow(self.speech_checkbox)
        form.addRow(self.gesture_checkbox)
        form.addRow(self.vision_checkbox)
        form.addRow(self.tts_checkbox)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(close_button)
