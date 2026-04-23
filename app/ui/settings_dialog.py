from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QLabel, QPushButton, QVBoxLayout

from app.ui.widgets import CardFrame


class SettingsDialog(QDialog):
    def __init__(self, snapshot: dict, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Kitchen Settings")
        self.resize(420, 340)

        self.speech_checkbox = QCheckBox("Hands-free speech capture")
        self.speech_checkbox.setToolTip("Cover your mouth to toggle speech recognition.")
        self.gesture_checkbox = QCheckBox("Gesture controls")
        self.gesture_checkbox.setToolTip("Enable or disable gesture control from the camera view.")
        self.vision_checkbox = QCheckBox("Visual scene awareness")
        self.vision_checkbox.setToolTip("Enable or disable visual context use.")
        self.tts_checkbox = QCheckBox("Spoken responses")
        self.tts_checkbox.setToolTip("Enable or disable spoken responses.")

        features = snapshot.get("features", {})
        self.speech_checkbox.setChecked(features.get("speech_enabled", True))
        self.gesture_checkbox.setChecked(features.get("gesture_enabled", True))
        self.vision_checkbox.setChecked(features.get("vision_enabled", True))
        self.tts_checkbox.setChecked(features.get("tts_enabled", True))

        layout = QVBoxLayout(self)
        intro = QLabel("Choose how interactive you want the assistant to be while you cook.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        card = CardFrame()
        for checkbox in [self.speech_checkbox, self.gesture_checkbox, self.vision_checkbox, self.tts_checkbox]:
            card.body.addWidget(checkbox)
        layout.addWidget(card)

        close_button = QPushButton("Save and close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
