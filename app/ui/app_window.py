from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.controller import DesktopController
from app.ui.camera_panel import CameraPanel
from app.ui.settings_dialog import SettingsDialog
from app.ui.timer_widget import TimerWidget
from app.ui.transcript_panel import TranscriptPanel


class AppWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("Cooking Assistant Demo")
        self.resize(1200, 760)

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=2)

        self.camera_panel = CameraPanel()
        self.status_label = QLabel("Initializing...")
        self.step_label = QLabel()
        self.timer_widget = TimerWidget()
        left.addWidget(self.camera_panel)
        left.addWidget(self.status_label)
        left.addWidget(self.step_label)
        left.addWidget(self.timer_widget)

        self.transcript_panel = TranscriptPanel()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type a message to the assistant...")
        send_button = QPushButton("Send")
        send_button.clicked.connect(self._send_text)
        next_button = QPushButton("Next Step")
        next_button.clicked.connect(self.controller.advance_step)
        repeat_button = QPushButton("Repeat")
        repeat_button.clicked.connect(self.controller.repeat_step)
        look_button = QPushButton("Look Now")
        look_button.clicked.connect(self.controller.look_now)
        interrupt_button = QPushButton("Interrupt")
        interrupt_button.clicked.connect(lambda: self.controller.interrupt("ui"))
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._open_settings)

        right.addWidget(self.transcript_panel)
        right.addWidget(self.input_line)
        controls = QHBoxLayout()
        for button in [send_button, next_button, repeat_button, look_button, interrupt_button, settings_button]:
            controls.addWidget(button)
        right.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

        self.controller.start()
        self._refresh()

    def closeEvent(self, event):  # noqa: ANN001, N802
        self.controller.stop()
        super().closeEvent(event)

    def _send_text(self) -> None:
        text = self.input_line.text().strip()
        if not text:
            return
        self.controller.send_text_message(text)
        self.input_line.clear()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.controller.get_state_snapshot(), self)
        if dialog.exec():
            self.controller.set_feature_flag("speech_enabled", dialog.speech_checkbox.isChecked())
            self.controller.set_feature_flag("gesture_enabled", dialog.gesture_checkbox.isChecked())
            self.controller.set_feature_flag("vision_enabled", dialog.vision_checkbox.isChecked())
            self.controller.set_feature_flag("tts_enabled", dialog.tts_checkbox.isChecked())

    def _refresh(self) -> None:
        frame = self.controller.get_latest_preview_rgb()
        self.camera_panel.update_frame(frame)

        snapshot = self.controller.get_state_snapshot()
        current_step = snapshot.get("current_step", {})

        title = current_step.get("title", "")
        instruction = current_step.get("instruction", "")
        self.step_label.setText(f"Current step: {title}\n{instruction}")

        self.timer_widget.update_timers(snapshot.get("timers", []))

        status = snapshot.get("realtime", {})
        self.status_label.setText(
            f"Backend ready: {status.get('connected', False)} | "
            f"Assistant speaking: {status.get('assistant_speaking', False)} | "
            f"User speaking: {status.get('user_speaking', False)}"
        )

        for item in self.controller.drain_ui_events():
            etype = item.get("type")
            if etype == "assistant_text":
                self.transcript_panel.append_line("assistant", item.get("text", ""))
            elif etype == "user_text":
                self.transcript_panel.append_line("user", item.get("text", ""))
            elif etype == "error":
                self.transcript_panel.append_line("system", f"Error: {item.get('message', '')}")
            elif etype == "gesture":
                self.transcript_panel.append_line("gesture", f"{item.get('name')} ({item.get('held_for_ms')} ms)")
            elif etype == "interrupt":
                self.transcript_panel.append_line("system", f"Interrupted via {item.get('source')}")
            elif etype == "tool":
                self.transcript_panel.append_line("tool", f"{item.get('tool_name')} called")
            elif etype == "status" and item.get("message"):
                self.transcript_panel.append_line("system", item["message"])
