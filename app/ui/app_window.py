from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.controller import DesktopController
from app.ui.camera_panel import CameraPanel
from app.ui.settings_dialog import SettingsDialog
from app.ui.startup_window import StartupWindow
from app.ui.timer_widget import TimerWidget
from app.ui.transcript_panel import TranscriptPanel
from app.ui.widgets import CardFrame, StatusBadge, StatusPill


class AppWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self._returning_to_menu = False
        self._startup_window: StartupWindow | None = None
        self.setWindowTitle("LLmmm")
        self.resize(1360, 900)
        self.setMinimumSize(1180, 760)

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        header = QHBoxLayout()
        header.setSpacing(12)
        root.addLayout(header)

        title_block = QVBoxLayout()
        self.recipe_name_label = QLabel(self.controller.recipe.name)
        self.recipe_name_label.setObjectName("headline")
        self.recipe_desc_label = QLabel(self.controller.recipe.description)
        self.recipe_desc_label.setObjectName("subtle")
        self.recipe_desc_label.setWordWrap(True)
        title_block.addWidget(self.recipe_name_label)
        title_block.addWidget(self.recipe_desc_label)
        header.addLayout(title_block, stretch=1)

        self.backend_pill = StatusPill("Backend")
        self.assistant_pill = StatusPill("Assistant")
        self.user_pill = StatusPill("Speech")
        for pill in [self.backend_pill, self.assistant_pill, self.user_pill]:
            header.addWidget(pill)

        workspace = QSplitter()
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(8)
        root.addWidget(workspace, stretch=1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        camera_card = CardFrame("Kitchen view")
        camera_card.setMinimumHeight(455)
        self.camera_panel = CameraPanel()
        camera_card.body.addWidget(self.camera_panel)
        left_layout.addWidget(camera_card, stretch=6)

        self.step_card = CardFrame("Current step", accent=True, compact=True)
        self.step_title = QLabel()
        self.step_title.setStyleSheet("font-size: 15pt; font-weight: 700;")
        self.step_instruction = QTextEdit()
        self.step_instruction.setReadOnly(True)
        self.step_instruction.setObjectName("StepInstruction")
        self.step_instruction.setLineWrapMode(QTextEdit.WidgetWidth)
        self.step_instruction.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.step_instruction.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.step_instruction.setMinimumHeight(74)
        self.step_instruction.setMaximumHeight(92)
        self.step_meta = QLabel()
        self.step_meta.setObjectName("subtle")
        self.step_meta.setWordWrap(True)
        self.step_card.body.addWidget(self.step_title)
        self.step_card.body.addWidget(self.step_instruction)
        self.step_card.body.addWidget(self.step_meta)
        self.step_card.setMaximumHeight(180)
        left_layout.addWidget(self.step_card, stretch=0)

        right_split = QSplitter()
        right_split.setOrientation(Qt.Vertical)
        right_split.setChildrenCollapsible(False)
        right_split.setHandleWidth(8)

        transcript_wrap = QWidget()
        transcript_layout = QVBoxLayout(transcript_wrap)
        transcript_layout.setContentsMargins(0, 0, 0, 0)
        transcript_layout.setSpacing(0)

        transcript_card = CardFrame()
        transcript_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.transcript_panel = TranscriptPanel()
        transcript_card.body.addWidget(self.transcript_panel, stretch=1)
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Ask for help, request a visual check, or type a note…")
        self.input_line.returnPressed.connect(self._send_text)
        transcript_card.body.addWidget(self.input_line)
        transcript_layout.addWidget(transcript_card)
        right_split.addWidget(transcript_wrap)

        lower_right = QWidget()
        lower_right_layout = QVBoxLayout(lower_right)
        lower_right_layout.setContentsMargins(0, 0, 0, 0)
        lower_right_layout.setSpacing(10)

        controls_card = CardFrame(compact=True)
        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls_card.body.addLayout(controls)
        for button in [
            self._make_button("Main menu", self._return_to_menu, secondary=True),
            self._make_button("Send", self._send_text),
            self._make_button("Next step", self.controller.advance_step),
            self._make_button("Repeat", self.controller.repeat_step, secondary=True),
            self._make_button("Look now", self.controller.look_now, secondary=True),
            self._make_button("Interrupt", lambda: self.controller.interrupt("ui"), secondary=True),
            self._make_button("Settings", self._open_settings, secondary=True),
        ]:
            button.setMinimumHeight(34)
            controls.addWidget(button)
        controls.addStretch(1)
        lower_right_layout.addWidget(controls_card)

        self.gesture_card = CardFrame(compact=True)
        gesture_layout = QGridLayout()
        gesture_layout.setContentsMargins(0, 0, 0, 0)
        gesture_layout.setHorizontalSpacing(8)
        gesture_layout.setVerticalSpacing(6)
        gesture_items = [
            "✋ Stop - Palm at camera",
            "🤫 Mute - Hand over mouth",
            "👍 Next - Thumbs up",
            "⇤ Back - Pinky up",
            "✊ Repeat - Fist up",
            "Pick - 1, 2, or 3 finger(s)",
        ]
        for index, text in enumerate(gesture_items):
            badge = StatusBadge(text, compact=True)
            badge.setMaximumHeight(40)
            gesture_layout.addWidget(badge, index // 3, index % 3)
        self.gesture_card.body.addLayout(gesture_layout)
        lower_right_layout.addWidget(self.gesture_card)

        self.timer_card = CardFrame(compact=True)
        self.timer_widget = TimerWidget()
        self.timer_card.body.addWidget(self.timer_widget)
        self.timer_card.setMaximumHeight(86)
        lower_right_layout.addWidget(self.timer_card)
        lower_right.setMaximumHeight(250)
        right_split.addWidget(lower_right)

        workspace.addWidget(left_panel)
        workspace.addWidget(right_split)
        workspace.setStretchFactor(0, 3)
        workspace.setStretchFactor(1, 2)
        workspace.setSizes([1000, 500])
        right_split.setSizes([860, 210])

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

        self.controller.start()
        self._refresh()
        self.showMaximized()

    def _make_button(self, label: str, callback, secondary: bool = False) -> QPushButton:  # noqa: ANN001
        button = QPushButton(label)
        if secondary:
            button.setProperty("secondary", "true")
        button.clicked.connect(callback)
        return button

    def closeEvent(self, event):  # noqa: ANN001, N802
        if self._timer.isActive():
            self._timer.stop()
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

    def _return_to_menu(self) -> None:
        if self._returning_to_menu:
            return
        self._returning_to_menu = True
        self._startup_window = StartupWindow(AppConfig())
        self._startup_window.show()
        self.close()

    def _refresh(self) -> None:
        frame = self.controller.get_latest_preview_rgb()
        self.camera_panel.update_frame(frame)

        snapshot = self.controller.get_state_snapshot()
        current_step = snapshot.get("current_step", {})
        recipe_state = snapshot.get("recipe_state", {})
        features = snapshot.get("features", {})
        realtime = snapshot.get("realtime", {})
        total_steps = len(snapshot.get("recipe", {}).get("steps", []))
        current_index = int(recipe_state.get("current_step_index", 0)) + 1

        self.step_title.setText(f"Step {current_index} of {max(1, total_steps)} — {current_step.get('title', '')}")
        self.step_instruction.setPlainText(current_step.get("instruction", ""))
        tips = current_step.get("tips", [])
        tip_text = f"Tip: {tips[0]}" if tips else "No extra tip for this step."
        vision_hint = "Visual check suggested" if current_step.get("visual_check_suggested") else "Visual check optional"
        self.step_meta.setText(f"{tip_text} • {vision_hint}")

        self.timer_widget.update_timers(snapshot.get("timers", []))
        self.backend_pill.set_value("Connected" if realtime.get("connected") else "Offline", active=bool(realtime.get("connected")))
        self.assistant_pill.set_value("Speaking" if realtime.get("assistant_speaking") else "Quiet", active=bool(realtime.get("assistant_speaking")))
        self.user_pill.set_value(
            "On" if features.get("speech_enabled") else "Muted",
            active=bool(features.get("speech_enabled")),
            warning=not bool(features.get("speech_enabled")),
        )

        for item in self.controller.drain_ui_events():
            etype = item.get("type")
            if etype == "assistant_text":
                self.transcript_panel.append_message("assistant", item.get("text", ""))
            elif etype == "user_text":
                self.transcript_panel.append_message("user", item.get("text", ""))
