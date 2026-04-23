from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea, QVBoxLayout, QWidget

from app.config import AppConfig
from app.controller import DesktopController
from app.core.recipe import RecipeLoader
from app.tts.kokoro_tts import KokoroTTS
from app.ui.widgets import CardFrame, RecipeOptionCard


class WarmupWorker(QThread):
    status_changed = Signal(str)
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            self.status_changed.emit("Checking local voice assets…")
            kokoro = KokoroTTS(
                model_dir=self.config.kokoro_model_dir,
                repo_id=self.config.kokoro_repo_id,
                voice=self.config.kokoro_voice,
                lang_code=self.config.kokoro_lang_code,
                speed=self.config.kokoro_speed,
            )
            kokoro.ensure_ready()
            self.status_changed.emit("Voice assets ready. Kitchen is preheated.")
            self.finished_ok.emit()
        except Exception as exc:  # pragma: no cover - UI worker
            self.failed.emit(str(exc))


class StartupWindow(QMainWindow):
    def __init__(self, base_config: AppConfig) -> None:
        super().__init__()
        self.base_config = base_config
        self.selected_recipe_path: str | None = None
        self._cards: list[RecipeOptionCard] = []
        self._warmup_ready = False
        self._controller_window = None
        self._closing_for_navigation = False
        self._dots = 0

        self.setWindowTitle("LLmmm")
        self.resize(1180, 760)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(26, 26, 26, 26)
        root.setSpacing(18)

        heading = QLabel("Pick a recipe and get the kitchen ready")
        heading.setObjectName("headline")
        subtitle = QLabel("The assistant can warm up its local voice assets while you choose what to make.")
        subtitle.setObjectName("subtle")
        root.addWidget(heading)
        root.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(18)
        root.addLayout(content, stretch=1)

        left_card = CardFrame("Tonight's menu", accent=True)
        self.recipes_grid = QGridLayout()
        self.recipes_grid.setSpacing(14)
        grid_wrapper = QWidget()
        grid_wrapper.setLayout(self.recipes_grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_wrapper)
        left_card.body.addWidget(scroll)
        content.addWidget(left_card, stretch=3)

        right_col = QVBoxLayout()
        content.addLayout(right_col, stretch=2)

        self.detail_card = CardFrame("Recipe details")
        self.detail_title = QLabel("Select a recipe")
        self.detail_title.setStyleSheet("font-size: 18pt; font-weight: 700;")
        self.detail_description = QLabel("Choose one of the available recipes to see the steps and launch the cooking workspace.")
        self.detail_description.setWordWrap(True)
        self.detail_meta = QLabel("")
        self.detail_meta.setObjectName("subtle")
        self.detail_steps = QLabel("")
        self.detail_steps.setWordWrap(True)
        self.detail_card.body.addWidget(self.detail_title)
        self.detail_card.body.addWidget(self.detail_description)
        self.detail_card.body.addWidget(self.detail_meta)
        self.detail_card.body.addWidget(self.detail_steps)
        right_col.addWidget(self.detail_card)

        self.status_card = CardFrame("Startup status")
        self.status_label = QLabel("Warming up local assets")
        self.status_detail = QLabel("Please wait while the voice stack gets ready.")
        self.status_detail.setWordWrap(True)
        self.status_detail.setObjectName("subtle")
        self.launch_button = QPushButton("Choose a recipe first")
        self.launch_button.setEnabled(False)
        self.launch_button.clicked.connect(self._launch)
        self.status_card.body.addWidget(self.status_label)
        self.status_card.body.addWidget(self.status_detail)
        self.status_card.body.addStretch(1)
        self.status_card.body.addWidget(self.launch_button)
        right_col.addWidget(self.status_card)
        right_col.addStretch(1)

        self._populate_recipes()
        self._start_warmup()

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_status)
        self._pulse_timer.start(500)

    def _populate_recipes(self) -> None:
        recipes = RecipeLoader.discover(Path("recipes"))
        for index, (recipe_path, recipe) in enumerate(recipes):
            card = RecipeOptionCard(str(recipe_path), recipe.name, recipe.description, f"{len(recipe.steps)} steps")
            card.clicked.connect(self._select_recipe)
            self._cards.append(card)
            self.recipes_grid.addWidget(card, index // 2, index % 2)
        if recipes:
            self._select_recipe(str(recipes[0][0]))

    def _select_recipe(self, recipe_path: str) -> None:
        self.selected_recipe_path = recipe_path
        recipe = RecipeLoader.load(recipe_path)
        for card in self._cards:
            card.set_selected(card.recipe_path == recipe_path)
        self.detail_title.setText(recipe.name)
        self.detail_description.setText(recipe.description)
        self.detail_meta.setText(f"{len(recipe.steps)} steps • file: {Path(recipe_path).name}")
        preview_steps = "\n".join(f"{idx + 1}. {step.title}" for idx, step in enumerate(recipe.steps[:5]))
        if len(recipe.steps) > 5:
            preview_steps += "\n…"
        self.detail_steps.setText(preview_steps)
        self._refresh_launch_state()

    def _start_warmup(self) -> None:
        self.worker = WarmupWorker(self.base_config)
        self.worker.status_changed.connect(self._set_status)
        self.worker.finished_ok.connect(self._on_warmup_ready)
        self.worker.failed.connect(self._on_warmup_failed)
        self.worker.start()

    def _tick_status(self) -> None:
        if self._warmup_ready:
            return
        self._dots = (self._dots + 1) % 4
        self.status_label.setText("Warming up local assets" + "." * self._dots)

    def _set_status(self, text: str) -> None:
        self.status_detail.setText(text)

    def _on_warmup_ready(self) -> None:
        self._warmup_ready = True
        self.status_label.setText("Everything is ready")
        self.status_detail.setText("Voice assets are ready, so the full assistant should open faster.")
        self._refresh_launch_state()

    def _on_warmup_failed(self, message: str) -> None:
        self._warmup_ready = True
        self.status_label.setText("Warmup skipped")
        self.status_detail.setText(f"The app can still launch. Warmup error: {message}")
        self._refresh_launch_state()

    def _refresh_launch_state(self) -> None:
        can_launch = bool(self.selected_recipe_path) and self._warmup_ready
        self.launch_button.setEnabled(can_launch)
        if not self.selected_recipe_path:
            self.launch_button.setText("Choose a recipe first")
        elif not self._warmup_ready:
            self.launch_button.setText("Finishing startup prep…")
        else:
            recipe = RecipeLoader.load(self.selected_recipe_path)
            self.launch_button.setText(f"Start cooking: {recipe.name}")

    def _launch(self) -> None:
        if not self.selected_recipe_path:
            return
        config = self.base_config.model_copy(update={"recipe_path": self.selected_recipe_path})
        controller = DesktopController(config)
        from app.ui.app_window import AppWindow
        self._closing_for_navigation = True
        self._controller_window = AppWindow(controller)
        self._controller_window.show()
        self.close()

    def closeEvent(self, event):  # noqa: ANN001, N802
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(1500)
        if not self._closing_for_navigation and self._controller_window is None:
            QApplication.instance().quit()
        super().closeEvent(event)
