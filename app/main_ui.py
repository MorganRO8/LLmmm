from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from app.logging_setup import configure_logging
from app.ui.startup_window import StartupWindow
from app.ui.theme import APP_STYLESHEET


def main() -> int:
    config = AppConfig()
    configure_logging(config.log_level)
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = StartupWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
