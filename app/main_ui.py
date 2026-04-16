from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.config import AppConfig
from app.controller import DesktopController
from app.logging_setup import configure_logging


def main() -> int:
    config = AppConfig()
    configure_logging(config.log_level)
    app = QApplication(sys.argv)
    controller = DesktopController(config)
    from app.ui.app_window import AppWindow
    window = AppWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
