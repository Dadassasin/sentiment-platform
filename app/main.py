"""Application entry point."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

try:
    from qt_material import apply_stylesheet
except ImportError:  # pragma: no cover - optional dependency guard
    apply_stylesheet = None

from app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Лаборатория анализа тональности")
    app.setOrganizationName("Sentiment Platform")

    if apply_stylesheet is not None:
        apply_stylesheet(app, theme="light_blue.xml", invert_secondary=True)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

