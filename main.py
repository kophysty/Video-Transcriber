#!/usr/bin/env python3
"""Video-Transcriber: точка входа приложения."""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Запустить приложение."""
    from app.gui.main_window import MainWindow

    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
