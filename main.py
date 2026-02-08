#!/usr/bin/env python3
"""Video-Transcriber: точка входа приложения."""

import logging
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_logging() -> None:
    """Настроить логирование в консоль."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Приглушаем слишком шумные библиотеки
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


def main():
    """Запустить приложение."""
    setup_logging()
    log = logging.getLogger("app")
    log.info("Video-Transcriber запущен")

    from app.gui.main_window import MainWindow

    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
