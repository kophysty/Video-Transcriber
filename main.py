#!/usr/bin/env python3
"""PG-Video-Transcriber: точка входа приложения."""

import logging
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def setup_logging() -> None:
    """Настроить логирование в консоль и файл."""
    from datetime import datetime

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"

    # Создаем папку Logs, если её нет
    logs_dir = project_root / "Logs"
    logs_dir.mkdir(exist_ok=True)

    # Файл логов с датой: Logs/transcriber_2025-02-12.log
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = logs_dir / f"transcriber_{today}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )
    # Приглушаем слишком шумные библиотеки
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)

    log = logging.getLogger("app")
    log.info(f"Логи сохраняются в: {log_file}")


def main():
    """Запустить приложение."""
    setup_logging()
    log = logging.getLogger("app")
    log.info("PG-Video-Transcriber запущен")

    from app.gui.main_window import MainWindow

    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
