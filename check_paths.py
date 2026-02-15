#!/usr/bin/env python3
"""
Скрипт для проверки кроссплатформенности путей.
Запустите на Windows/macOS/Linux чтобы убедиться, что все пути определяются корректно.
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_paths():
    """Проверить все динамические пути."""
    print("=" * 60)
    print("Проверка кроссплатформенности путей")
    print("=" * 60)
    print(f"Платформа: {sys.platform}")
    print(f"Python: {sys.version}")
    print()

    # 1. Пути конфигурации
    print("1. Пути приложения:")
    from app.utils.config import AppConfig

    config_path = AppConfig.get_config_path()
    models_dir = AppConfig.get_models_dir()
    whisper_dir = AppConfig.get_whisper_models_dir()

    print(f"   Config:  {config_path}")
    print(f"   Models:  {models_dir}")
    print(f"   Whisper: {whisper_dir}")
    print()

    # 2. FFmpeg
    print("2. FFmpeg:")
    from app.utils.ffmpeg_finder import find_ffmpeg, find_ffprobe

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffprobe()

    if ffmpeg:
        print(f"   [OK] FFmpeg найден:  {ffmpeg}")
    else:
        print("   [!!] FFmpeg не найден")

    if ffprobe:
        print(f"   [OK] FFprobe найден: {ffprobe}")
    else:
        print("   [!!] FFprobe не найден")
    print()

    # 3. Claude CLI
    print("3. Claude CLI:")
    from app.analysis.claude_cli import _find_claude_executable, is_claude_available

    claude_path = _find_claude_executable()
    if claude_path:
        print(f"   [OK] Claude найден: {claude_path}")
    else:
        print("   [!!] Claude не найден")
        print("   Установите: npm install -g @anthropic-ai/claude-code")
    print()

    # 4. Проверка записи в Logs
    print("4. Логи:")
    from datetime import datetime

    logs_dir = project_root / "Logs"
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = logs_dir / f"transcriber_{today}.log"

    print(f"   Logs dir:  {logs_dir}")
    print(f"   Log file:  {log_file}")
    print(f"   Существует: {log_file.exists()}")
    print()

    # 5. SpeechBrain models
    print("5. SpeechBrain модели:")
    sb_dir = models_dir / "speechbrain" / "spkrec-ecapa-voxceleb"
    print(f"   Path: {sb_dir}")
    print(f"   Существует: {sb_dir.exists()}")
    print()

    print("=" * 60)
    print("Проверка завершена!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_paths()
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
