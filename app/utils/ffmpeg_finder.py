"""Поиск FFmpeg бинарника."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def find_ffmpeg() -> Optional[Path]:
    """
    Найти FFmpeg бинарник.

    Порядок поиска:
    1. Переменная окружения FFMPEG_PATH
    2. Системный PATH
    3. Бандленный через imageio-ffmpeg

    Returns:
        Path к ffmpeg или None если не найден
    """
    # 1. Проверяем переменную окружения
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists() and _verify_ffmpeg(path):
            return path

    # 2. Проверяем системный PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        path = Path(system_ffmpeg)
        if _verify_ffmpeg(path):
            return path

    # 3. Пробуем imageio-ffmpeg
    try:
        import imageio_ffmpeg
        bundled_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if bundled_path.exists() and _verify_ffmpeg(bundled_path):
            return bundled_path
    except ImportError:
        pass
    except Exception:
        pass

    return None


def find_ffprobe() -> Optional[Path]:
    """
    Найти FFprobe бинарник (для получения метаданных).

    Returns:
        Path к ffprobe или None если не найден
    """
    # Если нашли ffmpeg, ffprobe обычно рядом
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        ffprobe_path = ffmpeg_path.parent / ffmpeg_path.name.replace("ffmpeg", "ffprobe")
        if ffprobe_path.exists():
            return ffprobe_path

    # Проверяем системный PATH
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return Path(system_ffprobe)

    return None


def _verify_ffmpeg(path: Path) -> bool:
    """Проверить что это рабочий FFmpeg."""
    try:
        result = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_audio_duration(file_path: Path) -> Optional[float]:
    """
    Получить длительность аудио/видео файла в секундах.

    Args:
        file_path: Путь к файлу

    Returns:
        Длительность в секундах или None при ошибке
    """
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None

    try:
        result = subprocess.run(
            [
                str(ffprobe),
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return None
