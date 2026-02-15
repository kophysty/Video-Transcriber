"""Скачивание медиа по URL через yt-dlp."""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Результат скачивания."""
    audio_path: Path       # путь к скачанному WAV файлу
    title: str             # название видео (для имени выходных файлов)
    duration: float        # длительность в секундах
    source_url: str        # исходный URL


def is_url(text: str) -> bool:
    """Определяет, является ли строка URL (http/https)."""
    return bool(re.match(r'^https?://', text.strip()))


def _sanitize_filename(title: str) -> str:
    """Очистить название для использования в имени файла."""
    # Убираем символы запрещённые в Windows-путях
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    # Убираем лишние пробелы
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # Ограничиваем длину
    if len(sanitized) > 150:
        sanitized = sanitized[:150].rsplit(' ', 1)[0]
    return sanitized or "download"


def download_audio(
    url: str,
    output_dir: Path,
    ffmpeg_path: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> DownloadResult:
    """
    Скачать аудио по URL через yt-dlp.

    Args:
        url: URL видео (YouTube, Vimeo, и т.д.)
        output_dir: Папка для сохранения
        ffmpeg_path: Путь к FFmpeg (если None — yt-dlp ищет сам)
        progress_callback: callback(progress: 0.0-1.0, message: str)

    Returns:
        DownloadResult с путём к WAV файлу и метаданными

    Raises:
        ImportError: если yt-dlp не установлен
        Exception: при ошибке скачивания
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp не установлен.\n"
            "Установите: pip install yt-dlp"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Шаблон имени файла — используем уникальный ID чтобы избежать конфликтов
    output_template = str(output_dir / "%(id)s.%(ext)s")

    # Progress hook для yt-dlp
    def progress_hook(d: dict) -> None:
        if not progress_callback:
            return

        status = d.get("status", "")

        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = downloaded / total
                speed = d.get("speed")
                speed_str = ""
                if speed:
                    if speed > 1_000_000:
                        speed_str = f" ({speed / 1_000_000:.1f} MB/s)"
                    else:
                        speed_str = f" ({speed / 1_000:.0f} KB/s)"
                progress_callback(pct * 0.7, f"Скачивание: {int(pct * 100)}%{speed_str}")
            else:
                # Неизвестный размер
                downloaded_mb = downloaded / 1_000_000
                progress_callback(0.3, f"Скачивание: {downloaded_mb:.1f} MB")

        elif status == "finished":
            progress_callback(0.7, "Конвертация в WAV...")

    # Опции yt-dlp — скачиваем аудио БЕЗ постпроцессинга
    # (FFmpegExtractAudio требует ffprobe, которого нет в imageio-ffmpeg)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,  # Скачиваем только одно видео, не плейлист
    }

    log.info("Скачивание: %s", url)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Извлекаем информацию
        info = ydl.extract_info(url, download=True)

        title = info.get("title", "video")
        duration = info.get("duration") or 0.0
        video_id = info.get("id", "unknown")

        log.info("Название: %s, длительность: %.0f сек", title, duration)

    # Находим скачанный файл (может быть webm, m4a, opus и т.д.)
    downloaded_path = None
    candidates = list(output_dir.glob(f"{video_id}.*"))
    if candidates:
        downloaded_path = candidates[0]
    else:
        raise FileNotFoundError(
            f"Скачанный файл не найден для ID: {video_id}"
        )

    log.info("Скачан файл: %s", downloaded_path.name)

    # Конвертируем в WAV 16kHz моно через ffmpeg (из imageio-ffmpeg)
    wav_path = output_dir / f"{video_id}.wav"
    if downloaded_path.suffix.lower() != ".wav":
        if progress_callback:
            progress_callback(0.75, "Конвертация в WAV...")

        _convert_to_wav(downloaded_path, wav_path, ffmpeg_path)

        # Удаляем исходный скачанный файл
        try:
            downloaded_path.unlink()
        except OSError:
            pass
    else:
        wav_path = downloaded_path

    if progress_callback:
        progress_callback(1.0, "Скачивание завершено")

    safe_title = _sanitize_filename(title)
    log.info("Скачано: %s → %s", safe_title, wav_path.name)

    return DownloadResult(
        audio_path=wav_path,
        title=safe_title,
        duration=duration,
        source_url=url,
    )


def _convert_to_wav(input_path: Path, output_path: Path, ffmpeg_path: Optional[str] = None) -> None:
    """Конвертировать аудио файл в WAV 16kHz моно через ffmpeg."""
    # Определяем путь к ffmpeg
    ffmpeg_bin = ffmpeg_path
    if not ffmpeg_bin:
        from app.utils.ffmpeg_finder import find_ffmpeg
        found = find_ffmpeg()
        if found:
            ffmpeg_bin = str(found)
        else:
            raise FileNotFoundError(
                "FFmpeg не найден. Установите FFmpeg или пакет imageio-ffmpeg."
            )

    cmd = [
        ffmpeg_bin,
        "-i", str(input_path),
        "-vn",                   # без видео
        "-acodec", "pcm_s16le",  # WAV PCM 16-bit
        "-ar", "16000",          # 16kHz
        "-ac", "1",              # моно
        "-y",                    # перезаписать
        str(output_path),
    ]

    log.info("Конвертация: %s → %s", input_path.name, output_path.name)

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=300,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg конвертация не удалась: {stderr[:500]}")
