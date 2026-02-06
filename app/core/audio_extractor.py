"""Извлечение аудио из видео с помощью FFmpeg."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable

from app.utils.ffmpeg_finder import find_ffmpeg, get_audio_duration


# Поддерживаемые форматы
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv"}


class AudioExtractor:
    """Извлекает аудио из видео/аудио файлов в формат для Whisper."""

    def __init__(
        self,
        enable_noise_reduction: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Args:
            enable_noise_reduction: Включить шумоподавление
            progress_callback: Callback(progress: 0.0-1.0, message: str)
        """
        self.enable_noise_reduction = enable_noise_reduction
        self.progress_callback = progress_callback
        self._ffmpeg_path = find_ffmpeg()

    def extract(self, input_path: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Извлечь аудио из файла.

        Args:
            input_path: Путь к входному файлу (видео или аудио)
            output_dir: Папка для выходного файла (если None, используем temp)

        Returns:
            Путь к извлечённому WAV файлу

        Raises:
            FileNotFoundError: FFmpeg не найден
            RuntimeError: Ошибка извлечения
        """
        if not self._ffmpeg_path:
            raise FileNotFoundError(
                "FFmpeg не найден. Установите FFmpeg или пакет imageio-ffmpeg."
            )

        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")

        # Определяем выходной файл
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}_audio.wav"
        else:
            output_path = Path(tempfile.mktemp(suffix=".wav"))

        # Получаем длительность для прогресса
        duration = get_audio_duration(input_path)

        # Строим команду FFmpeg
        cmd = self._build_ffmpeg_command(input_path, output_path)

        self._report_progress(0.0, "Извлечение аудио...")

        # Запускаем FFmpeg
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Парсим вывод для прогресса
        self._monitor_progress(process, duration)

        # Проверяем результат
        return_code = process.wait()
        if return_code != 0:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"FFmpeg завершился с ошибкой {return_code}: {stderr}")

        if not output_path.exists():
            raise RuntimeError(f"Выходной файл не создан: {output_path}")

        self._report_progress(1.0, "Аудио извлечено")
        return output_path

    def _build_ffmpeg_command(self, input_path: Path, output_path: Path) -> list[str]:
        """Построить команду FFmpeg."""
        cmd = [
            str(self._ffmpeg_path),
            "-y",  # Перезаписывать без вопросов
            "-i", str(input_path),
            "-vn",  # Без видео
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16kHz (требование Whisper)
            "-ac", "1",  # Моно
        ]

        # Аудио фильтры
        if self.enable_noise_reduction:
            # highpass: убрать низкочастотный гул <100Hz
            # lowpass: убрать высокочастотный шум >8kHz
            # afftdn: FFT-based шумоподавление
            cmd.extend(["-af", "highpass=f=100,lowpass=f=8000,afftdn=nf=-20"])

        cmd.append(str(output_path))
        return cmd

    def _monitor_progress(
        self,
        process: subprocess.Popen,
        total_duration: Optional[float],
    ) -> None:
        """Мониторить прогресс FFmpeg."""
        if not process.stderr:
            return

        time_pattern = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")

        for line in process.stderr:
            if not total_duration:
                continue

            match = time_pattern.search(line)
            if match:
                hours, minutes, seconds, centiseconds = map(int, match.groups())
                current_time = hours * 3600 + minutes * 60 + seconds + centiseconds / 100

                progress = min(current_time / total_duration, 0.99)
                self._report_progress(progress, f"Извлечение: {int(progress * 100)}%")

    def _report_progress(self, progress: float, message: str) -> None:
        """Сообщить о прогрессе."""
        if self.progress_callback:
            self.progress_callback(progress, message)


def is_audio_file(path: Path) -> bool:
    """Проверить, является ли файл аудио."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Проверить, является ли файл видео."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_supported_file(path: Path) -> bool:
    """Проверить, поддерживается ли файл."""
    return is_audio_file(path) or is_video_file(path)
