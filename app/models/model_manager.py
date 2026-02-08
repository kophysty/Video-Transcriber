"""Управление моделями: скачивание, удаление, проверка."""

import shutil
from pathlib import Path
from typing import Optional, Callable

from app.utils.config import AppConfig
from app.models.model_registry import (
    WHISPER_MODELS,
    WhisperModelInfo,
    get_model_info,
    get_all_models,
)
from app.utils.formats import format_file_size


class ModelManager:
    """Менеджер моделей."""

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Args:
            models_dir: Папка для моделей (по умолчанию ./models/)
        """
        if models_dir is None:
            models_dir = AppConfig.get_models_dir()

        self.models_dir = models_dir
        self.whisper_dir = models_dir / "whisper"
        # Создаём папки
        self.whisper_dir.mkdir(parents=True, exist_ok=True)

    def list_downloaded_whisper_models(self) -> list[str]:
        """Получить список скачанных Whisper моделей."""
        downloaded = []

        for model_name in WHISPER_MODELS.keys():
            if self.is_whisper_model_downloaded(model_name):
                downloaded.append(model_name)

        return downloaded

    def is_whisper_model_downloaded(self, model_name: str) -> bool:
        """Проверить, скачана ли Whisper модель."""
        model_path = self.whisper_dir / model_name

        if not model_path.exists():
            return False

        # Проверяем наличие ключевых файлов
        # CTranslate2 модели содержат model.bin или model.safetensors
        has_model = (
            (model_path / "model.bin").exists()
            or (model_path / "model.safetensors").exists()
        )

        return has_model

    def get_whisper_model_path(self, model_name: str) -> Path:
        """Получить путь к Whisper модели."""
        return self.whisper_dir / model_name

    def download_whisper_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Скачать Whisper модель.

        Args:
            model_name: Имя модели (например, "large-v3-turbo")
            progress_callback: Callback(progress: 0.0-1.0, message: str)

        Returns:
            Путь к скачанной модели

        Raises:
            ValueError: Неизвестная модель
            RuntimeError: Ошибка скачивания
        """
        model_info = get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Неизвестная модель: {model_name}")

        output_path = self.whisper_dir / model_name

        if progress_callback:
            progress_callback(0.0, f"Скачивание {model_name}...")

        try:
            from huggingface_hub import snapshot_download

            # Скачиваем в локальную папку
            snapshot_download(
                repo_id=model_info.hf_repo,
                local_dir=str(output_path),
                local_dir_use_symlinks=False,  # Копировать файлы, не симлинки
            )

            if progress_callback:
                progress_callback(1.0, f"Модель {model_name} скачана")

            return output_path

        except Exception as e:
            # Удаляем частично скачанную модель
            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)
            raise RuntimeError(f"Ошибка скачивания модели: {e}")

    def delete_whisper_model(self, model_name: str) -> None:
        """Удалить Whisper модель."""
        model_path = self.whisper_dir / model_name

        if model_path.exists():
            shutil.rmtree(model_path)

    def get_whisper_model_size(self, model_name: str) -> int:
        """Получить размер скачанной модели в байтах."""
        model_path = self.whisper_dir / model_name

        if not model_path.exists():
            return 0

        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

    def get_total_models_size(self) -> int:
        """Получить общий размер всех моделей в байтах."""
        total = 0

        if self.models_dir.exists():
            for file_path in self.models_dir.rglob("*"):
                if file_path.is_file():
                    total += file_path.stat().st_size

        return total

    def get_models_status(self) -> dict:
        """Получить статус всех моделей."""
        downloaded_whisper = self.list_downloaded_whisper_models()
        available_whisper = [m for m in WHISPER_MODELS.keys() if m not in downloaded_whisper]

        return {
            "whisper": {
                "downloaded": downloaded_whisper,
                "available": available_whisper,
            },
            "total_size": format_file_size(self.get_total_models_size()),
        }
