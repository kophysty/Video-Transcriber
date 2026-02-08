"""Конфигурация приложения — загрузка и сохранение настроек."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    """Конфигурация приложения."""

    # Пути
    last_input_dir: str = ""
    last_output_dir: str = ""

    # Модель
    whisper_model: str = "large-v3-turbo"
    compute_type: str = "float16"
    device: str = "cuda"

    # Транскрибация
    language: str = "auto"  # "auto", "ru", "en", etc.
    enable_diarization: bool = False

    # HuggingFace
    hf_token: str = ""

    # AI Analysis (Claude CLI — основной движок, API ключ опционально для web search)
    enable_ai_analysis: bool = False
    anthropic_api_key: str = ""  # Опционально: для поиска ссылок через web search

    # UI
    theme: str = "dark"

    @classmethod
    def get_config_path(cls) -> Path:
        """Путь к файлу конфигурации — в папке приложения."""
        return Path(__file__).parent.parent.parent / "config.json"

    @classmethod
    def get_models_dir(cls) -> Path:
        """Путь к папке с моделями."""
        return Path(__file__).parent.parent.parent / "models"

    @classmethod
    def get_whisper_models_dir(cls) -> Path:
        """Путь к папке с Whisper моделями."""
        return cls.get_models_dir() / "whisper"

    @classmethod
    def get_pyannote_models_dir(cls) -> Path:
        """Путь к папке с pyannote моделями."""
        return cls.get_models_dir() / "pyannote"

    @classmethod
    def load(cls) -> "AppConfig":
        """Загрузить конфигурацию из файла или создать дефолтную."""
        config_path = cls.get_config_path()

        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Ошибка загрузки конфига: {e}, используем дефолтный")

        return cls()

    def save(self) -> None:
        """Сохранить конфигурацию в файл."""
        config_path = self.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def get_whisper_model_path(self) -> Path:
        """Путь к выбранной Whisper модели."""
        return self.get_whisper_models_dir() / self.whisper_model
