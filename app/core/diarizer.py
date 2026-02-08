"""Диаризация спикеров с помощью pyannote.audio."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from app.utils.config import AppConfig

log = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    """Сегмент диаризации."""
    start: float
    end: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", etc.


@dataclass
class DiarizationResult:
    """Результат диаризации."""
    segments: list[DiarizationSegment]
    num_speakers: int


class Diarizer:
    """Диаризатор на основе pyannote.audio."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        hf_token: Optional[str] = None,
        device: str = "cuda",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Args:
            model_path: Путь к локальной модели или None для загрузки из HF
            hf_token: HuggingFace токен (нужен для скачивания модели)
            device: "cuda" или "cpu"
            progress_callback: Callback(progress: 0.0-1.0, message: str)
        """
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = device
        self.progress_callback = progress_callback
        self._pipeline = None

    def load_pipeline(self) -> None:
        """Загрузить пайплайн диаризации."""
        from pyannote.audio import Pipeline
        import torch

        self._report_progress(0.0, "Загрузка модели диаризации...")

        # Загружаем модель
        if self.model_path and self.model_path.exists():
            log.info("Загрузка pyannote из локальной папки: %s", self.model_path)
            try:
                self._pipeline = Pipeline.from_pretrained(str(self.model_path))
                log.info("Локальная модель pyannote загружена успешно")
            except Exception as e:
                log.warning("Не удалось загрузить локальную модель: %s", e)
                log.info("Попытка загрузки из HuggingFace...")
                self._load_from_hf(Pipeline)
        else:
            log.info("Локальная модель не найдена: %s", self.model_path)
            self._load_from_hf(Pipeline)

        # Переносим на устройство
        if self.device == "cuda" and torch.cuda.is_available():
            log.info("Перенос pyannote на CUDA")
            self._pipeline.to(torch.device("cuda"))
        else:
            log.info("Pyannote работает на CPU (device=%s, cuda=%s)",
                     self.device, torch.cuda.is_available())

        self._report_progress(0.1, "Модель диаризации загружена")

    def _load_from_hf(self, Pipeline) -> None:
        """Загрузить модель из HuggingFace."""
        if not self.hf_token:
            raise ValueError(
                "Для загрузки модели pyannote нужен HuggingFace токен. "
                "Получите его на https://huggingface.co/settings/tokens"
            )
        log.info("Загрузка pyannote из HuggingFace (токен: %s...)",
                 self.hf_token[:8] if len(self.hf_token) > 8 else "***")
        log.info("Убедитесь, что приняли лицензию на:")
        log.info("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        log.info("  https://huggingface.co/pyannote/segmentation-3.0")
        log.info("  https://huggingface.co/pyannote/speaker-diarization-community-1")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token,
        )
        log.info("Модель pyannote загружена из HuggingFace")

    def unload_pipeline(self) -> None:
        """Выгрузить пайплайн для освобождения VRAM."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

            # Очистить CUDA кэш
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def diarize(
        self,
        audio_path: Path,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Выполнить диаризацию аудио файла.

        Args:
            audio_path: Путь к WAV файлу
            num_speakers: Точное количество спикеров (если известно)
            min_speakers: Минимальное количество спикеров
            max_speakers: Максимальное количество спикеров

        Returns:
            DiarizationResult с сегментами и количеством спикеров
        """
        if self._pipeline is None:
            self.load_pipeline()

        self._report_progress(0.1, "Диаризация спикеров...")
        log.info("Запуск диаризации: %s", audio_path.name)

        # Параметры
        diarization_params = {}
        if num_speakers is not None:
            diarization_params["num_speakers"] = num_speakers
        if min_speakers is not None:
            diarization_params["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_params["max_speakers"] = max_speakers

        if diarization_params:
            log.info("Параметры диаризации: %s", diarization_params)

        # Запускаем диаризацию
        # Примечание: pyannote не даёт промежуточный прогресс,
        # поэтому просто ждём завершения
        log.info("Диаризация в процессе (это может занять несколько минут)...")
        diarization = self._pipeline(str(audio_path), **diarization_params)

        # Конвертируем результат
        segments = []
        speakers = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))
            speakers.add(speaker)

        log.info("Диаризация завершена: %d сегментов, %d спикеров",
                 len(segments), len(speakers))
        self._report_progress(1.0, "Диаризация завершена")

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers),
        )

    def _report_progress(self, progress: float, message: str) -> None:
        """Сообщить о прогрессе."""
        if self.progress_callback:
            self.progress_callback(progress, message)


def create_diarizer_from_config(
    config: AppConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Diarizer:
    """Создать диаризатор из конфигурации."""
    return Diarizer(
        model_path=config.get_pyannote_models_dir() / "speaker-diarization-3.1",
        hf_token=config.hf_token if config.hf_token else None,
        device=config.device,
        progress_callback=progress_callback,
    )
