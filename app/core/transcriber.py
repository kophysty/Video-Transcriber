"""Транскрибация аудио с помощью faster-whisper."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Generator, Any

from app.utils.config import AppConfig


@dataclass
class TranscriptionWord:
    """Слово с таймстампом."""
    start: float
    end: float
    text: str
    probability: float


@dataclass
class TranscriptionSegment:
    """Сегмент транскрипции."""
    start: float
    end: float
    text: str
    words: list[TranscriptionWord]
    speaker: Optional[str] = None  # Заполняется после диаризации


@dataclass
class TranscriptionInfo:
    """Метаданные транскрипции."""
    language: str
    language_probability: float
    duration: float


@dataclass
class TranscriptionResult:
    """Результат транскрибации."""
    segments: list[TranscriptionSegment]
    info: TranscriptionInfo


class Transcriber:
    """Транскрибатор на основе faster-whisper."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        compute_type: str = "float16",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Args:
            model_path: Путь к папке с моделью
            device: "cuda" или "cpu"
            compute_type: "float16", "int8", "float32"
            progress_callback: Callback(progress: 0.0-1.0, message: str)
        """
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.progress_callback = progress_callback
        self._model = None

    def load_model(self) -> None:
        """Загрузить модель Whisper."""
        from faster_whisper import WhisperModel

        self._report_progress(0.0, "Загрузка модели Whisper...")

        self._model = WhisperModel(
            str(self.model_path),
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=True,
        )

        self._report_progress(0.1, "Модель загружена")

    def unload_model(self) -> None:
        """Выгрузить модель для освобождения VRAM."""
        if self._model is not None:
            del self._model
            self._model = None

            # Очистить CUDA кэш
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        total_duration: Optional[float] = None,
    ) -> TranscriptionResult:
        """
        Транскрибировать аудио файл.

        Args:
            audio_path: Путь к WAV файлу (16kHz mono)
            language: Код языка ("ru", "en") или None для автодетекта
            total_duration: Общая длительность для прогресса

        Returns:
            TranscriptionResult с сегментами и метаданными
        """
        if self._model is None:
            self.load_model()

        self._report_progress(0.1, "Начинаем транскрибацию...")

        # Настройки транскрибации
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            language=language if language != "auto" else None,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        # Собираем сегменты и отслеживаем прогресс
        segments = []
        for segment in segments_gen:
            # Конвертируем слова
            words = []
            if segment.words:
                for word in segment.words:
                    words.append(TranscriptionWord(
                        start=word.start,
                        end=word.end,
                        text=word.word,
                        probability=word.probability,
                    ))

            segments.append(TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
            ))

            # Прогресс
            if total_duration and total_duration > 0:
                progress = min(0.1 + (segment.end / total_duration) * 0.8, 0.9)
                self._report_progress(
                    progress,
                    f"Транскрибация: {int(progress * 100)}%"
                )

        self._report_progress(0.9, "Транскрибация завершена")

        return TranscriptionResult(
            segments=segments,
            info=TranscriptionInfo(
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
            ),
        )

    def _report_progress(self, progress: float, message: str) -> None:
        """Сообщить о прогрессе."""
        if self.progress_callback:
            self.progress_callback(progress, message)


def create_transcriber_from_config(
    config: AppConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Transcriber:
    """Создать транскрибатор из конфигурации."""
    return Transcriber(
        model_path=config.get_whisper_model_path(),
        device=config.device,
        compute_type=config.compute_type,
        progress_callback=progress_callback,
    )
