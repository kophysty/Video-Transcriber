"""Главный пайплайн транскрибации."""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.analysis.analyzer import AnalysisResult

from app.utils.config import AppConfig
from app.utils.ffmpeg_finder import get_audio_duration
from app.core.audio_extractor import AudioExtractor, is_audio_file
from app.core.transcriber import Transcriber, TranscriptionResult, create_transcriber_from_config
from app.core.diarizer import Diarizer, DiarizationResult, create_diarizer_from_config
from app.core.merger import merge_transcription_with_diarization
from app.models.gpu_detector import GPUInfo


@dataclass
class PipelineProgress:
    """Прогресс пайплайна."""
    stage: str  # "extract", "transcribe", "diarize", "export", "analyze"
    stage_progress: float  # 0.0 - 1.0
    total_progress: float  # 0.0 - 1.0
    message: str


@dataclass
class PipelineResult:
    """Результат работы пайплайна."""
    transcription: TranscriptionResult
    diarization: Optional[DiarizationResult]
    analysis: Optional["AnalysisResult"] = None
    input_path: Path = None
    output_dir: Path = None
    duration: float = 0.0


class CancelledException(Exception):
    """Исключение при отмене операции."""
    pass


class TranscriptionPipeline:
    """
    Главный пайплайн транскрибации.

    Этапы:
    1. Извлечение аудио (10%)
    2. Транскрибация (50%)
    3. Диаризация (15%, опционально)
    4. AI-анализ (15%, опционально)
    5. Экспорт (10%)
    """

    # Веса этапов для общего прогресса
    STAGE_WEIGHTS = {
        "extract": 0.10,
        "transcribe": 0.50,
        "diarize": 0.15,
        "analyze": 0.15,
        "export": 0.10,
    }

    def __init__(
        self,
        config: AppConfig,
        gpu_info: Optional[GPUInfo] = None,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ):
        """
        Args:
            config: Конфигурация приложения
            gpu_info: Информация о GPU для управления VRAM
            progress_callback: Callback для отслеживания прогресса
        """
        self.config = config
        self.gpu_info = gpu_info
        self.progress_callback = progress_callback

        self._cancel_event = threading.Event()
        self._current_stage = ""
        self._stage_offset = 0.0

    def cancel(self) -> None:
        """Отменить выполнение."""
        self._cancel_event.set()

    def run(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> PipelineResult:
        """
        Запустить полный пайплайн.

        Args:
            input_path: Путь к входному файлу
            output_dir: Папка для результатов

        Returns:
            PipelineResult с результатами

        Raises:
            CancelledException: Если операция была отменена
            FileNotFoundError: Файл не найден
            RuntimeError: Ошибка обработки
        """
        self._cancel_event.clear()
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Извлечение аудио
        self._check_cancelled()
        audio_path = self._extract_audio(input_path, output_dir)

        try:
            # Получаем длительность
            duration = get_audio_duration(audio_path) or 0.0

            # 2. Транскрибация
            self._check_cancelled()
            transcription, transcriber = self._transcribe(audio_path, duration)

            # 3. Диаризация (опционально)
            diarization = None
            if self.config.enable_diarization:
                self._check_cancelled()

                # Управление VRAM: выгружаем Whisper перед pyannote
                # если VRAM < 16 GB
                if self.gpu_info and self.gpu_info.vram_total_mb < 16000:
                    transcriber.unload_model()

                diarization = self._diarize(audio_path)

            # 4. Объединение результатов
            transcription = merge_transcription_with_diarization(
                transcription, diarization
            )

            # 5. AI-анализ (опционально)
            analysis = None
            if self.config.enable_ai_analysis and self.config.anthropic_api_key:
                self._check_cancelled()
                analysis = self._analyze(transcription)

            # 6. Экспорт
            self._check_cancelled()
            self._export(transcription, diarization, analysis, input_path, output_dir)

            return PipelineResult(
                transcription=transcription,
                diarization=diarization,
                analysis=analysis,
                input_path=input_path,
                output_dir=output_dir,
                duration=duration,
            )

        finally:
            # Очистка временного аудио файла
            if audio_path != input_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass

    def _extract_audio(self, input_path: Path, output_dir: Path) -> Path:
        """Извлечь аудио из видео."""
        self._set_stage("extract")

        # Если уже аудио файл в нужном формате — пропускаем
        if is_audio_file(input_path) and input_path.suffix.lower() == ".wav":
            self._report_progress(1.0, "Аудио файл, конвертация не нужна")
            return input_path

        extractor = AudioExtractor(
            enable_noise_reduction=True,
            progress_callback=self._make_stage_callback(),
        )

        return extractor.extract(input_path, output_dir)

    def _transcribe(
        self,
        audio_path: Path,
        duration: float,
    ) -> tuple[TranscriptionResult, Transcriber]:
        """Транскрибировать аудио."""
        self._set_stage("transcribe")

        transcriber = create_transcriber_from_config(
            self.config,
            progress_callback=self._make_stage_callback(),
        )

        # Язык
        language = self.config.language if self.config.language != "auto" else None

        result = transcriber.transcribe(
            audio_path,
            language=language,
            total_duration=duration,
        )

        return result, transcriber

    def _diarize(self, audio_path: Path) -> DiarizationResult:
        """Выполнить диаризацию."""
        self._set_stage("diarize")

        diarizer = create_diarizer_from_config(
            self.config,
            progress_callback=self._make_stage_callback(),
        )

        return diarizer.diarize(audio_path)

    def _analyze(self, transcription: TranscriptionResult):
        """Выполнить AI-анализ."""
        self._set_stage("analyze")

        from app.analysis.analyzer import TranscriptAnalyzer

        analyzer = TranscriptAnalyzer(
            api_key=self.config.anthropic_api_key,
            progress_callback=self._make_stage_callback(),
        )

        return analyzer.analyze(transcription)

    def _export(
        self,
        transcription: TranscriptionResult,
        diarization: Optional[DiarizationResult],
        analysis,  # Optional[AnalysisResult]
        input_path: Path,
        output_dir: Path,
    ) -> None:
        """Экспортировать результаты."""
        self._set_stage("export")
        self._report_progress(0.0, "Экспорт результатов...")

        from app.exporters.json_exporter import export_json
        from app.exporters.srt_exporter import export_srt
        from app.exporters.vtt_exporter import export_vtt
        from app.exporters.txt_exporter import export_txt

        base_name = input_path.stem

        # JSON
        export_json(
            transcription, diarization,
            output_dir / f"{base_name}.json",
            source_file=input_path.name,
            model=self.config.whisper_model,
        )
        self._report_progress(0.25, "JSON экспортирован")

        # SRT
        export_srt(transcription, output_dir / f"{base_name}.srt")
        self._report_progress(0.5, "SRT экспортирован")

        # VTT
        export_vtt(transcription, output_dir / f"{base_name}.vtt")
        self._report_progress(0.75, "VTT экспортирован")

        # TXT
        export_txt(transcription, output_dir / f"{base_name}.txt")
        self._report_progress(0.8, "TXT экспортирован")

        # AI Analysis (если есть)
        if analysis:
            from app.analysis.analyzer import export_analysis
            export_analysis(analysis, output_dir, base_name)
            self._report_progress(1.0, "Анализ экспортирован")
        else:
            self._report_progress(1.0, "Экспорт завершён")

    def _set_stage(self, stage: str) -> None:
        """Установить текущий этап."""
        self._current_stage = stage

        # Вычисляем смещение для общего прогресса
        offset = 0.0
        for s, weight in self.STAGE_WEIGHTS.items():
            if s == stage:
                break
            # Пропускаем диаризацию если выключена
            if s == "diarize" and not self.config.enable_diarization:
                continue
            # Пропускаем анализ если выключен
            if s == "analyze" and not (self.config.enable_ai_analysis and self.config.anthropic_api_key):
                continue
            offset += weight

        self._stage_offset = offset

    def _make_stage_callback(self) -> Callable[[float, str], None]:
        """Создать callback для текущего этапа."""
        def callback(progress: float, message: str) -> None:
            self._report_progress(progress, message)
        return callback

    def _report_progress(self, stage_progress: float, message: str) -> None:
        """Сообщить о прогрессе."""
        if not self.progress_callback:
            return

        # Вычисляем общий прогресс
        stage_weight = self.STAGE_WEIGHTS.get(self._current_stage, 0.0)

        # Вычисляем суммарный вес активных этапов
        active_weight = 0.0
        for s, w in self.STAGE_WEIGHTS.items():
            if s == "diarize" and not self.config.enable_diarization:
                continue
            if s == "analyze" and not (self.config.enable_ai_analysis and self.config.anthropic_api_key):
                continue
            active_weight += w

        # Нормализуем прогресс чтобы он заполнял 0.0-1.0
        scale = 1.0 / active_weight if active_weight > 0 else 1.0

        total_progress = (self._stage_offset + stage_progress * stage_weight) * scale

        self.progress_callback(PipelineProgress(
            stage=self._current_stage,
            stage_progress=stage_progress,
            total_progress=min(total_progress, 1.0),
            message=message,
        ))

    def _check_cancelled(self) -> None:
        """Проверить, не была ли операция отменена."""
        if self._cancel_event.is_set():
            raise CancelledException("Операция отменена пользователем")
