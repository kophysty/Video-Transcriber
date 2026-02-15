"""Главный пайплайн транскрибации."""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.analysis.analyzer import AnalysisResult

from app.utils.config import AppConfig
from app.utils.ffmpeg_finder import get_audio_duration
from app.core.audio_extractor import AudioExtractor, is_audio_file
from app.core.transcriber import Transcriber, TranscriptionResult, create_transcriber_from_config
from app.core.diarizer import DiarizationResult, create_diarizer_from_config
from app.core.merger import merge_transcription_with_diarization
from app.models.gpu_detector import GPUInfo

log = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    """Прогресс пайплайна."""
    stage: str  # "extract", "transcribe", "diarize", "export", "analyze"
    stage_progress: float  # 0.0 - 1.0
    total_progress: float  # 0.0 - 1.0
    message: str
    elapsed_seconds: float = 0.0
    eta_seconds: float = 0.0


@dataclass
class PipelineResult:
    """Результат работы пайплайна."""
    transcription: TranscriptionResult
    diarization: Optional[DiarizationResult]
    analysis: Optional["AnalysisResult"] = None
    input_path: Path = None
    output_dir: Path = None
    duration: float = 0.0
    warnings: list[str] = field(default_factory=list)


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
        self.config = config
        self.gpu_info = gpu_info
        self.progress_callback = progress_callback

        self._cancel_event = threading.Event()
        self._current_stage = ""
        self._stage_offset = 0.0
        self._start_time = 0.0
        self._skip_transcription = False

    def cancel(self) -> None:
        """Отменить выполнение."""
        self._cancel_event.set()

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        existing_transcription: Optional['TranscriptionResult'] = None,
        base_name: Optional[str] = None,
    ) -> PipelineResult:
        """
        Запустить пайплайн.

        Args:
            input_path: Путь к входному файлу
            output_dir: Папка для результатов
            existing_transcription: Если передан — пропускаем транскрибацию,
                                    используем существующий результат
            base_name: Имя для выходных файлов (если None — берётся из input_path.stem)
        """
        self._cancel_event.clear()
        self._start_time = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        skip_transcription = existing_transcription is not None
        self._skip_transcription = skip_transcription

        log.info("=" * 60)
        log.info("Начало обработки: %s", input_path.name)
        log.info("Выходная папка: %s", output_dir)
        if skip_transcription:
            log.info("Режим: повторная обработка (транскрибация пропущена)")
        else:
            log.info("Модель: %s, устройство: %s, compute: %s",
                     self.config.whisper_model, self.config.device, self.config.compute_type)
        log.info("Диаризация: %s, AI-анализ: %s",
                 self.config.enable_diarization, self.config.enable_ai_analysis)
        log.info("=" * 60)

        audio_path = input_path  # для диаризации нужен аудио файл

        if skip_transcription:
            # Используем существующую транскрибацию
            transcription = existing_transcription
            duration = transcription.info.duration if transcription.info else 0.0
            log.info("Загружена транскрибация: %d сегментов, %.1f мин",
                     len(transcription.segments), duration / 60)

            # Для диаризации нужен аудио файл — извлекаем если надо
            if self.config.enable_diarization:
                self._check_cancelled()
                log.info("[1] Извлечение аудио для диаризации...")
                audio_path = self._extract_audio(input_path, output_dir)
                log.info("[1] Аудио готово: %s", audio_path.name)
        else:
            # 1. Извлечение аудио
            self._check_cancelled()
            log.info("[1/5] Извлечение аудио...")
            audio_path = self._extract_audio(input_path, output_dir)
            log.info("[1/5] Аудио готово: %s", audio_path.name)

        try:
            if not skip_transcription:
                # Получаем длительность
                duration = get_audio_duration(audio_path) or 0.0
                if duration > 0:
                    log.info("Длительность аудио (ffprobe): %.1f сек (%.1f мин)", duration, duration / 60)
                else:
                    log.warning("ffprobe не определил длительность, будет использована из faster-whisper")

                # 2. Транскрибация
                self._check_cancelled()
                log.info("[2/5] Транскрибация...")
                t0 = time.monotonic()
                transcription, transcriber = self._transcribe(audio_path, duration)
                t_transcribe = time.monotonic() - t0

                # Используем длительность из faster-whisper если ffprobe не сработал
                if duration <= 0 and transcription.info:
                    duration = transcription.info.duration
                    log.info("Длительность из faster-whisper: %.1f сек (%.1f мин)", duration, duration / 60)

                log.info("[2/5] Транскрибация завершена за %.1f сек, сегментов: %d, язык: %s",
                         t_transcribe, len(transcription.segments),
                         transcription.info.language if transcription.info else "?")

                # Раннее сохранение JSON — чтобы при крэше на следующих этапах результат не терялся
                try:
                    from app.exporters.json_exporter import export_json
                    _base = base_name or input_path.stem
                    export_json(
                        transcription, None,
                        output_dir / f"{_base}.json",
                        source_file=input_path.name,
                        model=self.config.whisper_model,
                    )
                    log.info("Транскрипция сохранена (промежуточно): %s.json", _base)
                except Exception as e:
                    log.warning("Не удалось сохранить промежуточный JSON: %s", e)

            # 3. Диаризация (опционально, с graceful fallback)
            diarization = None
            if self.config.enable_diarization:
                self._check_cancelled()
                log.info("[3/5] Диаризация спикеров...")

                try:
                    t0 = time.monotonic()
                    diarization = self._diarize(audio_path)
                    t_diarize = time.monotonic() - t0
                    log.info("[3/5] Диаризация завершена за %.1f сек, спикеров: %d",
                             t_diarize, diarization.num_speakers if diarization else 0)
                except Exception as e:
                    warn_msg = f"Диаризация пропущена: {e}"
                    warnings.append(warn_msg)
                    log.warning("[3/5] %s", warn_msg)
                    self._report_progress(1.0, f"Диаризация пропущена: ошибка")
            else:
                log.info("[3/5] Диаризация отключена, пропускаем")

            # 4. Объединение результатов
            transcription = merge_transcription_with_diarization(
                transcription, diarization
            )

            # 4.5. Идентификация спикеров (если диаризация успешна + Claude CLI)
            speaker_map: dict[str, str] = {}
            if diarization is not None:
                from app.analysis.claude_cli import is_claude_available, call_claude
                if is_claude_available():
                    try:
                        from app.analysis.speaker_identifier import identify_speakers
                        # Подготовка текста с метками спикеров
                        speaker_lines = []
                        for seg in transcription.segments:
                            if seg.speaker:
                                speaker_lines.append(f"{seg.speaker}: {seg.text}")
                            else:
                                speaker_lines.append(seg.text)
                        speaker_text = "\n".join(speaker_lines)

                        log.info("Идентификация спикеров через Claude CLI...")
                        speaker_map = identify_speakers(call_claude, speaker_text)
                        if speaker_map:
                            log.info("Определены имена спикеров: %s", speaker_map)
                        else:
                            log.info("Имена спикеров не удалось определить из контекста")
                    except Exception as e:
                        log.warning("Идентификация спикеров пропущена: %s", e)
                else:
                    log.info("Claude CLI недоступен, спикеры останутся с метками")

            # 5. AI-анализ (опционально, с graceful fallback)
            analysis = None
            if self.config.enable_ai_analysis:
                self._check_cancelled()
                log.info("[4/5] AI-анализ...")
                try:
                    t0 = time.monotonic()
                    analysis = self._analyze(transcription)
                    t_analyze = time.monotonic() - t0
                    log.info("[4/5] AI-анализ завершён за %.1f сек", t_analyze)
                except Exception as e:
                    warn_msg = f"AI-анализ пропущен: {e}"
                    warnings.append(warn_msg)
                    log.warning("[4/5] %s", warn_msg)
                    self._report_progress(1.0, f"AI-анализ пропущен: ошибка")
            else:
                log.info("[4/5] AI-анализ отключён, пропускаем")

            # 6. Экспорт (всегда выполняется если есть транскрипция)
            self._check_cancelled()
            log.info("[5/5] Экспорт результатов...")
            self._export(transcription, diarization, analysis, input_path, output_dir, speaker_map, base_name)

            total_time = time.monotonic() - self._start_time
            log.info("=" * 60)
            log.info("Обработка завершена за %.1f сек (%.1f мин)", total_time, total_time / 60)
            if warnings:
                log.info("Предупреждения: %s", "; ".join(warnings))
            log.info("=" * 60)

            return PipelineResult(
                transcription=transcription,
                diarization=diarization,
                analysis=analysis,
                input_path=input_path,
                output_dir=output_dir,
                duration=duration,
                warnings=warnings,
            )

        finally:
            # Очистка временного аудио файла
            if audio_path != input_path and audio_path.exists():
                try:
                    audio_path.unlink()
                    log.info("Временный аудио файл удалён")
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
        """Выполнить диаризацию (всегда на CPU — безопасно для CUDA)."""
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
            progress_callback=self._make_stage_callback(),
        )

        return analyzer.analyze(
            transcription,
            api_key=self.config.anthropic_api_key or None,
        )

    def _export(
        self,
        transcription: TranscriptionResult,
        diarization: Optional[DiarizationResult],
        analysis,  # Optional[AnalysisResult]
        input_path: Path,
        output_dir: Path,
        speaker_map: dict[str, str] | None = None,
        base_name: Optional[str] = None,
    ) -> None:
        """Экспортировать результаты."""
        self._set_stage("export")
        self._report_progress(0.0, "Экспорт результатов...")

        from app.exporters.json_exporter import export_json
        from app.exporters.srt_exporter import export_srt
        from app.exporters.vtt_exporter import export_vtt
        from app.exporters.txt_exporter import export_txt
        from app.exporters.md_exporter import export_md, export_diarized_md

        base_name = base_name or input_path.stem

        # MD (сплошной текст — всегда)
        export_md(transcription, output_dir / f"{base_name}.md")
        log.info("  Экспорт: %s.md", base_name)

        # Diarized MD (если есть диаризация)
        if diarization is not None:
            export_diarized_md(
                transcription,
                output_dir / f"{base_name}_diarized.md",
                speaker_map=speaker_map,
            )
            log.info("  Экспорт: %s_diarized.md", base_name)

        self._report_progress(0.2, "MD экспортирован")

        # JSON
        export_json(
            transcription, diarization,
            output_dir / f"{base_name}.json",
            source_file=input_path.name,
            model=self.config.whisper_model,
        )
        log.info("  Экспорт: %s.json", base_name)
        self._report_progress(0.4, "JSON экспортирован")

        # SRT
        export_srt(transcription, output_dir / f"{base_name}.srt")
        log.info("  Экспорт: %s.srt", base_name)
        self._report_progress(0.55, "SRT экспортирован")

        # VTT
        export_vtt(transcription, output_dir / f"{base_name}.vtt")
        log.info("  Экспорт: %s.vtt", base_name)
        self._report_progress(0.7, "VTT экспортирован")

        # TXT
        export_txt(transcription, output_dir / f"{base_name}.txt")
        log.info("  Экспорт: %s.txt", base_name)
        self._report_progress(0.8, "TXT экспортирован")

        # AI Analysis (если есть)
        if analysis:
            from app.analysis.analyzer import export_analysis
            export_analysis(analysis, output_dir, base_name)
            log.info("  Экспорт: %s_analysis.json + .md", base_name)
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
            # Пропускаем извлечение если транскрибация пропущена и диаризация выключена
            if s == "extract" and self._skip_transcription and not self.config.enable_diarization:
                continue
            # Пропускаем транскрибацию если пропущена
            if s == "transcribe" and self._skip_transcription:
                continue
            # Пропускаем диаризацию если выключена
            if s == "diarize" and not self.config.enable_diarization:
                continue
            # Пропускаем анализ если выключен
            if s == "analyze" and not self.config.enable_ai_analysis:
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
            if s == "extract" and self._skip_transcription and not self.config.enable_diarization:
                continue
            if s == "transcribe" and self._skip_transcription:
                continue
            if s == "diarize" and not self.config.enable_diarization:
                continue
            if s == "analyze" and not self.config.enable_ai_analysis:
                continue
            active_weight += w

        # Нормализуем прогресс чтобы он заполнял 0.0-1.0
        scale = 1.0 / active_weight if active_weight > 0 else 1.0

        total_progress = (self._stage_offset + stage_progress * stage_weight) * scale
        total_progress = min(total_progress, 1.0)

        # ETA
        elapsed = time.monotonic() - self._start_time
        eta = 0.0
        if total_progress > 0.05:
            eta = elapsed / total_progress * (1.0 - total_progress)

        self.progress_callback(PipelineProgress(
            stage=self._current_stage,
            stage_progress=stage_progress,
            total_progress=total_progress,
            message=message,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        ))

    def _check_cancelled(self) -> None:
        """Проверить, не была ли операция отменена."""
        if self._cancel_event.is_set():
            raise CancelledException("Операция отменена пользователем")
