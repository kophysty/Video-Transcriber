"""Диаризация спикеров.

Использует:
- Silero VAD для определения речевых сегментов
- SpeechBrain ECAPA-TDNN для speaker embeddings
- Agglomerative clustering для группировки спикеров

Все модели свободно доступны, HuggingFace токен НЕ нужен.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from app.utils.config import AppConfig

log = logging.getLogger(__name__)

# Параметры сегментации
CHUNK_SECONDS = 2.0       # Размер окна для извлечения embeddings
CHUNK_HOP_SECONDS = 1.0   # Шаг между окнами
MIN_CHUNK_SECONDS = 0.5   # Минимальная длина сегмента


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
    """Диаризатор на основе SpeechBrain ECAPA-TDNN + Silero VAD."""

    def __init__(
        self,
        device: str = "cuda",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Args:
            device: "cuda" или "cpu"
            progress_callback: Callback(progress: 0.0-1.0, message: str)
        """
        self.device = device
        self.progress_callback = progress_callback
        self._vad_model = None
        self._vad_utils = None
        self._speaker_model = None

    def load_pipeline(self) -> None:
        """Загрузить модели диаризации."""
        import torch

        self._report_progress(0.0, "Загрузка модели VAD...")

        # 1. Silero VAD (torch.hub, скачивается с GitHub, без токенов)
        log.info("Загрузка Silero VAD...")
        self._vad_model, self._vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        log.info("Silero VAD загружен")

        # 2. SpeechBrain ECAPA-TDNN (Apache 2.0, не gated, без токенов)
        self._report_progress(0.03, "Загрузка модели speaker embeddings...")
        log.info("Загрузка SpeechBrain ECAPA-TDNN...")

        # Совместимость: SpeechBrain может не поддерживать новый torchaudio
        import torchaudio as _ta
        if not hasattr(_ta, 'list_audio_backends'):
            _ta.list_audio_backends = lambda: []

        from speechbrain.inference.speaker import SpeakerRecognition

        models_dir = AppConfig.get_models_dir() / "speechbrain" / "spkrec-ecapa-voxceleb"

        run_opts = {}
        if self.device == "cuda" and torch.cuda.is_available():
            run_opts["device"] = "cuda"

        self._speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(models_dir),
            run_opts=run_opts,
        )
        log.info("SpeechBrain ECAPA-TDNN загружен, device=%s", self.device)

        self._report_progress(0.1, "Модели диаризации загружены")

    def unload_pipeline(self) -> None:
        """Выгрузить модели для освобождения VRAM."""
        self._vad_model = None
        self._vad_utils = None
        self._speaker_model = None

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
        import torch
        import torchaudio

        if self._vad_model is None:
            self.load_pipeline()

        self._report_progress(0.1, "Загрузка аудио...")
        log.info("Запуск диаризации: %s", audio_path.name)

        # 1. Загрузка аудио
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Ресемплинг в 16kHz
        if sample_rate != 16000:
            log.info("Ресемплинг %d -> 16000 Hz", sample_rate)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Моно
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 2. VAD — определение речевых сегментов
        self._report_progress(0.15, "Определение речевых сегментов (VAD)...")
        log.info("Запуск Silero VAD...")

        (get_speech_timestamps, _, _, _, _) = self._vad_utils

        speech_timestamps = get_speech_timestamps(
            waveform.squeeze(0),
            self._vad_model,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=500,
            min_silence_duration_ms=300,
        )

        log.info("VAD: найдено %d речевых сегментов", len(speech_timestamps))

        if not speech_timestamps:
            log.warning("VAD не нашёл речевых сегментов")
            return DiarizationResult(segments=[], num_speakers=0)

        # 3. Нарезаем речевые сегменты на окна и извлекаем embeddings
        self._report_progress(0.25, "Извлечение speaker embeddings...")

        windows, embeddings = self._extract_embeddings(
            waveform, sample_rate, speech_timestamps
        )

        log.info("Извлечено %d embeddings", len(embeddings))

        if len(embeddings) == 0:
            return DiarizationResult(segments=[], num_speakers=0)

        if len(embeddings) == 1:
            segments = [
                DiarizationSegment(w['start'], w['end'], 'SPEAKER_00')
                for w in windows
            ]
            return DiarizationResult(segments=segments, num_speakers=1)

        # 4. Кластеризация
        self._report_progress(0.75, "Кластеризация спикеров...")
        labels = self._cluster_embeddings(
            embeddings,
            num_speakers=num_speakers,
            min_speakers=min_speakers or 1,
            max_speakers=max_speakers or 10,
        )

        # 5. Формирование результата
        raw_segments = []
        for window, label in zip(windows, labels):
            speaker = f"SPEAKER_{int(label):02d}"
            raw_segments.append(DiarizationSegment(
                start=window['start'],
                end=window['end'],
                speaker=speaker,
            ))

        # Объединяем соседние сегменты с одним спикером
        merged = self._merge_segments(raw_segments)

        speakers = set(s.speaker for s in merged)
        log.info("Диаризация завершена: %d сегментов, %d спикеров",
                 len(merged), len(speakers))

        self._report_progress(1.0, "Диаризация завершена")

        return DiarizationResult(
            segments=merged,
            num_speakers=len(speakers),
        )

    def _extract_embeddings(
        self,
        waveform,
        sample_rate: int,
        speech_timestamps: list[dict],
    ) -> tuple[list[dict], list[np.ndarray]]:
        """Извлечь speaker embeddings из речевых сегментов."""
        import torch

        windows = []
        embeddings = []

        chunk_samples = int(CHUNK_SECONDS * sample_rate)
        hop_samples = int(CHUNK_HOP_SECONDS * sample_rate)
        min_samples = int(MIN_CHUNK_SECONDS * sample_rate)

        total_ts = len(speech_timestamps)

        for i, ts in enumerate(speech_timestamps):
            start_sample = ts['start']
            end_sample = ts['end']
            duration_samples = end_sample - start_sample

            if duration_samples < min_samples:
                continue

            if duration_samples <= chunk_samples:
                # Короткий сегмент — одно окно
                segment_audio = waveform[:, start_sample:end_sample]
                with torch.no_grad():
                    emb = self._speaker_model.encode_batch(segment_audio)
                embeddings.append(emb.squeeze().cpu().numpy())
                windows.append({
                    'start': start_sample / sample_rate,
                    'end': end_sample / sample_rate,
                })
            else:
                # Длинный сегмент — разбиваем на окна
                pos = start_sample
                while pos + min_samples <= end_sample:
                    window_end = min(pos + chunk_samples, end_sample)
                    if (window_end - pos) < min_samples:
                        break

                    segment_audio = waveform[:, pos:window_end]
                    with torch.no_grad():
                        emb = self._speaker_model.encode_batch(segment_audio)
                    embeddings.append(emb.squeeze().cpu().numpy())
                    windows.append({
                        'start': pos / sample_rate,
                        'end': window_end / sample_rate,
                    })
                    pos += hop_samples

            # Прогресс по VAD сегментам
            if i % 10 == 0:
                progress = 0.25 + 0.50 * (i / total_ts)
                self._report_progress(progress, f"Embeddings: {len(embeddings)} окон...")

        return windows, embeddings

    @staticmethod
    def _cluster_embeddings(
        embeddings: list[np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
    ) -> list[int]:
        """Кластеризовать embeddings для определения спикеров."""
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import fcluster, linkage

        emb_array = np.array(embeddings)
        distances = pdist(emb_array, metric='cosine')
        Z = linkage(distances, method='average')

        if num_speakers:
            labels = fcluster(Z, t=num_speakers, criterion='maxclust')
        else:
            # Автоопределение через порог расстояния
            labels = fcluster(Z, t=0.5, criterion='distance')
            n_clusters = len(set(labels))

            if n_clusters < min_speakers:
                labels = fcluster(Z, t=min_speakers, criterion='maxclust')
            elif n_clusters > max_speakers:
                labels = fcluster(Z, t=max_speakers, criterion='maxclust')

        # Нормализуем метки: 0, 1, 2, ...
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return [label_map[l] for l in labels]

    @staticmethod
    def _merge_segments(
        segments: list[DiarizationSegment],
        gap_threshold: float = 0.3,
    ) -> list[DiarizationSegment]:
        """Объединить соседние сегменты с одним спикером."""
        if not segments:
            return []

        merged = [DiarizationSegment(
            start=segments[0].start,
            end=segments[0].end,
            speaker=segments[0].speaker,
        )]

        for seg in segments[1:]:
            prev = merged[-1]
            if seg.speaker == prev.speaker and (seg.start - prev.end) < gap_threshold:
                merged[-1] = DiarizationSegment(
                    start=prev.start,
                    end=seg.end,
                    speaker=prev.speaker,
                )
            else:
                merged.append(DiarizationSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                ))

        return merged

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
        device=config.device,
        progress_callback=progress_callback,
    )
