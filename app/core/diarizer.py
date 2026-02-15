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

# Лимит окон: для длинных файлов берём не все окна, а с шагом
MAX_EMBEDDING_WINDOWS = 500   # Максимум окон для кластеризации


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
    """Диаризатор на основе SpeechBrain ECAPA-TDNN + Silero VAD.

    Всегда работает на CPU — модель маленькая (~25 MB), а CUDA
    вызывает hard crash на длинных файлах из-за нестабильности драйвера
    после интенсивной работы Whisper.
    """

    def __init__(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.progress_callback = progress_callback
        self._vad_model = None
        self._vad_utils = None
        self._speaker_model = None

    def load_pipeline(self) -> None:
        """Загрузить модели диаризации."""
        import os
        import torch

        # Подавить предупреждение о symlinks на Windows
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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

        # Совместимость: SpeechBrain 1.0.3 + huggingface_hub >=1.0:
        # 1) use_auth_token убран (теперь token)
        # 2) 404 выбрасывает EntryNotFoundError вместо HTTPError,
        #    а SpeechBrain ловит только HTTPError → конвертируем обратно
        import huggingface_hub as _hf_hub
        from requests.exceptions import HTTPError
        _original_hf_download = _hf_hub.hf_hub_download
        def _patched_hf_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            try:
                return _original_hf_download(*args, **kwargs)
            except _hf_hub.errors.EntryNotFoundError as e:
                raise HTTPError(f"404 Client Error: {e}") from e
        _hf_hub.hf_hub_download = _patched_hf_download

        from speechbrain.inference.speaker import SpeakerRecognition
        from speechbrain.utils.fetching import LocalStrategy

        models_dir = AppConfig.get_models_dir() / "speechbrain" / "spkrec-ecapa-voxceleb"

        self._speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(models_dir),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,  # Windows: symlinks требуют admin
        )
        log.info("SpeechBrain ECAPA-TDNN загружен (CPU)")

        self._report_progress(0.1, "Модели диаризации загружены")

    def unload_pipeline(self) -> None:
        """Выгрузить модели."""
        self._vad_model = None
        self._vad_utils = None
        self._speaker_model = None

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
        from scipy.io import wavfile as scipy_wav
        from scipy.signal import resample_poly
        from math import gcd

        if self._vad_model is None:
            self.load_pipeline()

        self._report_progress(0.1, "Загрузка аудио...")
        log.info("Запуск диаризации: %s", audio_path.name)

        # 1. Загрузка аудио (scipy вместо torchaudio — не требует бэкендов)
        sample_rate, audio_data = scipy_wav.read(str(audio_path))

        # Конвертация в float32 [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Моно
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Ресемплинг в 16kHz
        if sample_rate != 16000:
            log.info("Ресемплинг %d -> 16000 Hz", sample_rate)
            g = gcd(sample_rate, 16000)
            audio_data = resample_poly(audio_data, 16000 // g, sample_rate // g)
            sample_rate = 16000

        # numpy -> torch tensor [1, samples]
        waveform = torch.from_numpy(audio_data).unsqueeze(0)

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

        chunk_samples = int(CHUNK_SECONDS * sample_rate)
        hop_samples = int(CHUNK_HOP_SECONDS * sample_rate)
        min_samples = int(MIN_CHUNK_SECONDS * sample_rate)

        # 1. Сначала собираем ВСЕ окна (без GPU-вычислений)
        all_windows = []
        for ts in speech_timestamps:
            start_sample = ts['start']
            end_sample = ts['end']
            duration_samples = end_sample - start_sample

            if duration_samples < min_samples:
                continue

            if duration_samples <= chunk_samples:
                all_windows.append((start_sample, end_sample))
            else:
                pos = start_sample
                while pos + min_samples <= end_sample:
                    window_end = min(pos + chunk_samples, end_sample)
                    if (window_end - pos) < min_samples:
                        break
                    all_windows.append((pos, window_end))
                    pos += hop_samples

        log.info("Всего окон для embeddings: %d", len(all_windows))

        # 2. Если окон слишком много — прореживаем равномерно
        if len(all_windows) > MAX_EMBEDDING_WINDOWS:
            step = len(all_windows) / MAX_EMBEDDING_WINDOWS
            indices = [int(i * step) for i in range(MAX_EMBEDDING_WINDOWS)]
            all_windows = [all_windows[i] for i in indices]
            log.info("Прореживание до %d окон (длинный файл)", len(all_windows))

        # 3. Извлекаем embeddings (CPU)
        windows = []
        embeddings = []
        total = len(all_windows)

        for i, (start_sample, end_sample) in enumerate(all_windows):
            segment_audio = waveform[:, start_sample:end_sample]
            with torch.no_grad():
                emb = self._speaker_model.encode_batch(segment_audio)
            embeddings.append(emb.squeeze().numpy())
            windows.append({
                'start': start_sample / sample_rate,
                'end': end_sample / sample_rate,
            })

            # Прогресс
            if i % 20 == 0:
                progress = 0.25 + 0.50 * (i / total)
                self._report_progress(progress, f"Embeddings: {i + 1}/{total}...")

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
        progress_callback=progress_callback,
    )
