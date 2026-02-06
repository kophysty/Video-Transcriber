"""Объединение результатов транскрибации и диаризации."""

from typing import Optional

from app.core.transcriber import TranscriptionSegment, TranscriptionResult
from app.core.diarizer import DiarizationResult, DiarizationSegment


def merge_transcription_with_diarization(
    transcription: TranscriptionResult,
    diarization: Optional[DiarizationResult],
) -> TranscriptionResult:
    """
    Объединить транскрибацию с диаризацией.

    Алгоритм:
    1. Для каждого сегмента транскрибации находим пересекающиеся
       сегменты диаризации
    2. Назначаем спикера с максимальным пересечением

    Args:
        transcription: Результат транскрибации
        diarization: Результат диаризации (может быть None)

    Returns:
        TranscriptionResult с заполненными полями speaker
    """
    if diarization is None or not diarization.segments:
        return transcription

    # Для каждого сегмента транскрибации назначаем спикера
    for segment in transcription.segments:
        speaker = _find_best_speaker(segment, diarization.segments)
        segment.speaker = speaker

    return transcription


def _find_best_speaker(
    segment: TranscriptionSegment,
    diarization_segments: list[DiarizationSegment],
) -> Optional[str]:
    """
    Найти лучшего спикера для сегмента.

    Выбираем спикера с максимальным пересечением по времени.
    """
    best_speaker = None
    best_overlap = 0.0

    for diar_seg in diarization_segments:
        overlap = _calculate_overlap(
            segment.start, segment.end,
            diar_seg.start, diar_seg.end,
        )

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diar_seg.speaker

    return best_speaker


def _calculate_overlap(
    start1: float, end1: float,
    start2: float, end2: float,
) -> float:
    """
    Вычислить пересечение двух интервалов.

    Returns:
        Длительность пересечения в секундах (0 если не пересекаются)
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start < overlap_end:
        return overlap_end - overlap_start

    return 0.0


def assign_speakers_to_words(
    transcription: TranscriptionResult,
    diarization: Optional[DiarizationResult],
) -> None:
    """
    Назначить спикеров на уровне слов (более точная диаризация).

    Это полезно когда один сегмент транскрибации содержит речь
    нескольких спикеров.

    Модифицирует transcription in-place.
    """
    if diarization is None or not diarization.segments:
        return

    for segment in transcription.segments:
        if not segment.words:
            continue

        # Группируем слова по спикерам
        word_speakers = []
        for word in segment.words:
            speaker = _find_speaker_at_time(
                (word.start + word.end) / 2,  # Середина слова
                diarization.segments,
            )
            word_speakers.append(speaker)

        # Если все слова от одного спикера — назначаем сегменту
        unique_speakers = set(s for s in word_speakers if s)
        if len(unique_speakers) == 1:
            segment.speaker = unique_speakers.pop()
        elif len(unique_speakers) > 1:
            # Берём самого частого спикера
            from collections import Counter
            speaker_counts = Counter(s for s in word_speakers if s)
            if speaker_counts:
                segment.speaker = speaker_counts.most_common(1)[0][0]


def _find_speaker_at_time(
    time: float,
    diarization_segments: list[DiarizationSegment],
) -> Optional[str]:
    """Найти спикера в заданный момент времени."""
    for seg in diarization_segments:
        if seg.start <= time <= seg.end:
            return seg.speaker
    return None
