"""Экспорт в JSON формат — source of truth."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.transcriber import TranscriptionResult
from app.core.diarizer import DiarizationResult
from app import __version__


def export_json(
    transcription: TranscriptionResult,
    diarization: Optional[DiarizationResult],
    output_path: Path,
    source_file: str = "",
    model: str = "",
) -> None:
    """
    Экспортировать результаты в JSON.

    Args:
        transcription: Результат транскрибации
        diarization: Результат диаризации (может быть None)
        output_path: Путь к выходному файлу
        source_file: Имя исходного файла
        model: Имя использованной модели
    """
    data = {
        "metadata": {
            "source_file": source_file,
            "duration_seconds": transcription.info.duration,
            "language": transcription.info.language,
            "language_probability": round(transcription.info.language_probability, 4),
            "model": model,
            "num_speakers": diarization.num_speakers if diarization else None,
            "created_at": datetime.now().isoformat(),
            "app_version": __version__,
        },
        "segments": [],
    }

    for segment in transcription.segments:
        segment_data = {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text,
        }

        if segment.speaker:
            segment_data["speaker"] = segment.speaker

        if segment.words:
            segment_data["words"] = [
                {
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "text": word.text,
                    "probability": round(word.probability, 4),
                }
                for word in segment.words
            ]

        data["segments"].append(segment_data)

    # Записываем с отступами для читаемости
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
