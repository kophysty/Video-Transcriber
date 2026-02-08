"""Загрузка TranscriptionResult из ранее экспортированного JSON."""

import json
from pathlib import Path

from app.core.transcriber import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    TranscriptionInfo,
)


def load_transcription_from_json(json_path: Path) -> TranscriptionResult:
    """
    Загрузить результат транскрибации из JSON файла.

    Args:
        json_path: Путь к JSON файлу, ранее экспортированному export_json

    Returns:
        TranscriptionResult

    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если JSON не содержит нужных данных
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "segments" not in data:
        raise ValueError(f"JSON не содержит 'segments': {json_path}")

    metadata = data.get("metadata", {})

    info = TranscriptionInfo(
        language=metadata.get("language", "?"),
        language_probability=metadata.get("language_probability", 0.0),
        duration=metadata.get("duration_seconds", 0.0),
    )

    segments = []
    for seg_data in data["segments"]:
        words = []
        for w in seg_data.get("words", []):
            words.append(TranscriptionWord(
                start=w["start"],
                end=w["end"],
                text=w["text"],
                probability=w.get("probability", 0.0),
            ))

        segments.append(TranscriptionSegment(
            start=seg_data["start"],
            end=seg_data["end"],
            text=seg_data["text"],
            words=words,
            speaker=seg_data.get("speaker"),
        ))

    return TranscriptionResult(segments=segments, info=info)
