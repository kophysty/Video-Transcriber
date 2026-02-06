"""Экспорт в простой текстовый формат."""

from pathlib import Path

from app.core.transcriber import TranscriptionResult
from app.utils.formats import seconds_to_txt_timestamp


def export_txt(
    transcription: TranscriptionResult,
    output_path: Path,
    include_timestamps: bool = True,
    include_speaker: bool = True,
) -> None:
    """
    Экспортировать результаты в TXT формат.

    Формат:
    ```
    [00:00:00] SPEAKER_00: Текст первого сегмента.
    [00:00:05] SPEAKER_01: Текст второго сегмента.
    ```

    Или без спикеров:
    ```
    [00:00:00] Текст первого сегмента.
    [00:00:05] Текст второго сегмента.
    ```

    Args:
        transcription: Результат транскрибации
        output_path: Путь к выходному файлу
        include_timestamps: Включать таймстампы
        include_speaker: Включать метки спикеров
    """
    lines = []

    for segment in transcription.segments:
        parts = []

        # Таймстамп
        if include_timestamps:
            parts.append(seconds_to_txt_timestamp(segment.start))

        # Спикер
        if include_speaker and segment.speaker:
            parts.append(f"{segment.speaker}:")

        # Текст
        parts.append(segment.text)

        lines.append(" ".join(parts))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_plain_text(
    transcription: TranscriptionResult,
    output_path: Path,
) -> None:
    """
    Экспортировать чистый текст без таймстампов и спикеров.

    Полезно для дальнейшей обработки текста.
    """
    texts = [segment.text for segment in transcription.segments]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))
