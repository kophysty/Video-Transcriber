"""Экспорт в WebVTT формат субтитров."""

from pathlib import Path

from app.core.transcriber import TranscriptionResult
from app.utils.formats import seconds_to_vtt_timestamp


def export_vtt(
    transcription: TranscriptionResult,
    output_path: Path,
    include_speaker: bool = True,
) -> None:
    """
    Экспортировать результаты в VTT формат.

    Формат VTT:
    ```
    WEBVTT

    00:00:00.000 --> 00:00:05.000
    <v SPEAKER_00>Текст субтитра

    00:00:05.500 --> 00:00:10.000
    <v SPEAKER_01>Следующий субтитр
    ```

    Args:
        transcription: Результат транскрибации
        output_path: Путь к выходному файлу
        include_speaker: Включать метки спикеров
    """
    lines = ["WEBVTT", ""]

    for segment in transcription.segments:
        # Таймстампы
        start_ts = seconds_to_vtt_timestamp(segment.start)
        end_ts = seconds_to_vtt_timestamp(segment.end)
        lines.append(f"{start_ts} --> {end_ts}")

        # Текст с опциональной меткой спикера
        if include_speaker and segment.speaker:
            # VTT использует <v Speaker> тег для голосов
            lines.append(f"<v {segment.speaker}>{segment.text}")
        else:
            lines.append(segment.text)

        # Пустая строка между субтитрами
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
