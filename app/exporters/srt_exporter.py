"""Экспорт в SRT формат субтитров."""

from pathlib import Path

from app.core.transcriber import TranscriptionResult
from app.utils.formats import seconds_to_srt_timestamp


def export_srt(
    transcription: TranscriptionResult,
    output_path: Path,
    include_speaker: bool = True,
) -> None:
    """
    Экспортировать результаты в SRT формат.

    Формат SRT:
    ```
    1
    00:00:00,000 --> 00:00:05,000
    [SPEAKER_00] Текст субтитра

    2
    00:00:05,500 --> 00:00:10,000
    [SPEAKER_01] Следующий субтитр
    ```

    Args:
        transcription: Результат транскрибации
        output_path: Путь к выходному файлу
        include_speaker: Включать метки спикеров
    """
    lines = []

    for i, segment in enumerate(transcription.segments, start=1):
        # Номер субтитра
        lines.append(str(i))

        # Таймстампы
        start_ts = seconds_to_srt_timestamp(segment.start)
        end_ts = seconds_to_srt_timestamp(segment.end)
        lines.append(f"{start_ts} --> {end_ts}")

        # Текст
        text = segment.text
        if include_speaker and segment.speaker:
            text = f"[{segment.speaker}] {text}"
        lines.append(text)

        # Пустая строка между субтитрами
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
