"""Экспорт в читаемый Markdown формат."""

from pathlib import Path

from app.core.transcriber import TranscriptionResult
from app.utils.formats import seconds_to_txt_timestamp


# Интервал для таймкодов в секундах (без диаризации)
TIMESTAMP_INTERVAL = 60


def export_md(
    transcription: TranscriptionResult,
    output_path: Path,
    timestamp_interval: int = TIMESTAMP_INTERVAL,
) -> None:
    """
    Экспортировать транскрипцию в читаемый Markdown (сплошной текст).

    Всегда экспортирует plain-версию без разбивки по спикерам.
    Для версии с диаризацией используйте export_diarized_md().
    """
    content = _export_plain(transcription, timestamp_interval)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def export_diarized_md(
    transcription: TranscriptionResult,
    output_path: Path,
    speaker_map: dict[str, str] | None = None,
) -> None:
    """
    Экспортировать транскрипцию с диаризацией в Markdown.

    Args:
        transcription: Результат транскрипции с метками спикеров
        output_path: Путь для сохранения файла
        speaker_map: Маппинг меток на имена {"SPEAKER_00": "Иван"}
    """
    content = _export_with_speakers(transcription, speaker_map)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def _export_plain(
    transcription: TranscriptionResult,
    interval: int,
) -> str:
    """Экспорт без диаризации: сплошной текст с редкими таймкодами."""
    lines = []
    lines.append("# Транскрипция\n")

    # Мета
    if transcription.info:
        lang = transcription.info.language or "?"
        dur = transcription.info.duration
        dur_min = int(dur // 60)
        dur_sec = int(dur % 60)
        lines.append(f"*Язык: {lang} · Длительность: {dur_min} мин {dur_sec} сек*\n")
        lines.append("---\n")

    next_timestamp_at = 0.0
    paragraph_texts = []

    for segment in transcription.segments:
        # Пора вставить таймкод?
        if segment.start >= next_timestamp_at:
            # Сначала сбросим накопленный текст
            if paragraph_texts:
                lines.append(" ".join(paragraph_texts))
                lines.append("")
                paragraph_texts = []

            ts = seconds_to_txt_timestamp(segment.start)
            lines.append(f"**{ts}**\n")
            next_timestamp_at = segment.start + interval

        paragraph_texts.append(segment.text)

    # Оставшийся текст
    if paragraph_texts:
        lines.append(" ".join(paragraph_texts))
        lines.append("")

    return "\n".join(lines)


def _export_with_speakers(
    transcription: TranscriptionResult,
    speaker_map: dict[str, str] | None = None,
) -> str:
    """Экспорт с диаризацией: группировка по спикерам."""
    if speaker_map is None:
        speaker_map = {}

    lines = []
    lines.append("# Транскрипция по ролям\n")

    # Мета
    if transcription.info:
        lang = transcription.info.language or "?"
        dur = transcription.info.duration
        dur_min = int(dur // 60)
        dur_sec = int(dur % 60)
        lines.append(f"*Язык: {lang} · Длительность: {dur_min} мин {dur_sec} сек*\n")

    # Собираем уникальных спикеров и формируем имена
    raw_speakers = sorted(set(s.speaker for s in transcription.segments if s.speaker))
    if raw_speakers:
        speaker_names = []
        for i, spk in enumerate(raw_speakers):
            name = speaker_map.get(spk, f"Спикер {i + 1}")
            speaker_names.append(f"{name} ({spk})" if name != spk else spk)
        lines.append(f"*Спикеры ({len(raw_speakers)}): {', '.join(speaker_names)}*\n")
    lines.append("---\n")

    # Создаём полный маппинг с фоллбэками
    display_names: dict[str, str] = {}
    for i, spk in enumerate(raw_speakers):
        display_names[spk] = speaker_map.get(spk, f"Спикер {i + 1}")

    current_speaker = None
    speaker_texts = []

    for segment in transcription.segments:
        speaker = segment.speaker or "?"

        if speaker != current_speaker:
            # Сбрасываем предыдущий блок спикера
            if current_speaker is not None and speaker_texts:
                lines.append(" ".join(speaker_texts))
                lines.append("")
            speaker_texts = []

            # Новый блок спикера с таймкодом
            ts = seconds_to_txt_timestamp(segment.start)
            name = display_names.get(speaker, speaker)
            lines.append(f"**{name}** {ts}\n")
            current_speaker = speaker

        speaker_texts.append(segment.text)

    # Последний блок
    if speaker_texts:
        lines.append(" ".join(speaker_texts))
        lines.append("")

    return "\n".join(lines)
