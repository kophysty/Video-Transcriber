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
    Экспортировать транскрипцию в читаемый Markdown.

    Если есть спикеры (диаризация) — группирует по спикерам,
    таймкод ставится только при смене спикера.

    Если спикеров нет — сплошной текст,
    таймкод ставится раз в timestamp_interval секунд.
    """
    has_speakers = any(s.speaker for s in transcription.segments)

    if has_speakers:
        content = _export_with_speakers(transcription)
    else:
        content = _export_plain(transcription, timestamp_interval)

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


def _export_with_speakers(transcription: TranscriptionResult) -> str:
    """Экспорт с диаризацией: группировка по спикерам."""
    lines = []
    lines.append("# Транскрипция\n")

    # Мета
    if transcription.info:
        lang = transcription.info.language or "?"
        dur = transcription.info.duration
        dur_min = int(dur // 60)
        dur_sec = int(dur % 60)
        lines.append(f"*Язык: {lang} · Длительность: {dur_min} мин {dur_sec} сек*\n")

    # Собираем уникальных спикеров
    speakers = sorted(set(s.speaker for s in transcription.segments if s.speaker))
    if speakers:
        lines.append(f"*Спикеры: {len(speakers)}*\n")
    lines.append("---\n")

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
            lines.append(f"**{speaker}** {ts}\n")
            current_speaker = speaker

        speaker_texts.append(segment.text)

    # Последний блок
    if speaker_texts:
        lines.append(" ".join(speaker_texts))
        lines.append("")

    return "\n".join(lines)
