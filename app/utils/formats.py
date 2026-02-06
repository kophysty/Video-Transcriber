"""Хелперы для форматирования таймстампов."""


def seconds_to_srt_timestamp(seconds: float) -> str:
    """
    Конвертировать секунды в SRT формат таймстампа.

    Формат: HH:MM:SS,mmm (запятая как разделитель миллисекунд)

    Args:
        seconds: Время в секундах

    Returns:
        Строка в формате "00:00:00,000"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """
    Конвертировать секунды в VTT формат таймстампа.

    Формат: HH:MM:SS.mmm (точка как разделитель миллисекунд)

    Args:
        seconds: Время в секундах

    Returns:
        Строка в формате "00:00:00.000"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def seconds_to_txt_timestamp(seconds: float) -> str:
    """
    Конвертировать секунды в простой формат для TXT.

    Формат: [HH:MM:SS] или [MM:SS] если < 1 часа

    Args:
        seconds: Время в секундах

    Returns:
        Строка в формате "[00:00:00]" или "[00:00]"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def format_duration(seconds: float) -> str:
    """
    Форматировать длительность для отображения.

    Args:
        seconds: Длительность в секундах

    Returns:
        Человекочитаемая строка типа "1 ч 23 мин" или "5 мин 30 сек"
    """
    if seconds < 60:
        return f"{int(seconds)} сек"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours} ч")
    if minutes > 0:
        parts.append(f"{minutes} мин")
    if secs > 0 and hours == 0:
        parts.append(f"{secs} сек")

    return " ".join(parts)


def format_file_size(size_bytes: int) -> str:
    """
    Форматировать размер файла.

    Args:
        size_bytes: Размер в байтах

    Returns:
        Человекочитаемая строка типа "1.5 GB" или "256 MB"
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
