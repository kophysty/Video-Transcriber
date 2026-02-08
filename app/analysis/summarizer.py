"""Создание иерархического саммари транскрипции."""

import json
from typing import Callable

from app.analysis.analyzer import Summary


# Максимум токенов в одном чанке (примерно 10K слов)
MAX_CHUNK_TOKENS = 40000
# Примерное соотношение символов к токенам для русского текста
CHARS_PER_TOKEN = 3


def create_summary(call_fn: Callable[[str], str], transcript_text: str) -> Summary:
    """
    Создать иерархическое саммари транскрипции.

    Args:
        call_fn: Функция вызова LLM (prompt -> response text)
        transcript_text: Текст транскрипции с таймстампами

    Returns:
        Summary с one_liner, paragraph и detailed
    """
    # Проверяем длину текста
    estimated_tokens = len(transcript_text) // CHARS_PER_TOKEN

    if estimated_tokens > MAX_CHUNK_TOKENS:
        # Длинный транскрипт — используем chunking
        return _create_summary_chunked(call_fn, transcript_text)
    else:
        # Короткий — анализируем целиком
        return _create_summary_direct(call_fn, transcript_text)


def _create_summary_direct(call_fn: Callable[[str], str], transcript_text: str) -> Summary:
    """Создать саммари для короткого транскрипта."""
    prompt = f"""Проанализируй эту транскрипцию и создай структурированное саммари.

ТРАНСКРИПЦИЯ:
{transcript_text}

Верни JSON в формате:
{{
    "one_liner": "Краткое описание в одну строку (до 100 символов)",
    "paragraph": "Абзац с основными моментами (3-5 предложений)",
    "detailed": "Детальное саммари с разбивкой по темам/разделам (можно использовать markdown)"
}}

Отвечай ТОЛЬКО валидным JSON без дополнительного текста."""

    response = call_fn(prompt)
    return _parse_summary_response(response)


def _create_summary_chunked(call_fn: Callable[[str], str], transcript_text: str) -> Summary:
    """Создать саммари для длинного транскрипта через chunking."""
    # Разбиваем на чанки
    chunks = _split_into_chunks(transcript_text)

    # Создаём саммари для каждого чанка
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Проанализируй этот фрагмент транскрипции (часть {i+1} из {len(chunks)}).

ФРАГМЕНТ:
{chunk}

Кратко опиши основные темы и ключевые моменты этого фрагмента (2-3 абзаца).
Сохраняй упоминания таймстампов для важных моментов."""

        response = call_fn(prompt)
        chunk_summaries.append(response)

    # Синтезируем финальное саммари
    combined = "\n\n---\n\n".join([
        f"ЧАСТЬ {i+1}:\n{summary}"
        for i, summary in enumerate(chunk_summaries)
    ])

    synthesis_prompt = f"""На основе этих промежуточных саммари частей транскрипции создай единое структурированное саммари.

ПРОМЕЖУТОЧНЫЕ САММАРИ:
{combined}

Верни JSON в формате:
{{
    "one_liner": "Краткое описание в одну строку (до 100 символов)",
    "paragraph": "Абзац с основными моментами (3-5 предложений)",
    "detailed": "Детальное саммари с разбивкой по темам/разделам (можно использовать markdown)"
}}

Объедини информацию логично, избегай повторений. Отвечай ТОЛЬКО валидным JSON."""

    response = call_fn(synthesis_prompt)
    return _parse_summary_response(response)


def _split_into_chunks(text: str) -> list[str]:
    """Разбить текст на чанки по границам предложений."""
    # Целевой размер чанка в символах
    target_chunk_size = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN

    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line)

        if current_size + line_size > target_chunk_size and current_chunk:
            # Сохраняем текущий чанк
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size

    # Добавляем последний чанк
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def _parse_summary_response(response_text: str) -> Summary:
    """Парсить ответ модели в Summary."""
    try:
        # Пытаемся найти JSON в ответе
        text = response_text.strip()

        # Убираем markdown code block если есть
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        data = json.loads(text.strip())

        return Summary(
            one_liner=data.get("one_liner", ""),
            paragraph=data.get("paragraph", ""),
            detailed=data.get("detailed", ""),
        )
    except json.JSONDecodeError:
        # Если не удалось распарсить JSON, используем текст как есть
        return Summary(
            one_liner="",
            paragraph=response_text[:500],
            detailed=response_text,
        )
