"""Извлечение сущностей и ключевых моментов из транскрипции."""

import json
import re
from typing import Any

from app.analysis.analyzer import Highlight, Entity, Entities
from app.core.transcriber import TranscriptionResult


def extract_highlights_and_entities(
    client: Any,
    model: str,
    transcript_text: str,
    transcription: TranscriptionResult,
) -> tuple[list[Highlight], Entities]:
    """
    Извлечь ключевые моменты и сущности из транскрипции.

    Args:
        client: Anthropic клиент
        model: Модель Claude
        transcript_text: Текст транскрипции с таймстампами
        transcription: Оригинальный результат транскрибации

    Returns:
        Tuple из списка Highlight и Entities
    """
    prompt = f"""Проанализируй эту транскрипцию и извлеки:
1. Ключевые моменты (highlights) — важные утверждения, инсайты, решения
2. Упомянутые сущности — компании, сервисы/продукты, люди

ТРАНСКРИПЦИЯ:
{transcript_text}

Верни JSON в формате:
{{
    "highlights": [
        {{
            "timestamp": "MM:SS",
            "text": "Краткое описание важного момента"
        }}
    ],
    "entities": {{
        "companies": [
            {{
                "name": "Название компании",
                "description": "Краткое описание (1 предложение)"
            }}
        ],
        "services": [
            {{
                "name": "Название сервиса/продукта",
                "description": "Что это за сервис (1 предложение)"
            }}
        ],
        "people": [
            {{
                "name": "Имя человека",
                "context": "Кто это / в каком контексте упомянут"
            }}
        ]
    }}
}}

Правила:
- Выбирай только действительно важные моменты (3-10 штук)
- Для сущностей указывай только явно упомянутые
- Таймстампы бери из транскрипции
- Отвечай ТОЛЬКО валидным JSON"""

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    return _parse_extraction_response(response.content[0].text, transcription)


def _parse_extraction_response(
    response_text: str,
    transcription: TranscriptionResult,
) -> tuple[list[Highlight], Entities]:
    """Парсить ответ модели."""
    highlights = []
    entities = Entities()

    try:
        # Убираем markdown code block если есть
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        data = json.loads(text.strip())

        # Парсим highlights
        for h in data.get("highlights", []):
            timestamp = h.get("timestamp", "00:00")
            start_seconds = _timestamp_to_seconds(timestamp)

            highlights.append(Highlight(
                timestamp=timestamp,
                text=h.get("text", ""),
                start_seconds=start_seconds,
            ))

        # Парсим entities
        entities_data = data.get("entities", {})

        for c in entities_data.get("companies", []):
            entities.companies.append(Entity(
                name=c.get("name", ""),
                description=c.get("description", ""),
            ))

        for s in entities_data.get("services", []):
            entities.services.append(Entity(
                name=s.get("name", ""),
                description=s.get("description", ""),
            ))

        for p in entities_data.get("people", []):
            entities.people.append(Entity(
                name=p.get("name", ""),
                description="",
                context=p.get("context", ""),
            ))

    except json.JSONDecodeError:
        # Если не удалось распарсить — возвращаем пустые результаты
        pass

    return highlights, entities


def _timestamp_to_seconds(timestamp: str) -> float:
    """Конвертировать таймстамп MM:SS или HH:MM:SS в секунды."""
    parts = timestamp.split(":")

    try:
        if len(parts) == 2:
            # MM:SS
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        pass

    return 0.0
