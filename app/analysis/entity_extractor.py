"""Извлечение сущностей и ключевых моментов из транскрипции."""

import json
from typing import Callable

from app.analysis.analyzer import Highlight, Entity, Entities
from app.core.transcriber import TranscriptionResult


def extract_highlights_and_entities(
    call_fn: Callable[[str], str],
    transcript_text: str,
    transcription: TranscriptionResult,
) -> tuple[list[Highlight], Entities]:
    """
    Извлечь ключевые моменты и сущности из транскрипции.

    Args:
        call_fn: Функция вызова LLM (prompt -> response text)
        transcript_text: Текст транскрипции с таймстампами
        transcription: Оригинальный результат транскрибации

    Returns:
        Tuple из списка Highlight и Entities
    """
    prompt = f"""Проанализируй эту транскрипцию и извлеки:
1. Ключевые моменты (highlights) — важные утверждения, инсайты, решения, интересные факты
2. Упомянутые сущности — компании, сервисы/продукты, люди

ТРАНСКРИПЦИЯ:
{transcript_text}

Верни JSON в формате:
{{
    "highlights": [
        {{
            "timestamp": "MM:SS",
            "text": "Развёрнутое описание важного момента (1-2 предложения с конкретикой: цифры, имена, суть)"
        }}
    ],
    "entities": {{
        "companies": [
            {{
                "name": "Название компании",
                "description": "Краткое описание компании (1 предложение)",
                "context": "Подробный контекст: зачем и как именно упоминалась в разговоре (1-2 предложения)"
            }}
        ],
        "services": [
            {{
                "name": "Название сервиса/продукта",
                "description": "Что это за сервис (1 предложение)",
                "context": "Подробный контекст: как именно использовался или упоминался в разговоре (1-2 предложения)"
            }}
        ],
        "people": [
            {{
                "name": "Имя человека (SPEAKER_XX если известен)",
                "context": "Кто это, роль в обсуждении и что именно говорил/делал (1-2 предложения)"
            }}
        ]
    }}
}}

Правила:
- Highlights: выбирай ВСЕ значимые моменты — инсайты, цифры, решения, кейсы (8-20 штук для длинных записей, минимум 5)
- Описание highlight должно быть информативным: не просто "обсудили тему", а конкретика — что именно сказали, какие цифры привели
- Для сущностей указывай ВСЕ явно упомянутые (не пропускай даже кратко упомянутые)
- Поле "context" ОБЯЗАТЕЛЬНО и ПОДРОБНО — объясни, зачем/почему сущность упоминалась в конкретном контексте разговора
- Для людей: если известен SPEAKER_XX, указывай в скобках рядом с именем
- Таймстампы бери из транскрипции
- Отвечай ТОЛЬКО валидным JSON"""

    response = call_fn(prompt)
    return _parse_extraction_response(response, transcription)


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
                context=c.get("context", ""),
            ))

        for s in entities_data.get("services", []):
            entities.services.append(Entity(
                name=s.get("name", ""),
                description=s.get("description", ""),
                context=s.get("context", ""),
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
