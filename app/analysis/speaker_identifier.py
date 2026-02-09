"""Идентификация имён спикеров через Claude CLI."""

import json
import logging
from typing import Callable

log = logging.getLogger(__name__)


def identify_speakers(
    call_fn: Callable[[str], str],
    transcript_text: str,
) -> dict[str, str]:
    """
    Определить имена спикеров по контексту транскрипции.

    Args:
        call_fn: Функция вызова LLM (prompt -> response text)
        transcript_text: Текст транскрипции с метками SPEAKER_XX

    Returns:
        Маппинг {"SPEAKER_00": "Иван Петров", "SPEAKER_01": "Мария"}
        Если имя не определилось — метка не включается в результат.
    """
    prompt = f"""Проанализируй транскрипцию и определи реальные имена спикеров.

Спикеры часто представляются в начале разговора, обращаются друг к другу по имени,
или их имена упоминаются в контексте.

ТРАНСКРИПЦИЯ:
{transcript_text}

Верни JSON-объект, где ключи — метки спикеров (SPEAKER_00, SPEAKER_01, ...),
а значения — определённые имена.

Правила:
- Если имя точно определяется — используй полное имя (например "Иван Петров")
- Если определяется только имя — используй его (например "Мария")
- Если имя НЕ удаётся определить — НЕ включай этого спикера в результат

Пример ответа:
{{"SPEAKER_00": "Иван Петров", "SPEAKER_01": "Мария"}}

Отвечай ТОЛЬКО валидным JSON без дополнительного текста."""

    try:
        response = call_fn(prompt)
        return _parse_speaker_response(response)
    except Exception as e:
        log.warning("Не удалось определить имена спикеров: %s", e)
        return {}


def _parse_speaker_response(response_text: str) -> dict[str, str]:
    """Парсить ответ модели в маппинг спикеров."""
    text = response_text.strip()

    # Убираем markdown code block если есть
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    data = json.loads(text.strip())

    if not isinstance(data, dict):
        log.warning("Ответ не является объектом: %s", type(data))
        return {}

    # Фильтруем: только строковые ключи SPEAKER_XX и строковые значения
    result = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("SPEAKER_") and isinstance(value, str) and value.strip():
            result[key] = value.strip()

    return result
