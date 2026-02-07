"""Поиск ссылок на упомянутые сервисы и компании."""

import json
from typing import Any

from app.analysis.analyzer import Entity, Entities


def resolve_entity_links(
    client: Any,
    model: str,
    entities: Entities,
) -> Entities:
    """
    Найти ссылки на упомянутые сервисы и компании.

    Использует Claude для поиска актуальных URL.

    Args:
        client: Anthropic клиент
        model: Модель Claude
        entities: Сущности для поиска ссылок

    Returns:
        Entities с заполненными URL
    """
    # Собираем все сервисы и компании для поиска
    items_to_resolve = []

    for company in entities.companies:
        items_to_resolve.append({
            "type": "company",
            "name": company.name,
            "description": company.description,
        })

    for service in entities.services:
        items_to_resolve.append({
            "type": "service",
            "name": service.name,
            "description": service.description,
        })

    if not items_to_resolve:
        return entities

    # Запрашиваем ссылки у Claude
    items_json = json.dumps(items_to_resolve, ensure_ascii=False, indent=2)

    prompt = f"""Для каждого сервиса или компании из списка укажи официальный сайт.

СПИСОК:
{items_json}

Верни JSON массив в том же порядке:
[
    {{
        "name": "Название",
        "url": "https://example.com" или null если не знаешь
    }}
]

Правила:
- Указывай только официальные сайты
- Если не уверен в URL — ставь null
- Не придумывай URL, только известные тебе
- Отвечай ТОЛЬКО валидным JSON массивом"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        urls = _parse_urls_response(response.content[0].text)

        # Применяем найденные URL к сущностям
        url_map = {item["name"].lower(): item.get("url") for item in urls}

        for company in entities.companies:
            url = url_map.get(company.name.lower())
            if url:
                company.url = url

        for service in entities.services:
            url = url_map.get(service.name.lower())
            if url:
                service.url = url

    except Exception:
        # При ошибке просто возвращаем сущности без URL
        pass

    return entities


def _parse_urls_response(response_text: str) -> list[dict]:
    """Парсить ответ с URL."""
    try:
        text = response_text.strip()

        # Убираем markdown code block
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        data = json.loads(text.strip())

        if isinstance(data, list):
            return data

    except json.JSONDecodeError:
        pass

    return []
