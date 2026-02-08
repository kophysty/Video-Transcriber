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

    Использует Claude с web_search tool для поиска актуальных URL.
    При ошибке — фоллбэк на запрос из памяти модели.

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

    # Пробуем с web_search, при ошибке — фоллбэк
    try:
        url_map = _resolve_with_web_search(client, model, items_to_resolve)
    except Exception:
        try:
            url_map = _resolve_from_memory(client, model, items_to_resolve)
        except Exception:
            return entities

    # Применяем найденные URL к сущностям
    for company in entities.companies:
        url = url_map.get(company.name.lower())
        if url:
            company.url = url

    for service in entities.services:
        url = url_map.get(service.name.lower())
        if url:
            service.url = url

    return entities


def _resolve_with_web_search(
    client: Any,
    model: str,
    items: list[dict],
) -> dict[str, str]:
    """Найти URL через веб-поиск Claude."""
    items_json = json.dumps(items, ensure_ascii=False, indent=2)

    prompt = (
        f"Найди официальные сайты для каждого сервиса/компании из списка.\n\n"
        f"СПИСОК:\n{items_json}\n\n"
        f"Для каждого элемента найди через поиск их официальный сайт.\n"
        f"Верни JSON массив:\n"
        f'[{{"name": "Название", "url": "https://..."}}]\n\n'
        f"Если не нашёл сайт — ставь url: null.\n"
        f"Отвечай ТОЛЬКО валидным JSON массивом."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        tools=[{"name": "web_search", "type": "web_search_20250305"}],
        messages=[{"role": "user", "content": prompt}],
    )

    # Извлекаем текстовый ответ
    for block in response.content:
        if hasattr(block, "text"):
            return _parse_urls_to_map(block.text)

    return {}


def _resolve_from_memory(
    client: Any,
    model: str,
    items: list[dict],
) -> dict[str, str]:
    """Фоллбэк: запросить URL из памяти модели."""
    items_json = json.dumps(items, ensure_ascii=False, indent=2)

    prompt = (
        f"Для каждого сервиса или компании из списка укажи официальный сайт.\n\n"
        f"СПИСОК:\n{items_json}\n\n"
        f"Верни JSON массив в том же порядке:\n"
        f'[{{"name": "Название", "url": "https://example.com"}}]\n\n'
        f"Правила:\n"
        f"- Указывай только официальные сайты\n"
        f"- Если не уверен в URL — ставь null\n"
        f"- Не придумывай URL, только известные тебе\n"
        f"- Отвечай ТОЛЬКО валидным JSON массивом"
    )

    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    return _parse_urls_to_map(response.content[0].text)


def _parse_urls_to_map(response_text: str) -> dict[str, str]:
    """Парсить ответ с URL в словарь name→url."""
    urls = _parse_urls_response(response_text)
    return {
        item["name"].lower(): item["url"]
        for item in urls
        if item.get("url")
    }


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
