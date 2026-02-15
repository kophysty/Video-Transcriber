"""Веб-поиск через DuckDuckGo (бесплатный, без API-ключей)."""

import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Результат веб-поиска."""
    title: str
    url: str
    snippet: str


def search_web(query: str, max_results: int = 5, max_retries: int = 3) -> list[SearchResult]:
    """
    Поиск через DuckDuckGo с retry при rate limit.

    Args:
        query: Поисковый запрос
        max_results: Максимум результатов
        max_retries: Максимум попыток при rate limit

    Returns:
        Список SearchResult
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log.warning("duckduckgo-search не установлен, веб-поиск недоступен")
        return []

    from duckduckgo_search.exceptions import RatelimitException

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in results
                if r.get("href")
            ]

        except RatelimitException:
            wait_time = 2 ** attempt
            log.warning(
                "DuckDuckGo rate limit, попытка %d/%d, ожидание %ds...",
                attempt + 1, max_retries, wait_time,
            )
            if attempt < max_retries - 1:
                time.sleep(wait_time)

        except Exception as e:
            log.error("Ошибка DuckDuckGo поиска: %s", e)
            return []

    log.warning("DuckDuckGo: все попытки исчерпаны для запроса: %s", query)
    return []
