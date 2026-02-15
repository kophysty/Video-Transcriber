"""Факт-чекинг утверждений из транскрипции."""

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass
class FactCheckResult:
    """Результат проверки одного утверждения."""
    claim: str
    timestamp: str
    verdict: str  # confirmed, disputed, unverified, likely_false
    confidence: float  # 0.0-1.0
    explanation: str
    sources: list[str] = field(default_factory=list)


def fact_check_transcript(
    call_fn: Callable[[str], str],
    transcript_text: str,
    api_key: Optional[str] = None,
    max_claims: int = 10,
) -> list[FactCheckResult]:
    """
    Проверить факты из транскрипции.

    3-шаговая архитектура:
    1. Extract claims — LLM извлекает проверяемые утверждения
    2. Search evidence — поиск через DuckDuckGo или Anthropic web_search
    3. Synthesize verdict — LLM анализирует результаты → вердикт

    Args:
        call_fn: Функция вызова LLM (prompt -> response text)
        transcript_text: Текст транскрипции с таймкодами
        api_key: Опциональный Anthropic API ключ для web search
        max_claims: Максимум утверждений для проверки

    Returns:
        Список FactCheckResult
    """
    # Шаг 1: Извлечение проверяемых утверждений
    log.info("Факт-чекинг: извлечение утверждений...")
    claims = _extract_claims(call_fn, transcript_text, max_claims)
    if not claims:
        log.info("Проверяемые утверждения не найдены")
        return []

    log.info("Найдено %d утверждений для проверки", len(claims))

    # Шаг 2: Поиск доказательств
    log.info("Факт-чекинг: поиск доказательств...")
    claims_with_evidence = _search_evidence(claims, api_key)

    # Шаг 3: Синтез вердиктов
    log.info("Факт-чекинг: синтез вердиктов...")
    results = _synthesize_verdicts(call_fn, claims_with_evidence)

    return results


def _extract_claims(
    call_fn: Callable[[str], str],
    transcript_text: str,
    max_claims: int,
) -> list[dict]:
    """Шаг 1: извлечь проверяемые утверждения из транскрипции."""
    prompt = (
        f"Проанализируй транскрипцию и извлеки ПРОВЕРЯЕМЫЕ фактические утверждения.\n\n"
        f"ТРАНСКРИПЦИЯ:\n{transcript_text}\n\n"
        f"Ищи утверждения, которые можно проверить через интернет:\n"
        f"- Числовые данные (статистика, цены, даты)\n"
        f"- Факты о компаниях (основатели, выручка, количество сотрудников)\n"
        f"- Исторические события и даты\n"
        f"- Технические характеристики\n"
        f"- Цитаты и приписываемые высказывания\n\n"
        f"НЕ извлекай:\n"
        f"- Субъективные мнения\n"
        f"- Очевидные факты\n"
        f"- Расплывчатые утверждения без конкретики\n\n"
        f"Верни JSON массив (максимум {max_claims} утверждений):\n"
        f'[{{"claim": "Утверждение на языке оригинала", '
        f'"timestamp": "MM:SS", '
        f'"search_query": "Поисковый запрос для проверки на английском"}}]\n\n'
        f"Отвечай ТОЛЬКО валидным JSON массивом."
    )

    try:
        response = call_fn(prompt)
        return _parse_json_array(response)
    except Exception as e:
        log.error("Ошибка извлечения утверждений: %s", e)
        return []


def _search_evidence(
    claims: list[dict],
    api_key: Optional[str] = None,
) -> list[dict]:
    """Шаг 2: найти доказательства для каждого утверждения."""
    if api_key:
        return _search_with_anthropic(claims, api_key)
    else:
        return _search_with_duckduckgo(claims)


def _search_with_duckduckgo(claims: list[dict]) -> list[dict]:
    """Поиск доказательств через DuckDuckGo."""
    from app.analysis.web_search import search_web

    for claim in claims:
        query = claim.get("search_query", claim.get("claim", ""))
        if not query:
            claim["evidence"] = []
            continue

        results = search_web(query, max_results=3)
        claim["evidence"] = [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ]

    return claims


def _search_with_anthropic(claims: list[dict], api_key: str) -> list[dict]:
    """Поиск доказательств через Anthropic web_search. При ошибке API — fallback на DDG."""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
    except Exception as e:
        log.warning("Anthropic API недоступен для факт-чекинга: %s, фоллбэк на DDG", e)
        return _search_with_duckduckgo(claims)

    # Пробуем первый запрос — если API не работает, сразу переключаемся на DDG
    first_claim = True
    for claim in claims:
        query = claim.get("search_query", claim.get("claim", ""))
        if not query:
            claim["evidence"] = []
            continue

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                tools=[{"name": "web_search", "type": "web_search_20250305"}],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search for evidence about this claim: {query}\n"
                        f"Return 2-3 key findings as a JSON array:\n"
                        f'[{{"title": "...", "url": "...", "snippet": "..."}}]\n'
                        f"Only valid JSON array."
                    ),
                }],
            )

            evidence = []
            for block in response.content:
                if hasattr(block, "text"):
                    parsed = _parse_json_array(block.text)
                    evidence = [
                        {"title": e.get("title", ""), "url": e.get("url", ""), "snippet": e.get("snippet", "")}
                        for e in parsed
                    ]
                    break

            claim["evidence"] = evidence
            first_claim = False

        except Exception as e:
            if first_claim:
                log.warning("Anthropic API ошибка: %s — переключаюсь на DuckDuckGo для всех claims", e)
                return _search_with_duckduckgo(claims)
            log.warning("Anthropic поиск не удался для '%s': %s", query, e)
            claim["evidence"] = []

    return claims


def _synthesize_verdicts(
    call_fn: Callable[[str], str],
    claims_with_evidence: list[dict],
) -> list[FactCheckResult]:
    """Шаг 3: LLM анализирует доказательства и выносит вердикт."""
    claims_json = json.dumps(claims_with_evidence, ensure_ascii=False, indent=2)

    prompt = (
        f"Проанализируй утверждения и найденные доказательства, вынеси вердикт по каждому.\n\n"
        f"УТВЕРЖДЕНИЯ С ДОКАЗАТЕЛЬСТВАМИ:\n{claims_json}\n\n"
        f"Для каждого утверждения определи:\n"
        f"- verdict: одно из [confirmed, disputed, unverified, likely_false]\n"
        f"  - confirmed: доказательства подтверждают утверждение\n"
        f"  - disputed: есть противоречивые данные\n"
        f"  - unverified: недостаточно данных для проверки\n"
        f"  - likely_false: доказательства опровергают утверждение\n"
        f"- confidence: уверенность от 0.0 до 1.0\n"
        f"- explanation: краткое объяснение вердикта\n\n"
        f"Верни JSON массив:\n"
        f'[{{"claim": "...", "timestamp": "MM:SS", "verdict": "confirmed", '
        f'"confidence": 0.85, "explanation": "Объяснение...", '
        f'"sources": ["https://..."]}}]\n\n'
        f"Отвечай ТОЛЬКО валидным JSON массивом."
    )

    try:
        response = call_fn(prompt)
        parsed = _parse_json_array(response)

        results = []
        for item in parsed:
            verdict = item.get("verdict", "unverified")
            if verdict not in ("confirmed", "disputed", "unverified", "likely_false"):
                verdict = "unverified"

            confidence = item.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)):
                confidence = 0.5
            confidence = max(0.0, min(1.0, float(confidence)))

            results.append(FactCheckResult(
                claim=item.get("claim", ""),
                timestamp=item.get("timestamp", ""),
                verdict=verdict,
                confidence=confidence,
                explanation=item.get("explanation", ""),
                sources=item.get("sources", []),
            ))

        return results

    except Exception as e:
        log.error("Ошибка синтеза вердиктов: %s", e)
        return []


def _parse_json_array(text: str) -> list[dict]:
    """Парсить JSON массив из текстового ответа LLM."""
    try:
        text = text.strip()
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
