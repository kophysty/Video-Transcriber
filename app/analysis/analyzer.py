"""Главный класс анализатора транскрипций."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from app.core.transcriber import TranscriptionResult

log = logging.getLogger(__name__)


@dataclass
class Summary:
    """Саммари транскрипции."""
    one_liner: str = ""
    paragraph: str = ""
    detailed: str = ""


@dataclass
class Highlight:
    """Важный момент с таймкодом."""
    timestamp: str
    text: str
    start_seconds: float = 0.0


@dataclass
class Entity:
    """Извлечённая сущность."""
    name: str
    description: str
    url: Optional[str] = None
    context: Optional[str] = None


@dataclass
class Entities:
    """Все извлечённые сущности."""
    companies: list[Entity] = field(default_factory=list)
    services: list[Entity] = field(default_factory=list)
    people: list[Entity] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Результат AI-анализа."""
    summary: Summary
    highlights: list[Highlight]
    entities: Entities
    fact_checks: list = field(default_factory=list)  # list[FactCheckResult]

    def to_dict(self) -> dict:
        """Конвертировать в словарь для JSON."""
        return {
            "summary": {
                "one_liner": self.summary.one_liner,
                "paragraph": self.summary.paragraph,
                "detailed": self.summary.detailed,
            },
            "highlights": [
                {
                    "timestamp": h.timestamp,
                    "text": h.text,
                    "start_seconds": h.start_seconds,
                }
                for h in self.highlights
            ],
            "entities": {
                "companies": [
                    {"name": e.name, "description": e.description, "url": e.url, "context": e.context}
                    for e in self.entities.companies
                ],
                "services": [
                    {"name": e.name, "description": e.description, "url": e.url, "context": e.context}
                    for e in self.entities.services
                ],
                "people": [
                    {"name": e.name, "context": e.context}
                    for e in self.entities.people
                ],
            },
            **({"fact_checks": [
                {
                    "claim": fc.claim,
                    "timestamp": fc.timestamp,
                    "verdict": fc.verdict,
                    "confidence": fc.confidence,
                    "explanation": fc.explanation,
                    "sources": fc.sources,
                }
                for fc in self.fact_checks
            ]} if self.fact_checks else {}),
        }

    def to_markdown(self) -> str:
        """Конвертировать в Markdown."""
        lines = ["# Анализ транскрипции\n"]

        # Саммари
        lines.append("## Краткое содержание\n")
        lines.append(f"**В одной строке:** {self.summary.one_liner}\n")
        lines.append(f"\n{self.summary.paragraph}\n")

        if self.summary.detailed:
            lines.append("\n### Детальное описание\n")
            lines.append(self.summary.detailed)
            lines.append("\n")

        # Ключевые моменты
        if self.highlights:
            lines.append("\n## Ключевые моменты\n")
            for h in self.highlights:
                lines.append(f"- **{h.timestamp}** - {h.text}")
            lines.append("\n")

        # Сущности
        if self.entities.companies or self.entities.services or self.entities.people:
            lines.append("\n## Упомянутые сущности\n")

            if self.entities.companies:
                lines.append("\n### Компании\n")
                for e in self.entities.companies:
                    url_part = f" [{e.url}]({e.url})" if e.url else ""
                    lines.append(f"- **{e.name}**{url_part}: {e.description}")
                    if e.context:
                        lines.append(f"  *Контекст: {e.context}*")

            if self.entities.services:
                lines.append("\n### Сервисы и продукты\n")
                for e in self.entities.services:
                    url_part = f" [{e.url}]({e.url})" if e.url else ""
                    lines.append(f"- **{e.name}**{url_part}: {e.description}")
                    if e.context:
                        lines.append(f"  *Контекст: {e.context}*")

            if self.entities.people:
                lines.append("\n### Люди\n")
                for e in self.entities.people:
                    lines.append(f"- **{e.name}**")
                    if e.context:
                        lines.append(f"  *{e.context}*")

        # Факт-чекинг
        if self.fact_checks:
            lines.append("\n## Проверка фактов\n")

            verdict_emoji = {
                "confirmed": "[OK]",
                "disputed": "[??]",
                "unverified": "[--]",
                "likely_false": "[!!]",
            }
            verdict_label = {
                "confirmed": "Подтверждено",
                "disputed": "Спорно",
                "unverified": "Не проверено",
                "likely_false": "Вероятно неверно",
            }

            for fc in self.fact_checks:
                emoji = verdict_emoji.get(fc.verdict, "[--]")
                label = verdict_label.get(fc.verdict, fc.verdict)
                confidence_pct = int(fc.confidence * 100)

                lines.append(f"### {emoji} {fc.claim}\n")
                lines.append(f"- **Таймкод:** {fc.timestamp}")
                lines.append(f"- **Вердикт:** {label} (уверенность: {confidence_pct}%)")
                lines.append(f"- **Пояснение:** {fc.explanation}")

                if fc.sources:
                    lines.append("- **Источники:**")
                    for src in fc.sources:
                        lines.append(f"  - {src}")

                lines.append("")

        return "\n".join(lines)


class TranscriptAnalyzer:
    """Анализатор транскрипций через Claude CLI."""

    def __init__(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Args:
            progress_callback: Callback(progress: 0.0-1.0, message: str)
        """
        self.progress_callback = progress_callback

    def analyze(
        self,
        transcription: TranscriptionResult,
        resolve_links: bool = True,
        api_key: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Анализировать транскрипцию.

        Args:
            transcription: Результат транскрибации
            resolve_links: Искать ссылки на сервисы
            api_key: Опциональный Anthropic API ключ для web search

        Returns:
            AnalysisResult с саммари, хайлайтами и сущностями
        """
        from app.analysis.claude_cli import call_claude

        self._report_progress(0.0, "Подготовка текста...")

        # Собираем текст транскрипции
        full_text = self._prepare_transcript_text(transcription)

        # Создаём саммари (0.0 – 0.25)
        self._report_progress(0.05, "Создание саммари...")
        from app.analysis.summarizer import create_summary
        summary = create_summary(call_claude, full_text)

        # Извлекаем ключевые моменты и сущности (0.25 – 0.50)
        self._report_progress(0.25, "Извлечение ключевых моментов...")
        from app.analysis.entity_extractor import extract_highlights_and_entities
        highlights, entities = extract_highlights_and_entities(
            call_claude, full_text, transcription
        )

        # Факт-чекинг (0.50 – 0.75)
        fact_checks = []
        self._report_progress(0.50, "Проверка фактов...")
        try:
            from app.analysis.fact_checker import fact_check_transcript
            fact_checks = fact_check_transcript(
                call_claude, full_text, api_key=api_key
            )
        except Exception as e:
            log.warning("Факт-чекинг не удался: %s", e)

        # Ищем ссылки на сервисы (0.75 – 0.95)
        if resolve_links and (entities.companies or entities.services):
            self._report_progress(0.75, "Поиск ссылок...")
            from app.analysis.link_resolver import resolve_entity_links
            entities = resolve_entity_links(
                call_claude, entities, api_key=api_key
            )

        self._report_progress(1.0, "Анализ завершён")

        return AnalysisResult(
            summary=summary,
            highlights=highlights,
            entities=entities,
            fact_checks=fact_checks,
        )

    def _prepare_transcript_text(self, transcription: TranscriptionResult) -> str:
        """Подготовить текст транскрипции для анализа."""
        lines = []

        for segment in transcription.segments:
            # Форматируем таймстамп
            minutes = int(segment.start // 60)
            seconds = int(segment.start % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"

            # Добавляем спикера если есть
            if segment.speaker:
                lines.append(f"{timestamp} {segment.speaker}: {segment.text}")
            else:
                lines.append(f"{timestamp} {segment.text}")

        return "\n".join(lines)

    def _report_progress(self, progress: float, message: str) -> None:
        """Сообщить о прогрессе."""
        if self.progress_callback:
            self.progress_callback(progress, message)


def export_analysis(
    result: AnalysisResult,
    output_dir: Path,
    base_name: str,
) -> tuple[Path, Path]:
    """
    Экспортировать результаты анализа.

    Returns:
        Пути к JSON и Markdown файлам
    """
    # JSON
    json_path = output_dir / f"{base_name}_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    # Markdown
    md_path = output_dir / f"{base_name}_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.to_markdown())

    return json_path, md_path
