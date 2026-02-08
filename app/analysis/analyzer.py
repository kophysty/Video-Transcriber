"""Главный класс анализатора транскрипций."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import json

from app.core.transcriber import TranscriptionResult


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

        # Создаём саммари
        self._report_progress(0.1, "Создание саммари...")
        from app.analysis.summarizer import create_summary
        summary = create_summary(call_claude, full_text)

        # Извлекаем ключевые моменты и сущности
        self._report_progress(0.4, "Извлечение ключевых моментов...")
        from app.analysis.entity_extractor import extract_highlights_and_entities
        highlights, entities = extract_highlights_and_entities(
            call_claude, full_text, transcription
        )

        # Ищем ссылки на сервисы
        if resolve_links and (entities.companies or entities.services):
            self._report_progress(0.7, "Поиск ссылок...")
            from app.analysis.link_resolver import resolve_entity_links
            entities = resolve_entity_links(
                call_claude, entities, api_key=api_key
            )

        self._report_progress(1.0, "Анализ завершён")

        return AnalysisResult(
            summary=summary,
            highlights=highlights,
            entities=entities,
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
