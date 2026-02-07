"""Главный класс анализатора транскрипций."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any
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
                    {"name": e.name, "description": e.description, "url": e.url}
                    for e in self.entities.companies
                ],
                "services": [
                    {"name": e.name, "description": e.description, "url": e.url}
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
                    url_part = f" - [{e.url}]({e.url})" if e.url else ""
                    lines.append(f"- **{e.name}**: {e.description}{url_part}")

            if self.entities.services:
                lines.append("\n### Сервисы\n")
                for e in self.entities.services:
                    url_part = f" - [{e.url}]({e.url})" if e.url else ""
                    lines.append(f"- **{e.name}**: {e.description}{url_part}")

            if self.entities.people:
                lines.append("\n### Люди\n")
                for e in self.entities.people:
                    context_part = f" ({e.context})" if e.context else ""
                    lines.append(f"- **{e.name}**{context_part}")

        return "\n".join(lines)


class TranscriptAnalyzer:
    """Анализатор транскрипций через Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250514",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Args:
            api_key: Anthropic API ключ
            model: Модель Claude для анализа
            progress_callback: Callback(progress: 0.0-1.0, message: str)
        """
        self.api_key = api_key
        self.model = model
        self.progress_callback = progress_callback
        self._client = None

    def _get_client(self):
        """Получить или создать клиент Anthropic."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Пакет anthropic не установлен. "
                    "Установите: pip install anthropic"
                )
        return self._client

    def validate_api_key(self) -> bool:
        """Проверить валидность API ключа."""
        if not self.api_key or not self.api_key.startswith("sk-ant-"):
            return False

        try:
            client = self._get_client()
            # Минимальный запрос для проверки ключа
            client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception:
            return False

    def analyze(
        self,
        transcription: TranscriptionResult,
        resolve_links: bool = True,
    ) -> AnalysisResult:
        """
        Анализировать транскрипцию.

        Args:
            transcription: Результат транскрибации
            resolve_links: Искать ссылки на сервисы

        Returns:
            AnalysisResult с саммари, хайлайтами и сущностями
        """
        self._report_progress(0.0, "Подготовка текста...")

        # Собираем текст транскрипции
        full_text = self._prepare_transcript_text(transcription)

        # Создаём саммари
        self._report_progress(0.1, "Создание саммари...")
        from app.analysis.summarizer import create_summary
        summary = create_summary(self._get_client(), self.model, full_text)

        # Извлекаем ключевые моменты и сущности
        self._report_progress(0.4, "Извлечение ключевых моментов...")
        from app.analysis.entity_extractor import extract_highlights_and_entities
        highlights, entities = extract_highlights_and_entities(
            self._get_client(), self.model, full_text, transcription
        )

        # Ищем ссылки на сервисы
        if resolve_links and (entities.companies or entities.services):
            self._report_progress(0.7, "Поиск ссылок...")
            from app.analysis.link_resolver import resolve_entity_links
            entities = resolve_entity_links(
                self._get_client(), self.model, entities
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
