"""Обёртка для вызова Claude Code CLI."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _find_claude_executable() -> str | None:
    """
    Найти путь к Claude CLI.

    shutil.which не всегда находит npm-глобальные пакеты на Windows,
    поэтому проверяем типичные пути вручную.
    """
    # 1. Стандартный поиск через PATH
    found = shutil.which("claude")
    if found:
        return found

    # 2. Windows: проверяем npm global директорию
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            npm_dir = Path(appdata) / "npm"
            for name in ("claude.cmd", "claude.exe", "claude"):
                candidate = npm_dir / name
                if candidate.exists():
                    log.info("Claude CLI найден: %s", candidate)
                    return str(candidate)

    return None


def is_claude_available() -> bool:
    """Проверить наличие Claude CLI в системе."""
    result = _find_claude_executable() is not None
    if not result:
        log.warning("Claude CLI не найден в PATH и типичных директориях npm")
    return result


def call_claude(prompt: str, timeout: int = 300) -> str:
    """
    Вызвать Claude CLI и вернуть текстовый ответ.

    Промпт передаётся через stdin чтобы не было проблем
    с длинными текстами и спецсимволами в аргументах.

    Args:
        prompt: Текст промпта
        timeout: Таймаут в секундах (по умолчанию 300)

    Returns:
        Текстовый ответ Claude

    Raises:
        RuntimeError: Если Claude CLI недоступен или вернул ошибку
    """
    claude_path = _find_claude_executable()
    if not claude_path:
        raise RuntimeError(
            "Claude CLI не найден. Установите Claude Code: "
            "npm install -g @anthropic-ai/claude-code"
        )

    log.info("Вызов Claude CLI: %s", claude_path)

    try:
        result = subprocess.run(
            [claude_path, "-p", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Claude CLI вернул ошибку (код {result.returncode}): {stderr}"
            )

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Claude CLI не ответил за {timeout} секунд. "
            "Попробуйте уменьшить объём текста."
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Claude CLI не найден. Установите Claude Code: "
            "npm install -g @anthropic-ai/claude-code"
        )
