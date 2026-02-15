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
    Найти путь к Claude CLI на любой платформе.

    Проверяет PATH и типичные директории npm global install
    для Windows, macOS и Linux.
    """
    # 1. Стандартный поиск через PATH (работает везде если настроен правильно)
    found = shutil.which("claude")
    if found:
        log.info("Claude CLI найден через PATH: %s", found)
        return found

    # 2. Windows: проверяем npm global директорию
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            npm_dir = Path(appdata) / "npm"
            for name in ("claude.cmd", "claude.exe", "claude"):
                candidate = npm_dir / name
                if candidate.exists():
                    log.info("Claude CLI найден (Windows npm): %s", candidate)
                    return str(candidate)

    # 3. macOS/Linux: проверяем типичные пути npm global
    else:
        home = Path.home()
        npm_paths = [
            Path("/usr/local/bin/claude"),           # npm -g (system-wide)
            home / ".npm-global/bin/claude",          # custom npm prefix
            home / ".local/bin/claude",               # local install
        ]

        # Также проверяем nvm (Node Version Manager)
        nvm_dir = os.environ.get("NVM_DIR")
        if nvm_dir:
            nvm_path = Path(nvm_dir)
            # Ищем claude в текущей версии node
            for node_dir in (nvm_path / "versions/node").glob("v*/bin"):
                npm_paths.append(node_dir / "claude")

        for candidate in npm_paths:
            if candidate.exists() and candidate.is_file():
                log.info("Claude CLI найден (Unix npm): %s", candidate)
                return str(candidate)

    log.warning("Claude CLI не найден в PATH и типичных директориях npm")
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
