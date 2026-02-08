"""Обёртка для вызова Claude Code CLI."""

import shutil
import subprocess


def is_claude_available() -> bool:
    """Проверить наличие Claude CLI в системе."""
    return shutil.which("claude") is not None


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
    if not is_claude_available():
        raise RuntimeError(
            "Claude CLI не найден. Установите Claude Code: "
            "npm install -g @anthropic-ai/claude-code"
        )

    try:
        result = subprocess.run(
            ["claude", "-p", "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
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
