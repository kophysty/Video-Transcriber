"""Реестр моделей с метаданными."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WhisperModelInfo:
    """Информация о Whisper модели."""
    name: str
    hf_repo: str
    params: str
    size_mb: int  # Примерный размер на диске
    vram_fp16_mb: int
    vram_int8_mb: int
    description_ru: str
    description_en: str


# Доступные Whisper модели
WHISPER_MODELS: dict[str, WhisperModelInfo] = {
    "large-v3": WhisperModelInfo(
        name="large-v3",
        hf_repo="Systran/faster-whisper-large-v3",
        params="1550M",
        size_mb=3100,
        vram_fp16_mb=5000,
        vram_int8_mb=3000,
        description_ru="Лучшее качество, 1.5B параметров",
        description_en="Best quality, 1.5B params",
    ),
    "large-v3-turbo": WhisperModelInfo(
        name="large-v3-turbo",
        hf_repo="deepdml/faster-whisper-large-v3-turbo-ct2",
        params="809M",
        size_mb=1550,
        vram_fp16_mb=3000,
        vram_int8_mb=2000,
        description_ru="Близко к large-v3, в 8\u00d7 быстрее",
        description_en="Near large-v3 quality, 8x faster",
    ),
    "medium": WhisperModelInfo(
        name="medium",
        hf_repo="Systran/faster-whisper-medium",
        params="769M",
        size_mb=1500,
        vram_fp16_mb=2500,
        vram_int8_mb=1500,
        description_ru="Баланс скорости и качества, 769M параметров",
        description_en="Speed/quality balance, 769M params",
    ),
    "small": WhisperModelInfo(
        name="small",
        hf_repo="Systran/faster-whisper-small",
        params="244M",
        size_mb=500,
        vram_fp16_mb=1000,
        vram_int8_mb=700,
        description_ru="Быстрый, хорош для чистого аудио",
        description_en="Fast, good for clean audio",
    ),
    "base": WhisperModelInfo(
        name="base",
        hf_repo="Systran/faster-whisper-base",
        params="74M",
        size_mb=150,
        vram_fp16_mb=500,
        vram_int8_mb=300,
        description_ru="Очень быстрый, для простых записей",
        description_en="Very fast, for simple recordings",
    ),
    "tiny": WhisperModelInfo(
        name="tiny",
        hf_repo="Systran/faster-whisper-tiny",
        params="39M",
        size_mb=80,
        vram_fp16_mb=300,
        vram_int8_mb=200,
        description_ru="Мгновенный, минимальное качество",
        description_en="Instant, minimal quality",
    ),
}


# Порядок моделей для отображения (от лучшей к простой)
MODEL_ORDER = ["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"]


def get_model_info(model_name: str) -> Optional[WhisperModelInfo]:
    """Получить информацию о модели."""
    return WHISPER_MODELS.get(model_name)


def get_all_models() -> list[WhisperModelInfo]:
    """Получить все модели в порядке отображения."""
    return [WHISPER_MODELS[name] for name in MODEL_ORDER if name in WHISPER_MODELS]


def get_vram_requirement(model_name: str, compute_type: str) -> int:
    """Получить требование к VRAM для модели."""
    info = get_model_info(model_name)
    if not info:
        return 0

    if compute_type == "float16":
        return info.vram_fp16_mb
    else:  # int8, float32
        return info.vram_int8_mb
