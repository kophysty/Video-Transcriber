"""Определение GPU, VRAM, архитектуры и рекомендация настроек."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Информация о GPU."""

    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: tuple[int, int]  # (major, minor)
    architecture: str  # "ampere", "ada", "turing", "pascal", etc.
    recommended_compute_type: str  # "float16", "int8", "float32"
    recommended_model: str  # "large-v3-turbo", "medium", etc.


# Архитектуры NVIDIA по compute capability
ARCHITECTURES = {
    (6, 0): "pascal",   # GP100
    (6, 1): "pascal",   # GTX 1080, 1070, 1060
    (6, 2): "pascal",   # GP10B (Jetson TX2)
    (7, 0): "volta",    # V100
    (7, 5): "turing",   # RTX 2080, 2070, 2060
    (8, 0): "ampere",   # A100
    (8, 6): "ampere",   # RTX 3090, 3080, 3070, 3060
    (8, 7): "ampere",   # Jetson Orin
    (8, 9): "ada",      # RTX 4090, 4080, 4070, 4060
    (9, 0): "hopper",   # H100
}


def detect_gpu() -> Optional[GPUInfo]:
    """
    Определить GPU и рекомендовать настройки.

    Returns:
        GPUInfo или None если NVIDIA GPU не найден
    """
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return None

    try:
        # Берём первый GPU (для multi-GPU можно расширить)
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return None

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Имя
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        # Память
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total_mb = mem_info.total // (1024 * 1024)
        vram_free_mb = mem_info.free // (1024 * 1024)

        # Compute capability
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = (major, minor)

        # Архитектура
        architecture = _classify_architecture(major, minor)

        # Рекомендации
        recommended_compute_type = _recommend_compute_type(architecture)
        recommended_model = _recommend_model(vram_free_mb, recommended_compute_type)

        return GPUInfo(
            name=name,
            vram_total_mb=vram_total_mb,
            vram_free_mb=vram_free_mb,
            compute_capability=compute_capability,
            architecture=architecture,
            recommended_compute_type=recommended_compute_type,
            recommended_model=recommended_model,
        )

    except Exception as e:
        print(f"Ошибка определения GPU: {e}")
        return None

    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _classify_architecture(major: int, minor: int) -> str:
    """Определить архитектуру по compute capability."""

    # Точное совпадение
    if (major, minor) in ARCHITECTURES:
        return ARCHITECTURES[(major, minor)]

    # По major версии
    if major >= 9:
        return "hopper"
    elif major == 8:
        if minor >= 9:
            return "ada"
        return "ampere"
    elif major == 7:
        if minor >= 5:
            return "turing"
        return "volta"
    elif major == 6:
        return "pascal"

    return "unknown"


def _recommend_compute_type(architecture: str) -> str:
    """
    Рекомендовать compute_type для архитектуры.

    - Ampere+ (8.0+): float16 — нативные FP16 tensor cores
    - Turing (7.5): float16 — есть FP16 tensor cores
    - Pascal (6.x): int8 — нет FP16 tensor cores, FP16 медленнее
    - Другие: int8 как безопасный вариант
    """
    if architecture in ("ampere", "ada", "hopper"):
        return "float16"
    elif architecture == "turing":
        return "float16"
    elif architecture in ("volta",):
        return "float16"
    elif architecture == "pascal":
        return "int8"  # FP16 на Pascal медленнее чем FP32

    return "int8"


def _recommend_model(vram_free_mb: int, compute_type: str) -> str:
    """
    Рекомендовать модель на основе свободной VRAM.

    VRAM требования (CTranslate2, примерные):
      large-v3       float16: ~5 GB,  int8: ~3 GB
      large-v3-turbo float16: ~3 GB,  int8: ~2 GB
      medium         float16: ~2.5 GB, int8: ~1.5 GB
      small          float16: ~1 GB,  int8: ~0.7 GB

    Оставляем запас ~2 GB на pyannote и системные нужды.
    """
    # Вычитаем запас
    available = vram_free_mb - 2000

    if compute_type == "float16":
        if available >= 5000:
            return "large-v3"
        elif available >= 3000:
            return "large-v3-turbo"
        elif available >= 2500:
            return "medium"
        elif available >= 1000:
            return "small"
        else:
            return "base"
    else:  # int8
        if available >= 3000:
            return "large-v3"
        elif available >= 2000:
            return "large-v3-turbo"
        elif available >= 1500:
            return "medium"
        elif available >= 700:
            return "small"
        else:
            return "base"


def get_device_string(gpu_info: Optional[GPUInfo]) -> str:
    """Получить строку устройства для UI."""
    if gpu_info:
        return f"{gpu_info.name} ({gpu_info.vram_total_mb // 1024} GB)"
    return "CPU (GPU не обнаружен)"


def get_compute_type_for_device(gpu_info: Optional[GPUInfo]) -> str:
    """Получить compute_type для устройства."""
    if gpu_info:
        return gpu_info.recommended_compute_type
    return "int8"  # Для CPU


def get_device_name(gpu_info: Optional[GPUInfo]) -> str:
    """Получить имя устройства для faster-whisper."""
    if gpu_info:
        return "cuda"
    return "cpu"
