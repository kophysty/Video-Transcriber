#!/usr/bin/env python3
"""CLI-раннер транскрибации без GUI.

Прогоняет существующий TranscriptionPipeline (тот же движок, что и GUI)
на одном видеофайле и складывает артефакты в указанную папку.

Использование:
    python transcribe_cli.py --input "C:/.../video.mp4" --output "D:/Transcribe/2026-07-23_Тема"

Результат печатается на stdout одной строкой JSON после маркера ---RESULT-JSON---,
чтобы вызывающий скилл мог его распарсить независимо от логов пайплайна.
"""

import argparse
import json
import sys
from pathlib import Path

# Корень проекта Video-Transcribe (родитель automation/) в sys.path,
# чтобы работали импорты app.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows workaround: platform.uname() зависает — как в main.py
if sys.platform == "win32":
    import os
    import platform
    _wv = sys.getwindowsversion()
    platform._uname_cache = platform.uname_result(
        system="Windows",
        node=os.environ.get("COMPUTERNAME", ""),
        release="11" if _wv.build >= 22000 else "10",
        version=f"{_wv.major}.{_wv.minor}.{_wv.build}",
        machine=os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
    )


def _emit(payload: dict) -> None:
    """Напечатать машиночитаемый результат для вызывающего процесса."""
    print("---RESULT-JSON---")
    print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Транскрибация одного видео без GUI")
    parser.add_argument("--input", required=True, help="Путь к видеофайлу")
    parser.add_argument("--output", required=True, help="Папка для артефактов")
    parser.add_argument(
        "--base-name",
        default=None,
        help="Имя выходных файлов (по умолчанию — из имени видео)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        _emit({"ok": False, "error": f"Видео не найдено: {input_path}"})
        return 1

    # Логирование в файл проекта (как в main.py), чтобы не мешать stdout-результату
    import logging
    from datetime import datetime
    logs_dir = PROJECT_ROOT / "Logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"automation_{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),  # логи в stderr, результат — в stdout
            logging.FileHandler(str(log_file), encoding="utf-8"),
        ],
    )
    for noisy in ("urllib3", "httpx", "httpcore", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from app.utils.config import AppConfig
    from app.models.gpu_detector import detect_gpu
    from app.core.pipeline import TranscriptionPipeline

    config = AppConfig.load()
    gpu_info = detect_gpu()
    # Как в GUI: если есть GPU, берём рекомендованный compute_type
    if gpu_info:
        config.compute_type = gpu_info.recommended_compute_type

    base_name = args.base_name or input_path.stem

    pipeline = TranscriptionPipeline(config=config, gpu_info=gpu_info)

    try:
        result = pipeline.run(input_path, output_dir, base_name=base_name)
    except Exception as e:  # noqa: BLE001 — верхний уровень, отдаём ошибку наружу
        logging.getLogger("automation").exception("Пайплайн упал")
        _emit({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1

    analysis_md = output_dir / f"{base_name}_analysis.md"
    transcript_md = output_dir / f"{base_name}.md"

    _emit({
        "ok": True,
        "output_dir": str(output_dir),
        "base_name": base_name,
        "duration_sec": round(result.duration, 1),
        "segments": len(result.transcription.segments),
        "has_analysis": analysis_md.exists(),
        "analysis_md": str(analysis_md) if analysis_md.exists() else None,
        "transcript_md": str(transcript_md) if transcript_md.exists() else None,
        "warnings": result.warnings,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
