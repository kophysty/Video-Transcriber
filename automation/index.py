#!/usr/bin/env python3
"""Индекс обработанных видео.

Хранит список уже транскрибированных видео и порог даты, с которого
скилл начал вести учёт. Старые видео (до порога) игнорируются —
не всплывают как "новые".

Формат индекса (JSON, UTF-8):
{
  "threshold_ts": 1690000000.0,     # видео старше — не показываем
  "processed": {
    "<video_name>": {
      "mtime": 1690000000.0,
      "size": 661979693,
      "folder": "D:/Transcribe/2026-07-23_Тема",
      "notion_url": "https://www.notion.so/...",
      "processed_at": "2026-07-23T16:40:00"
    }
  }
}

CLI:
    python index.py list   --videos-dir DIR --index PATH   # новые видео -> JSON stdout
    python index.py add    --index PATH --video-name NAME --mtime M --size S \
                           --folder F [--notion-url URL] [--processed-at ISO]
"""

import argparse
import json
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4a", ".mp3", ".wav", ".flac", ".ogg"}


def load_index(index_path: Path) -> dict:
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"threshold_ts": None, "processed": {}}


def save_index(index_path: Path, data: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_new_videos(videos_dir: Path, index_path: Path, now_ts: float) -> dict:
    """Вернуть новые видео (новее порога и не в индексе).

    now_ts передаётся снаружи (скилл знает текущее время), чтобы избежать
    зависимости от локальных часов внутри и упростить тестирование.
    При первом запуске порог = now_ts: учёт стартует с сегодняшнего момента,
    старьё не трогаем.
    """
    data = load_index(index_path)

    first_run = data.get("threshold_ts") is None
    if first_run:
        data["threshold_ts"] = now_ts
        save_index(index_path, data)

    threshold = data["threshold_ts"]
    processed = data.get("processed", {})

    new_videos = []
    if videos_dir.exists():
        for entry in videos_dir.iterdir():
            if not entry.is_file() or entry.suffix.lower() not in VIDEO_EXTS:
                continue
            st = entry.stat()
            if st.st_mtime < threshold:
                continue
            if entry.name in processed:
                continue
            new_videos.append({
                "name": entry.name,
                "path": str(entry),
                "mtime": st.st_mtime,
                "size": st.st_size,
            })

    new_videos.sort(key=lambda v: v["mtime"], reverse=True)
    return {
        "first_run": first_run,
        "threshold_ts": threshold,
        "new_videos": new_videos,
    }


def add_processed(index_path: Path, video_name: str, mtime: float, size: int,
                  folder: str, notion_url: str | None, processed_at: str) -> None:
    data = load_index(index_path)
    data.setdefault("processed", {})[video_name] = {
        "mtime": mtime,
        "size": size,
        "folder": folder,
        "notion_url": notion_url,
        "processed_at": processed_at,
    }
    save_index(index_path, data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Индекс обработанных видео")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Список новых видео")
    p_list.add_argument("--videos-dir", required=True)
    p_list.add_argument("--index", required=True)
    p_list.add_argument("--now-ts", type=float, required=True,
                        help="Текущее время (unix ts) — для порога первого запуска")

    p_add = sub.add_parser("add", help="Отметить видео обработанным")
    p_add.add_argument("--index", required=True)
    p_add.add_argument("--video-name", required=True)
    p_add.add_argument("--mtime", type=float, required=True)
    p_add.add_argument("--size", type=int, required=True)
    p_add.add_argument("--folder", required=True)
    p_add.add_argument("--notion-url", default=None)
    p_add.add_argument("--processed-at", required=True)

    args = parser.parse_args()

    if args.cmd == "list":
        result = list_new_videos(Path(args.videos_dir), Path(args.index), args.now_ts)
        print(json.dumps(result, ensure_ascii=False))
    elif args.cmd == "add":
        add_processed(
            Path(args.index), args.video_name, args.mtime, args.size,
            args.folder, args.notion_url, args.processed_at,
        )
        print(json.dumps({"ok": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
