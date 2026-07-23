#!/usr/bin/env python3
r"""Публикация summary встречи в Notion-базу Meeting Notes.

Создаёт новую страницу в базе с properties (Meeting name, Date, Client,
Category) и телом из markdown-файла анализа (_analysis.md).

Паттерн работы с Notion API — как в проекте OpenClaw:
httpx + Bearer token + Notion-Version + Content-Type charset=utf-8.

Секреты (токен, id базы) НЕ хардкодятся — приходят аргументами из конфига,
который лежит в персональной папке пользователя.

CLI:
    python notion_push.py \
        --token-env NOTION_TOKEN \        # или --token <val>
        --database-id <id> \
        --title "2026-07-23 — Тема встречи" \
        --date 2026-07-23 \
        --client Imperia \                # опционально
        --category "Customer call" \      # опционально, можно несколько через ;
        --md-file "D:/Transcribe/.../video_analysis.md"

Результат: строка JSON после ---RESULT-JSON--- с url созданной страницы.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import httpx

NOTION_VERSION = "2022-06-28"
MAX_TEXT = 1900  # запас под лимит Notion в 2000 символов на rich_text
MAX_CHILDREN = 100  # лимит блоков за один запрос


def _rich(text: str) -> list:
    """Разбить длинный текст на несколько rich_text (лимит 2000)."""
    chunks = [text[i:i + MAX_TEXT] for i in range(0, len(text), MAX_TEXT)] or [""]
    return [{"type": "text", "text": {"content": c}} for c in chunks]


def _block(btype: str, text: str) -> dict:
    return {"object": "block", "type": btype, btype: {"rich_text": _rich(text)}}


def md_to_blocks(md: str) -> list:
    """Простой markdown → Notion blocks. Покрывает то, что генерит _analysis.md:
    заголовки #/##/###, буллеты (-/*), нумерованные, цитаты (>), остальное — параграф.
    Пустые строки пропускаются. Код-фенсы ``` сбрасываются в параграфы.
    """
    blocks: list = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("```"):
            continue  # игнорируем маркеры код-блоков, содержимое пойдёт параграфами
        if line.startswith("### "):
            blocks.append(_block("heading_3", line[4:]))
        elif line.startswith("## "):
            blocks.append(_block("heading_2", line[3:]))
        elif line.startswith("# "):
            blocks.append(_block("heading_1", line[2:]))
        elif line.startswith("> "):
            blocks.append(_block("quote", line[2:]))
        elif re.match(r"^\s*[-*] ", line):
            blocks.append(_block("bulleted_list_item", re.sub(r"^\s*[-*] ", "", line)))
        elif re.match(r"^\s*\d+\. ", line):
            blocks.append(_block("numbered_list_item", re.sub(r"^\s*\d+\. ", "", line)))
        else:
            blocks.append(_block("paragraph", line))
    return blocks


def build_properties(title: str, date: str | None, client: str | None,
                     category: str | None) -> dict:
    props: dict = {
        "Meeting name": {"title": [{"type": "text", "text": {"content": title}}]},
    }
    if date:
        props["Date"] = {"date": {"start": date}}
    if client:
        props["Client"] = {"select": {"name": client}}
    if category:
        names = [c.strip() for c in category.split(";") if c.strip()]
        props["Category"] = {"multi_select": [{"name": n} for n in names]}
    return props


def push(token: str, database_id: str, title: str, date: str | None,
         client: str | None, category: str | None, blocks: list) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json; charset=utf-8",
    }
    # Создаём страницу с первыми 100 блоками
    payload = {
        "parent": {"database_id": database_id},
        "properties": build_properties(title, date, client, category),
        "children": blocks[:MAX_CHILDREN],
    }
    r = httpx.post(
        "https://api.notion.com/v1/pages",
        headers=headers,
        content=json.dumps(payload).encode("utf-8"),
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Notion create page {r.status_code}: {r.text[:500]}")
    page = r.json()
    page_id = page["id"]

    # Дописываем оставшиеся блоки батчами по 100
    rest = blocks[MAX_CHILDREN:]
    for i in range(0, len(rest), MAX_CHILDREN):
        chunk = rest[i:i + MAX_CHILDREN]
        ra = httpx.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=headers,
            content=json.dumps({"children": chunk}).encode("utf-8"),
            timeout=60,
        )
        if ra.status_code != 200:
            raise RuntimeError(f"Notion append {ra.status_code}: {ra.text[:500]}")

    return {"id": page_id, "url": page.get("url")}


def _unwrap_nested_json(text: str) -> dict | None:
    """Встроенный анализатор иногда кладёт весь ответ Claude сырым JSON-блоком
    внутрь поля (см. summary.detailed). Достаём настоящие поля обратно.

    Возвращает dict с ключами one_liner/paragraph/detailed, либо None если
    вложенного JSON нет.
    """
    if not text:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        # иногда без fence — просто голый объект после префикса
        m2 = re.search(r"(\{\s*\"one_liner\".*\})", text, re.DOTALL)
        if not m2:
            return None
        raw = m2.group(1)
    else:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def blocks_from_analysis_json(path: Path) -> list:
    """Собрать чистое тело Notion из _analysis.json (устойчиво к багу, когда
    настоящий summary завёрнут в JSON-строку внутри detailed).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {}) or {}
    one_liner = (summary.get("one_liner") or "").strip()
    paragraph = (summary.get("paragraph") or "").strip()
    detailed = (summary.get("detailed") or "").strip()

    # Если detailed на самом деле — обёрнутый JSON, распаковываем и берём
    # оттуда полные поля (они не обрезаны, в отличие от верхних).
    nested = _unwrap_nested_json(detailed)
    if nested:
        one_liner = one_liner or (nested.get("one_liner") or "").strip()
        paragraph = (nested.get("paragraph") or paragraph).strip()
        detailed = (nested.get("detailed") or "").strip()

    blocks: list = []
    if one_liner:
        blocks.append(_block("heading_2", "Кратко"))
        blocks.append(_block("paragraph", one_liner))
    if paragraph:
        blocks.append(_block("heading_2", "О встрече"))
        blocks.append(_block("paragraph", paragraph))
    if detailed:
        blocks.append(_block("heading_2", "Детально"))
        # detailed — markdown с ## / буллетами → разложить как обычный md
        blocks.extend(md_to_blocks(detailed))

    # Highlights и entities, если анализатор их заполнил
    highlights = data.get("highlights", []) or []
    if highlights:
        blocks.append(_block("heading_2", "Ключевые моменты"))
        for h in highlights:
            ts = h.get("timestamp", "")
            txt = h.get("text", "")
            blocks.append(_block("bulleted_list_item", f"{ts} — {txt}".strip(" —")))

    ents = data.get("entities", {}) or {}
    ent_lines = []
    for grp, label in (("companies", "Компании"), ("services", "Сервисы"), ("people", "Люди")):
        items = ents.get(grp, []) or []
        for e in items:
            name = e.get("name", "")
            desc = e.get("description") or e.get("context") or ""
            ent_lines.append(f"{name}: {desc}".strip(": "))
    if ent_lines:
        blocks.append(_block("heading_2", "Упомянутые сущности"))
        for line in ent_lines:
            blocks.append(_block("bulleted_list_item", line))

    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Публикация встречи в Notion")
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-env", default="NOTION_TOKEN")
    parser.add_argument("--database-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--date", default=None)
    parser.add_argument("--client", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--analysis-json", default=None,
                        help="Путь к _analysis.json — чистый режим (приоритетный)")
    parser.add_argument("--md-file", default=None,
                        help="Путь к _analysis.md — фоллбэк, если нет --analysis-json")
    args = parser.parse_args()

    token = args.token or os.environ.get(args.token_env, "")
    if not token:
        print("---RESULT-JSON---")
        print(json.dumps({"ok": False, "error": "Нет Notion токена"}, ensure_ascii=False))
        return 1

    # Приоритет: структурированный _analysis.json (устойчив к багу форматирования)
    if args.analysis_json:
        aj = Path(args.analysis_json)
        if not aj.exists():
            print("---RESULT-JSON---")
            print(json.dumps({"ok": False, "error": f"Нет analysis-json: {aj}"}, ensure_ascii=False))
            return 1
        blocks = blocks_from_analysis_json(aj)
    elif args.md_file:
        md_path = Path(args.md_file)
        if not md_path.exists():
            print("---RESULT-JSON---")
            print(json.dumps({"ok": False, "error": f"Нет md-файла: {md_path}"}, ensure_ascii=False))
            return 1
        blocks = md_to_blocks(md_path.read_text(encoding="utf-8"))
    else:
        print("---RESULT-JSON---")
        print(json.dumps({"ok": False, "error": "Нужен --analysis-json или --md-file"}, ensure_ascii=False))
        return 1

    try:
        res = push(token, args.database_id, args.title, args.date,
                   args.client, args.category, blocks)
    except Exception as e:  # noqa: BLE001
        print("---RESULT-JSON---")
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        return 1

    print("---RESULT-JSON---")
    print(json.dumps({"ok": True, "url": res["url"], "page_id": res["id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
