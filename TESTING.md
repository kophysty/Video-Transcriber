# Тестирование кроссплатформенности

## Проверка на разных ОС

Для тестирования на Windows/macOS/Linux запустите:

```bash
python check_paths.py
```

### Что проверяется:

1. **Пути приложения** — config.json, models/, whisper/
2. **FFmpeg** — автопоиск в системе
3. **Claude CLI** — поддержка npm global на всех платформах
4. **Логи** — папка Logs/ и файлы логов
5. **SpeechBrain модели** — путь к диаризации

### Ожидаемый вывод:

```
============================================================
Проверка кроссплатформенности путей
============================================================
Платформа: win32 / darwin / linux
Python: 3.x.x

1. Пути приложения:
   Config:  /path/to/PG-Video-Transcriber/config.json
   Models:  /path/to/PG-Video-Transcriber/models
   Whisper: /path/to/PG-Video-Transcriber/models/whisper

2. FFmpeg:
   [OK] FFmpeg найден:  /usr/local/bin/ffmpeg
   [OK] FFprobe найден: /usr/local/bin/ffprobe

3. Claude CLI:
   [OK] Claude найден: /usr/local/bin/claude
   (или [!!] Claude не найден — установите через npm)

4. Логи:
   Logs dir:  /path/to/PG-Video-Transcriber/Logs
   Log file:  /path/to/PG-Video-Transcriber/Logs/transcriber_2025-02-12.log
   Существует: True/False

5. SpeechBrain модели:
   Path: /path/to/PG-Video-Transcriber/models/speechbrain/spkrec-ecapa-voxceleb
   Существует: True/False
============================================================
```

## Claude CLI — пути на разных ОС

### Windows
- `%APPDATA%\npm\claude.cmd` (npm global default)
- `C:\Program Files\nodejs\claude.cmd`

### macOS/Linux
- `/usr/local/bin/claude` (npm -g, system-wide)
- `~/.npm-global/bin/claude` (custom npm prefix)
- `~/.local/bin/claude` (local install)
- `$NVM_DIR/versions/node/vX.X.X/bin/claude` (nvm)

### Установка Claude CLI:

```bash
npm install -g @anthropic-ai/claude-code
```

После установки перезапустите приложение или проверьте через:

```bash
which claude    # macOS/Linux
where claude    # Windows
```

## Отправка логов при ошибках

Если приложение вылетело или работает некорректно:

1. Зайдите в папку `Logs/` внутри приложения
2. Найдите файл с датой ошибки: `transcriber_YYYY-MM-DD.log`
3. Отправьте файл разработчику

**Путь к логам:**
- Windows: `C:\path\to\PG-Video-Transcriber\Logs\`
- macOS/Linux: `/path/to/PG-Video-Transcriber/Logs/`

Логи **безопасны** — не содержат личных данных, только техническую информацию.
