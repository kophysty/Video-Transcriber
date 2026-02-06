# Video-Transcribe: План реализации

## Стек технологий

| Компонент | Решение | Почему |
|-----------|---------|--------|
| STT-модель | **faster-whisper** (CTranslate2) + Whisper Large V3 Turbo | В 4x быстрее оригинала, VAD встроен |
| VAD | **Silero VAD** (через faster-whisper `vad_filter=True`) | 1.8 MB, убирает галлюцинации на тишине |
| Диаризация | **pyannote.audio** (community-1) | Zero-shot, state-of-the-art, локальный |
| Аудио | **FFmpeg** через `imageio-ffmpeg` (бандлится в pip) | Не нужна системная установка FFmpeg |
| GUI | **CustomTkinter** | Нативный десктоп, ~200 KB, тёмная тема из коробки |
| GPU-детект | **pynvml** | Определяет VRAM, архитектуру, compute capability |
| Язык | **Python 3.11+** | Все ML-библиотеки нативно Python |

## Структура проекта

```
Video-Transcribe/
├── main.py                      # Точка входа
├── requirements.txt
├── config.json                  # Настройки пользователя (создаётся при первом запуске)
├── app/
│   ├── __init__.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py       # Основное окно CustomTkinter
│   │   └── dialogs.py           # Диалог управления моделями
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Оркестратор всего пайплайна
│   │   ├── audio_extractor.py   # FFmpeg: видео → 16kHz mono WAV
│   │   ├── transcriber.py       # faster-whisper обёртка
│   │   ├── diarizer.py          # pyannote.audio обёртка
│   │   └── merger.py            # Склейка транскрипции + диаризации
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── json_exporter.py     # JSON (source of truth)
│   │   ├── srt_exporter.py      # SRT субтитры
│   │   ├── vtt_exporter.py      # WebVTT
│   │   └── txt_exporter.py      # Плоский текст
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gpu_detector.py      # Определение GPU, VRAM, архитектуры
│   │   ├── model_manager.py     # Скачивание/удаление моделей
│   │   └── model_registry.py    # Метаданные моделей (размеры, VRAM)
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Загрузка/сохранение config.json
│       ├── ffmpeg_finder.py     # Поиск FFmpeg бинарника
│       └── formats.py           # Хелперы форматирования таймстампов
├── models/                      # Скачанные модели (в .gitignore)
│   ├── whisper/
│   └── pyannote/
└── tests/
    ├── test_transcriber.py
    ├── test_merger.py
    ├── test_exporters.py
    └── test_gpu_detector.py
```

## Пайплайн обработки

```
Видео/аудио файл
  → FFmpeg: извлечение аудио (16kHz mono WAV, шумоподавление)
    → Silero VAD: разметка речевых сегментов
      → faster-whisper: транскрибация (word-level timestamps)
        → [опционально] pyannote: диаризация спикеров
          → Merger: склейка транскрипции + спикеров
            → Экспорт: JSON + SRT + VTT + TXT
```

## Адаптация под железо

### Автодетект GPU и выбор модели

```
Определить VRAM + архитектуру → рекомендовать модель и compute_type

≥12 GB, Ampere+ (RTX 30xx/40xx) → large-v3-turbo, float16
≥12 GB, Pascal  (GTX 10xx)      → large-v3-turbo, int8
≥6 GB,  Ampere+                  → large-v3-turbo, float16
≥6 GB,  Pascal                   → medium, float32 или int8
<6 GB                            → small, int8
CPU only                         → small или tiny, int8
```

### Управление VRAM при диаризации

```
VRAM ≥ 16 GB  → Whisper + pyannote на GPU одновременно
VRAM 11-16 GB → Последовательно: выгрузить Whisper, загрузить pyannote
VRAM < 11 GB  → Whisper на GPU, pyannote на CPU
```

## Решения по дизайну

- **Форматы вывода**: генерируем все 4 формата (JSON + SRT + VTT + TXT) каждый раз — это почти бесплатно по времени
- **Язык интерфейса**: русский

## GUI — макет основного окна

```
+--------------------------------------------------+
|  Video Transcribe                          [_][X] |
+--------------------------------------------------+
|                                                    |
|  ВХОДНОЙ ФАЙЛ                                      |
|  [ /path/to/video.mp4                  ] [Обзор]   |
|                                                    |
|  ПАПКА ДЛЯ РЕЗУЛЬТАТОВ                             |
|  [ D:\Transcripts\                     ] [Обзор]   |
|                                                    |
|  ─── Настройки ───                                 |
|  Модель:   [ large-v3-turbo  ▾]  [Управление]     |
|  Язык:     [ Авто            ▾]                    |
|  ☑ Диаризация спикеров                             |
|                                                    |
|  GPU: NVIDIA RTX 3090 (24 GB) · float16            |
|                                                    |
|  ─── Прогресс ───                                  |
|  [████████████░░░░░░░░] 65%                        |
|  Транскрибация... сегмент 142/218                  |
|                                                    |
|          [ Начать транскрибацию ]                   |
+--------------------------------------------------+
```

**Диалог "Управление моделями":**
- Таблица скачанных моделей (имя, размер, VRAM) + кнопка удаления
- Таблица доступных для скачивания + кнопка скачивания с прогрессом
- Секция pyannote: статус + кнопка скачивания (требует HF токен)

## Управление моделями

- Модели скачиваются в `./models/whisper/<model-name>/` через `huggingface_hub.snapshot_download()`
- При загрузке в faster-whisper передаётся локальный путь + `local_files_only=True`
- pyannote модель скачивается отдельно, требует бесплатный HuggingFace токен
- В UI показываем какие модели скачаны, какие доступны, сколько места занимают

## Зависимости (requirements.txt)

```
# ML
faster-whisper>=1.1.0
pyannote.audio>=3.3.0
torch>=2.1.0        # устанавливать с --index-url .../cu124
torchaudio>=2.1.0

# Аудио
imageio-ffmpeg>=0.5.1

# GUI
customtkinter>=5.2.0

# GPU
pynvml>=11.5.0

# Модели
huggingface-hub>=0.20.0
```

## Порядок реализации

### Фаза 1: Фундамент
1. **Scaffolding** — создать структуру директорий, venv, requirements.txt, установить зависимости
2. **config.py** — AppConfig датакласс, загрузка/сохранение config.json
3. **ffmpeg_finder.py** — поиск FFmpeg (imageio-ffmpeg → системный PATH → env var)
4. **gpu_detector.py** — определение GPU через pynvml, классификация архитектуры, рекомендация модели

### Фаза 2: Core Pipeline
5. **audio_extractor.py** — FFmpeg: извлечение аудио с шумоподавлением, парсинг прогресса
6. **transcriber.py** — обёртка faster-whisper с VAD, поддержка языков, прогресс через генератор сегментов
7. **diarizer.py** — обёртка pyannote, управление VRAM (выгрузка Whisper перед загрузкой pyannote)
8. **merger.py** — алгоритм назначения спикеров через overlap интервалов
9. **pipeline.py** — оркестратор: связывает все компоненты, агрегирует прогресс, обработка ошибок

### Фаза 3: Экспорт
10. **Все 4 экспортера** — JSON (с метаданными), SRT, VTT, TXT. Без спикеров если диаризация выключена

### Фаза 4: Модели
11. **model_registry.py** — статические метаданные моделей (repo, VRAM, описание)
12. **model_manager.py** — скачивание через huggingface_hub, список скачанных, удаление, подсчёт размера

### Фаза 5: GUI
13. **main_window.py** — основной интерфейс, файл-пикеры, дропдауны, прогресс-бар
14. **dialogs.py** — менеджер моделей, ввод HF токена для pyannote
15. **Threading** — пайплайн в фоновом потоке, обновление UI через `after()`, кнопка отмены
16. **Полировка** — запоминание настроек, фильтры типов файлов, обработка ошибок (CUDA OOM → предложить модель поменьше)

## Ключевые технические решения

### FFmpeg команда извлечения аудио
```bash
ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 \
  -af "highpass=f=100,lowpass=f=8000,afftdn=nf=-20" output.wav
```

### faster-whisper параметры транскрибации
```python
model = WhisperModel(model_path, device="cuda", compute_type="float16", local_files_only=True)
segments, info = model.transcribe(audio_path, language=None, beam_size=5,
    word_timestamps=True, vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200})
```

### Отмена обработки
`threading.Event` — проверяется между этапами пайплайна и между сегментами при транскрибации. По нажатию "Отмена" — set event, cleanup temp файлов.

## Верификация

### Автотесты
- `test_gpu_detector.py` — классификация архитектур, рекомендации моделей для разных VRAM
- `test_merger.py` — назначение спикеров на известных overlap-ах, edge cases
- `test_exporters.py` — формат таймстампов SRT/VTT, UTF-8, отсутствие спикеров

### Ручная проверка
- [ ] Чистая установка из venv на Windows
- [ ] Первый запуск: предложение скачать модель
- [ ] Скачать large-v3-turbo, проверить что легла в `./models/whisper/`
- [ ] Транскрибация 5-минутного видео на русском → проверить SRT в видеоплеере
- [ ] Транскрибация на английском → проверить auto-detect языка
- [ ] Диаризация на 2 спикерах → проверить метки спикеров
- [ ] Тест на 2-часовой записи → убедиться что нет OOM
- [ ] Отмена посреди обработки → нет зависших temp файлов
- [ ] Закрыть/открыть приложение → настройки сохранились
