#!/bin/bash
# ============================================================
# PG-Video-Transcriber — macOS/Linux автоустановка и запуск
# ============================================================

set -e
cd "$(dirname "$0")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "  PG-Video-Transcriber — Launcher"
echo "============================================"
echo ""

# 1. Проверяем Python
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        # Проверяем версию 3.10+
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}[ОШИБКА] Python 3.10+ не найден.${NC}"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Установите через Homebrew:"
        echo "  brew install python@3.11"
        echo ""
        echo "Или скачайте: https://www.python.org/downloads/"
    else
        echo "Установите Python 3.10+:"
        echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
        echo "  Fedora: sudo dnf install python3.11"
    fi
    exit 1
fi

echo -e "${GREEN}Python $($PYTHON --version 2>&1 | cut -d' ' -f2) — OK${NC}"

# 2. Создаём виртуальное окружение если нет
if [ ! -f "venv/bin/activate" ]; then
    echo ""
    echo "[1/4] Создание виртуального окружения..."
    $PYTHON -m venv venv
    echo "Виртуальное окружение создано."
fi

# 3. Активируем venv
source venv/bin/activate

# 4. Проверяем, нужна ли установка пакетов
MARKER="venv/.installed"
NEED_INSTALL=0

if [ ! -f "$MARKER" ]; then
    NEED_INSTALL=1
else
    # Проверяем, изменился ли requirements.txt
    if command -v md5sum &>/dev/null; then
        REQHASH=$(md5sum requirements.txt | cut -d' ' -f1)
    else
        REQHASH=$(md5 -q requirements.txt)
    fi
    OLDHASH=$(cat "$MARKER" 2>/dev/null || echo "")
    if [ "$REQHASH" != "$OLDHASH" ]; then
        NEED_INSTALL=1
    fi
fi

if [ "$NEED_INSTALL" == "1" ]; then
    echo ""
    echo "[2/4] Установка PyTorch (CPU)..."
    echo "Это может занять несколько минут при первой установке."
    echo ""

    # macOS: CPU-only torch (нет CUDA)
    pip install --upgrade pip
    pip install torch torchaudio

    echo ""
    echo "[3/4] Установка остальных зависимостей..."

    # nvidia-ml-py не нужен на macOS/Linux без NVIDIA
    # Создаём временный requirements без nvidia-ml-py
    grep -v "nvidia-ml-py" requirements.txt > /tmp/req_macos.txt
    pip install -r /tmp/req_macos.txt
    rm /tmp/req_macos.txt

    # Сохраняем хэш requirements.txt
    if command -v md5sum &>/dev/null; then
        md5sum requirements.txt | cut -d' ' -f1 > "$MARKER"
    else
        md5 -q requirements.txt > "$MARKER"
    fi

    echo ""
    echo -e "${GREEN}Установка завершена!${NC}"
fi

# 5. Запуск приложения
echo ""
echo "[4/4] Запуск PG-Video-Transcriber..."
echo ""
python main.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo -e "${RED}[ОШИБКА] Приложение завершилось с ошибкой (код: $EXIT_CODE).${NC}"
    echo "Логи: transcriber.log"
fi
