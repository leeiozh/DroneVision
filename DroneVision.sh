#!/usr/bin/env bash
# Local Linux launcher.  Keep this terminal open while DroneVision is in use.
set -u
cd -- "$(dirname -- "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "DroneVision: не найден Python 3.10 или новее."
  echo "Ubuntu/Mint: sudo apt install python3 python3-venv python3-tk ffmpeg"
  read -r -p "Нажмите Enter, чтобы закрыть окно..." _
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "DroneVision: нужны ffmpeg и ffprobe."
  echo "Ubuntu/Mint: sudo apt install ffmpeg"
  read -r -p "Нажмите Enter, чтобы закрыть окно..." _
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Первый запуск: создаю отдельное окружение DroneVision..."
  if ! python3 -m venv --system-site-packages .venv; then
    echo "Не удалось создать .venv. Установите пакет python3-venv."
    read -r -p "Нажмите Enter, чтобы закрыть окно..." _
    exit 1
  fi
fi
if ! .venv/bin/python dronevision/bootstrap.py; then
  echo "Не удалось подготовить библиотеки DroneVision. Проверьте интернет и запустите снова."
  read -r -p "Нажмите Enter, чтобы закрыть окно..." _
  exit 1
fi

exec .venv/bin/python -m dronevision web --open
