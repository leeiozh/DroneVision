@echo off
rem Self-check: Python packages, ffmpeg, dji-log, coastline, data folder. Downloads what is missing.
setlocal EnableExtensions
title DroneVision check
pushd "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" echo  Start DroneVision.bat first: the environment is not prepared yet.
if exist ".venv\Scripts\python.exe" ".venv\Scripts\python.exe" -m dronevision setup
echo.
pause
popd
