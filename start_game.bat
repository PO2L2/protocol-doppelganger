@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python environment...
    python -m venv .venv
)

".venv\Scripts\python.exe" -c "import pygame, numpy, torch; assert torch.__version__.startswith('2.13.0+cu130')" >nul 2>nul
if errorlevel 1 (
    echo Installing game and AI dependencies...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

".venv\Scripts\python.exe" run_game.py
if errorlevel 1 pause
