@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo ERROR: The local game environment does not exist.
    echo Run start_game.bat once before installing the AI dependencies.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: The existing .venv uses an unsupported Python version.
    echo Delete the .venv folder, install Python 3.10 or 3.11, and run start_game.bat again.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import numpy, torch; assert torch.__version__.startswith('2.13.0+cu130')" >nul 2>nul
if not errorlevel 1 (
    echo The full AI dependencies are already installed.
    pause
    exit /b 0
)

echo Installing NumPy and PyTorch with CUDA support...
echo This is a large optional download and may take a while.
"%VENV_PYTHON%" -m pip install --disable-pip-version-check --timeout 180 -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: The AI dependencies could not be installed.
    echo Check the internet connection and free disk space, then try again.
    pause
    exit /b 1
)

echo.
echo AI dependencies installed successfully.
pause
exit /b 0
