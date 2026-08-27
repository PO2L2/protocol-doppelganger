@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" goto :check_venv

echo [1/3] Looking for a supported Python version...
set "PYTHON_EXE="
set "PYTHON_ARG="

py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARG=-3.11"
)

if not defined PYTHON_EXE (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARG=-3.12"
    )
)

if not defined PYTHON_EXE (
    py -3.13 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARG=-3.13"
    )
)

if not defined PYTHON_EXE (
    python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE goto :python_error

echo [2/3] Creating the local environment...
"%PYTHON_EXE%" %PYTHON_ARG% -m venv ".venv"
if errorlevel 1 goto :venv_error

:check_venv
"%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul
if errorlevel 1 goto :unsupported_venv

"%VENV_PYTHON%" -c "import pygame; raise SystemExit(0 if pygame.version.ver == '2.6.1' else 1)" >nul 2>nul
if errorlevel 1 (
    echo [3/3] Installing the lightweight game dependency...
    "%VENV_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --index-url https://pypi.org/simple --timeout 120 -r requirements-player.txt
    if errorlevel 1 goto :dependency_error
)

if /I "%~1"=="--check" (
    echo Launcher check passed.
    exit /b 0
)

echo Starting Protocol: Doppelganger...
"%VENV_PYTHON%" run_game.py
set "GAME_EXIT=%ERRORLEVEL%"
if not "%GAME_EXIT%"=="0" (
    echo.
    echo The game stopped with error code %GAME_EXIT%.
    pause
)
exit /b %GAME_EXIT%

:python_error
echo.
echo ERROR: Python 3.11, 3.12, or 3.13 was not found.
echo Python 3.14 is not supported by pygame 2.6.1 on Windows.
echo Install 64-bit Python 3.11 from https://www.python.org/downloads/
echo Enable "Add python.exe to PATH" and "Python Launcher" during setup.
pause
exit /b 1

:unsupported_venv
echo.
echo ERROR: The existing .venv uses an unsupported Python version.
echo Delete the .venv folder, install Python 3.11, and run this file again.
pause
exit /b 1

:venv_error
echo.
echo ERROR: The local Python environment could not be created.
echo Check the Python installation and free disk space, then try again.
pause
exit /b 1

:dependency_error
echo.
echo ERROR: pygame could not be downloaded from PyPI.
echo Check the internet connection, VPN, proxy, firewall, and antivirus.
echo Then run start_game.bat again.
pause
exit /b 1
