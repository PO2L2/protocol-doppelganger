@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" goto :missing_environment

echo [1/5] Installing the build tool...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check -r requirements-build.txt
if errorlevel 1 goto :build_error

echo [2/5] Running tests...
"%VENV_PYTHON%" -m unittest discover -s tests -q
if errorlevel 1 goto :test_error

echo [3/5] Building Protocol-Doppelganger.exe...
"%VENV_PYTHON%" -m PyInstaller --noconfirm --clean --onefile --windowed --name "Protocol-Doppelganger" --exclude-module torch --exclude-module numpy run_game.py
if errorlevel 1 goto :build_error

echo [4/5] Preparing the portable folder...
if not exist "dist\ДАННЫЕ_ДЛЯ_ОТПРАВКИ" mkdir "dist\ДАННЫЕ_ДЛЯ_ОТПРАВКИ"
copy /Y "ИНСТРУКЦИЯ_ИГРОКУ.txt" "dist\ИНСТРУКЦИЯ_ИГРОКУ.txt" >nul
copy /Y "ИНСТРУКЦИЯ_ИГРОКУ.txt" "dist\ДАННЫЕ_ДЛЯ_ОТПРАВКИ\ЧТО_НУЖНО_ОТПРАВИТЬ.txt" >nul

echo [5/5] Creating the archive for players...
powershell -NoProfile -Command "Compress-Archive -LiteralPath 'dist\Protocol-Doppelganger.exe','dist\ИНСТРУКЦИЯ_ИГРОКУ.txt','dist\ДАННЫЕ_ДЛЯ_ОТПРАВКИ' -DestinationPath 'dist\Protocol-Doppelganger-Windows.zip' -Force"
if errorlevel 1 goto :build_error

echo.
echo Build completed: dist\Protocol-Doppelganger.exe
echo Player archive: dist\Protocol-Doppelganger-Windows.zip
exit /b 0

:missing_environment
echo ERROR: .venv was not found. Run start_game.bat first.
exit /b 1

:test_error
echo ERROR: Tests failed. The executable was not rebuilt.
exit /b 1

:build_error
echo ERROR: The Windows build failed.
exit /b 1
