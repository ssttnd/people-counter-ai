@echo off
rem One-click launcher: sets up Python environment on first run, starts the
rem server and opens the browser at http://127.0.0.1:5000
cd /d "%~dp0"
title ARIA people counter

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating Python environment, this takes a few minutes...
    python -m venv .venv || (
        echo Python 3.10+ is required: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (
        echo Installing packages failed.
        pause
        exit /b 1
    )
)

echo Starting... the browser will open automatically. Close this window to stop.
".venv\Scripts\python.exe" app.py --open-browser
pause
