@echo off
rem One-click launcher. First run: sets up the Python environment (console visible).
rem After that the server runs in the background without a window and the browser opens.
rem Stop it with the Quit button in the app.
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    title ARIA people counter - setup
    echo First run: installing, this takes a few minutes...
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

start "" ".venv\Scripts\pythonw.exe" app.py --open-browser
exit /b 0
