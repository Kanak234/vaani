@echo off
setlocal

:: Find the repository root (where this script lives)
set "HERE=%~dp0"

:: Check for .venv
if not exist "%HERE%.venv\Scripts\python.exe" (
    echo Virtual environment not found. Creating...
    python -m venv "%HERE%.venv"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        exit /b 1
    )
    echo Installing Vaani...
    "%HERE%.venv\Scripts\pip.exe" install -e "%HERE%[stt,translate]"
)

:: Launch Vaani
"%HERE%.venv\Scripts\python.exe" -m vaani.cli %*
