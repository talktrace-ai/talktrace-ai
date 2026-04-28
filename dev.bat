@echo off
REM TalkTrace AI -- developer launcher (hot-reload, browser).
REM
REM Run from the project folder:
REM   dev.bat
REM
REM Edits any .py file under talktrace_ai/ -> server auto-restarts.
REM Press Ctrl+C to stop.

setlocal

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"

set "VENV_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [TalkTrace dev] Virtual environment not found at "%VENV_PY%".
    echo [TalkTrace dev] Run start.bat once to create the venv, then re-run dev.bat.
    pause
    endlocal
    exit /b 1
)

echo [TalkTrace dev] Project root: %PROJECT_ROOT%
echo [TalkTrace dev] Hot-reload enabled. Edit any .py file under talktrace_ai\ to trigger a restart.
echo [TalkTrace dev] Press Ctrl+C to stop.
echo.

"%VENV_PY%" -m shiny run --reload --launch-browser --reload-dir "%PROJECT_ROOT%\talktrace_ai" --port 8000 talktrace_ai.app:app

endlocal
