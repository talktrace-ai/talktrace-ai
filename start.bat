@echo off
REM TalkTrace AI -- install + launch helper for Windows.
REM
REM Usage (from this folder):
REM   start.bat               install (if needed) and start the app
REM   start.bat /reinstall    force-recreate the virtual environment
REM   start.bat /nowindow     start the app headless (no desktop window)
REM
REM For development with hot-reload, use dev.bat instead.

if not defined TT_MINIMIZED (
    set "TT_MINIMIZED=1"
    start /min "" "%~f0" %*
    exit /b 0
)

setlocal

set "REINSTALL="
set "NOWINDOW="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="/reinstall" goto set_reinstall
if /I "%~1"=="-reinstall" goto set_reinstall
if /I "%~1"=="/nowindow"  goto set_nowindow
if /I "%~1"=="-nowindow"  goto set_nowindow
echo Unknown argument: %~1
goto fail
:set_reinstall
set "REINSTALL=1"
shift
goto parse_args
:set_nowindow
set "NOWINDOW=1"
shift
goto parse_args
:args_done

set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"
if errorlevel 1 goto fail

echo [TalkTrace] Project root: %PROJECT_ROOT%

REM --- 1. Locate Python ---------------------------------------------------
set "PY_CMD="
py -3 --version >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"

if not defined PY_CMD (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    python3 --version >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python3"
)

if not defined PY_CMD (
    echo [TalkTrace] Python 3 not found. Install from https://www.python.org/downloads/ and re-run.
    goto fail
)
echo [TalkTrace] Using Python launcher: %PY_CMD%

REM --- 2. Virtual environment ---------------------------------------------
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not defined REINSTALL goto skip_reinstall
if not exist "%VENV_DIR%"  goto skip_reinstall
echo [TalkTrace] Removing existing venv ^(/reinstall^)...
rmdir /s /q "%VENV_DIR%"
:skip_reinstall

if not exist "%VENV_PY%" (
    echo [TalkTrace] Creating virtual environment in .venv ...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 goto venv_fail
)
goto venv_ok
:venv_fail
echo [TalkTrace] venv creation failed
goto fail
:venv_ok

REM --- 3. Install dependencies --------------------------------------------
set "REQ_FILE=%PROJECT_ROOT%\requirements.txt"
if not exist "%REQ_FILE%" (
    echo [TalkTrace] requirements.txt not found at %REQ_FILE%
    goto fail
)

set "STAMP_FILE=%VENV_DIR%\.requirements.sha256"
set "CURRENT_HASH="
for /f "skip=1 tokens=* usebackq" %%H in (`certutil -hashfile "%REQ_FILE%" SHA256 ^| findstr /v ":"`) do (
    if not defined CURRENT_HASH set "CURRENT_HASH=%%H"
)
set "CURRENT_HASH=%CURRENT_HASH: =%"

set "STORED_HASH="
if exist "%STAMP_FILE%" set /p STORED_HASH=<"%STAMP_FILE%"

if /I "%STORED_HASH%"=="%CURRENT_HASH%" goto deps_ok

echo [TalkTrace] Installing/upgrading dependencies ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto pip_fail
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto pip_fail
>"%STAMP_FILE%" echo %CURRENT_HASH%
goto deps_ok
:pip_fail
echo [TalkTrace] pip install failed
goto fail
:deps_ok

REM --- 4. Launch the app --------------------------------------------------
echo [TalkTrace] Starting Shiny app ... press Ctrl+C to stop.

if defined NOWINDOW goto launch_nowindow
"%VENV_PY%" -c "from talktrace_ai.app import main; main()"
goto launch_done
:launch_nowindow
"%VENV_PY%" -c "from talktrace_ai.app import main; main(open_window=False)"
:launch_done
if errorlevel 1 goto fail

endlocal
exit /b 0

:fail
echo.
echo [TalkTrace] Startup failed. Press any key to close this window.
pause >nul
endlocal
exit /b 1
