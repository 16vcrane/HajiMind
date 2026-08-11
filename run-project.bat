@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-project.ps1"
if errorlevel 1 (
    echo.
    echo HajiMind startup failed. Review the message above and press any key to close.
    pause >nul
)
