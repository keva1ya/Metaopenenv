@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing virtualenv interpreter at .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo Starting Loan Approval app...
".venv\Scripts\python.exe" app.py

echo.
echo App stopped.
pause
