@echo off
REM Gideon 2.0 - Windows Launcher Script
REM Avvia Backend e Frontend automaticamente

echo.
echo ╔════════════════════════════════════════╗
echo ║    GIDEON 2.0 - Desktop Assistant    ║
echo ║        Starting Application...        ║
echo ╚════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python non trovato!
    echo    Scarica da https://www.python.org/
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js non trovato!
    echo    Scarica da https://nodejs.org/
    pause
    exit /b 1
)

echo ✅ Python e Node.js trovati
echo.
echo 🚀 Avvio Backend...
start cmd /k "cd backend && C:\OneDrive\OneDrive - Technetpro\Desktop\gideon\.venv\Scripts\activate.ps1 && python main.py"

REM Wait for backend to start
timeout /t 3 /nobreak

echo 🚀 Avvio Frontend...
start cmd /k "cd frontend && npm start"

echo.
echo ✅ Applicazione avviata!
echo    Backend: http://localhost:8001
echo    Frontend: http://localhost:3000
echo.
pause
