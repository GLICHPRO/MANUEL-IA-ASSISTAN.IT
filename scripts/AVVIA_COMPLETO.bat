@echo off
title Gideon 2.0 - Avvio Completo
color 0B

echo ========================================
echo   GIDEON 2.0 - Avvio Completo
echo ========================================
echo.
echo Avvio Backend + Frontend...
echo.

cd /d "%~dp0"

REM Avvia Backend in nuova finestra
start "Gideon Backend" cmd /k "cd backend && call "%~dp0.venv\Scripts\activate.bat" && python -m uvicorn main:app --host 127.0.0.1 --port 8001"

REM Aspetta 3 secondi
timeout /t 3 /nobreak >nul

REM Avvia Frontend in nuova finestra
start "Gideon Frontend" cmd /k "cd frontend && call "%~dp0.venv\Scripts\activate.bat" && python server.py"

REM Aspetta 2 secondi
timeout /t 2 /nobreak >nul

REM Apri browser
start http://localhost:3000/index.html

echo.
echo ========================================
echo   ✓ Gideon 2.0 Avviato!
echo ========================================
echo.
echo  Backend: http://127.0.0.1:8001
echo  Frontend: http://localhost:3000
echo  Dashboard: http://localhost:3000/index.html
echo.
echo Premi un tasto per chiudere questa finestra...
pause >nul
