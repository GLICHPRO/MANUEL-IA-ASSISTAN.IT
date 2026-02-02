@echo off
title GIDEON 3.0 - Sistema Completo
color 0A

echo ========================================
echo   GIDEON 3.0 - Avvio Sistema Completo
echo ========================================
echo.

set PROJECT_ROOT=%~dp0
set VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe
set BACKEND_DIR=%PROJECT_ROOT%backend
set FRONTEND_DIR=%PROJECT_ROOT%frontend

REM Verifica Python
if not exist "%VENV_PYTHON%" (
    echo [ERRORE] Python venv non trovato!
    pause
    exit /b 1
)

echo [1/4] Chiusura processi precedenti...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo [2/4] Avvio Backend (porta 8001)...
start "GIDEON Backend" /min cmd /c "cd /d %BACKEND_DIR% && %VENV_PYTHON% -m uvicorn main:app --host 127.0.0.1 --port 8001"

echo [3/4] Avvio Frontend (porta 3000)...
start "GIDEON Frontend" /min cmd /c "cd /d %FRONTEND_DIR% && %VENV_PYTHON% -m http.server 3000"

echo [4/4] Attendere avvio server...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo     GIDEON 3.0 PRONTO!
echo ========================================
echo.
echo Backend API:  http://127.0.0.1:8001
echo Health:       http://127.0.0.1:8001/health
echo Dev API:      http://127.0.0.1:8001/api/dev/status
echo.
echo Frontend:     http://127.0.0.1:3000
echo Chat:         http://127.0.0.1:3000/chat.html
echo.
echo Per fermare: Chiudi le finestre cmd minimizzate
echo ========================================
echo.

REM Apri browser
start http://127.0.0.1:3000/chat.html

pause
