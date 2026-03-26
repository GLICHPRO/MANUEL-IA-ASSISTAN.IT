@echo off
REM ═══════════════════════════════════════════════════════════════════════════════
REM  🤖 GIDEON 4.0 - Ultimate Unified AI System
REM  Single Startup Point - ONE script to start everything!
REM ═══════════════════════════════════════════════════════════════════════════════
REM
REM  G.I.D.E.O.N. = Generative Intelligence for Dynamic Executive Operations Network
REM
REM  This script starts:
REM  - Backend API (FastAPI) on port 8001
REM  - Frontend Server on port 3000
REM
REM  GIDEON 4.0 Unified Endpoints:
REM  - Main API:     http://localhost:8001/api/gideon4
REM  - Health:       http://localhost:8001/api/gideon4/health
REM  - Status:       http://localhost:8001/api/gideon4/status
REM  - Frontend:     http://localhost:3000/gideon_unified.html
REM
REM ═══════════════════════════════════════════════════════════════════════════════

title GIDEON 4.0 - Ultimate Unified AI

echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo  🤖 GIDEON 4.0 - Ultimate Unified AI System
echo  G.I.D.E.O.N. = Generative Intelligence for Dynamic Executive Operations Network
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo  ╔═══════════════════════════════════════════════════════════════════════════╗
echo  ║                         VERSION HISTORY                                    ║
echo  ╠═══════════════════════════════════════════════════════════════════════════╣
echo  ║  v1.0 - Basic Assistant       : Simple Q^&A, basic commands                ║
echo  ║  v2.0 - Brain Integration     : NLP, Memory, Reasoning, AI Providers      ║
echo  ║  v3.0 - Cognitive Module      : Predictor, Analyzer, Simulator            ║
echo  ║  v4.0 - Ultimate Unified      : All versions + Executive AI + Autonomy    ║
echo  ╚═══════════════════════════════════════════════════════════════════════════╝
echo.

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if .venv exists
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found!
    echo    Please run: python -m venv .venv
    echo    And then: pip install -r backend\requirements.txt
    pause
    exit /b 1
)

echo [1/4] 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

echo [2/4] 🧹 Stopping existing Python processes...
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo [3/4] 🚀 Starting GIDEON 4.0 Backend (port 8001)...
start "GIDEON 4.0 Backend" cmd /k "cd /d "%SCRIPT_DIR%" && .venv\Scripts\python.exe backend\main.py"

echo [4/4] 🌐 Starting Frontend Server (port 3000)...
timeout /t 3 /nobreak >nul
start "GIDEON 4.0 Frontend" cmd /k "cd /d "%SCRIPT_DIR%" && .venv\Scripts\python.exe frontend\server.py"

echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo.
echo  ✅ GIDEON 4.0 is starting!
echo.
echo  📊 Endpoints:
echo     * Main API:     http://localhost:8001/api/gideon4
echo     * Health:       http://localhost:8001/api/gideon4/health
echo     * Status:       http://localhost:8001/api/gideon4/status
echo     * Process:      http://localhost:8001/api/gideon4/process
echo     * Analyze:      http://localhost:8001/api/gideon4/analyze
echo     * Predict:      http://localhost:8001/api/gideon4/predict
echo     * Execute:      http://localhost:8001/api/gideon4/execute
echo     * WebSocket:    ws://localhost:8001/api/gideon4/ws
echo.
echo  🌐 Frontend:      http://localhost:3000/gideon_unified.html
echo.
echo ═══════════════════════════════════════════════════════════════════════════════
echo.

REM Wait and then open browser
timeout /t 5 /nobreak >nul
start http://localhost:3000/gideon_unified.html

echo Press any key to open the browser again, or close this window.
pause >nul
start http://localhost:3000/gideon_unified.html
