@echo off
echo ========================================
echo   GIDEON 2.0 - Backend Startup
echo ========================================
echo.

cd /d "%~dp0backend"
call "%~dp0.venv\Scripts\activate.bat"

echo Starting Gideon 2.0 Backend on port 8001...
echo.
echo Press CTRL+C to stop the server
echo.

uvicorn main:app --host 0.0.0.0 --port 8001 --log-level info

pause
