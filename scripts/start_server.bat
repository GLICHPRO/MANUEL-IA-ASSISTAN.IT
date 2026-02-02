@echo off
echo Starting GIDEON Backend Server...
cd /d "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\backend"
call "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0\.venv\Scripts\activate.bat"
python -m uvicorn main:app --host 127.0.0.1 --port 8001
pause
