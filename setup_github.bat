@echo off
echo === GitHub Setup Script ===

REM Add gh to PATH
set PATH=%PATH%;C:\Program Files\GitHub CLI

REM Check gh auth
echo.
echo Checking GitHub authentication...
gh auth status

echo.
echo Creating repository and pushing...
cd /d "C:\OneDrive\OneDrive - Technetpro\Desktop\gideon2.0"

REM Create repo
gh repo create gideon-ai --public --source=. --remote=origin --push

echo.
echo === Done! ===
pause
