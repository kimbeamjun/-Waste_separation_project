@echo off
title step5 - AI Server

cd /d "%~dp0"

echo  Clearing port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do taskkill /f /pid %%a >nul 2>&1

call .venv\Scripts\activate.bat
echo  Starting AI server...
python step5_api_server.py
pause
