@echo off
title Waste Classifier - Stop All

echo.
echo  Stopping all services...
echo.

taskkill /f /fi "WINDOWTITLE eq step5-server*"    >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq step3-webcam*"    >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq step8-arduino*"   >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq step6-dashboard*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq step4-gradio*"    >nul 2>&1

echo  Clearing ports 8000 and 8080...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080 "') do taskkill /f /pid %%a >nul 2>&1

echo.
echo  All services stopped.
echo.
timeout /t 2 /nobreak >nul
