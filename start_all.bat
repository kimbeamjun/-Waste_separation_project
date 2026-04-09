@echo off
title Waste Classifier - Launcher

cls
echo.
echo  ================================================
echo   Waste Classifier AI - Service Launcher
echo  ================================================
echo.

set PROJECT_DIR=%~dp0
set VENV_ACTIVATE=%PROJECT_DIR%.venv\Scripts\activate.bat

if not exist "%VENV_ACTIVATE%" (
    echo  [ERROR] .venv not found
    echo  Run: python -m venv .venv
    pause
    exit /b 1
)

echo  Select services to start:
echo.
echo   [1] All          - Server + Webcam + Arduino
echo   [2] Server only  - step5 AI server
echo   [3] Server + Cam - step5 + step3
echo   [4] Dashboard    - step5 + step6
echo   [5] Dev mode     - step5 + step6 + Gradio
echo.
set /p CHOICE="  Enter number (1-5, default=1): "
if "%CHOICE%"=="" set CHOICE=1

set RUN_ARDUINO=0
if "%CHOICE%"=="1" (
    echo.
    set /p ARD="  Arduino connected? [Y/N] (default=Y): "
    if /i "%ARD%"=="N" (set RUN_ARDUINO=0) else (set RUN_ARDUINO=1)
)

echo.
echo  Starting services...
echo.

echo  [step5] AI server ^> localhost:8000
start "step5-server" cmd /k "cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python step5_api_server.py"
echo  Waiting 3 seconds...
timeout /t 3 /nobreak >nul

if "%CHOICE%"=="2" goto done

if "%CHOICE%"=="1" (
    echo  [step3] Webcam client...
    start "step3-webcam" cmd /k "cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python step3_webcam.py"
    timeout /t 2 /nobreak >nul
    if "%RUN_ARDUINO%"=="1" (
        echo  [step8] Arduino LED controller...
        start "step8-arduino" cmd /k "cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python step8_arduino.py"
    )
    goto done
)

if "%CHOICE%"=="3" (
    echo  [step3] Webcam client...
    start "step3-webcam" cmd /k "cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python step3_webcam.py"
    goto done
)

if "%CHOICE%"=="4" (
    echo  [step6] Dashboard ^> localhost:8080
    start "step6-dashboard" cmd /k "cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python step6_dashboard.py"
    goto done
)

if "%CHOICE%"=="5" (
    echo  [step6] Dashboard ^> localhost:8080
    start "step6-dashboard" cmd /k "cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python step6_dashboard.py"
    timeout /t 1 /nobreak >nul
    echo  [step4] Gradio demo...
    start "step4-gradio" cmd /k "cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python step4_gradio_demo.py"
    goto done
)

:done
echo.
echo  ================================================
echo   Done! Services running:
echo   - AI Server : http://localhost:8000
echo   - API Docs  : http://localhost:8000/docs
echo   To stop     : run stop_all.bat
echo  ================================================
echo.
set /p BROWSER="  Open browser? [Y/N] (default=Y): "
if /i not "%BROWSER%"=="N" start http://localhost:8000
echo.
echo  This window can be closed.
pause
