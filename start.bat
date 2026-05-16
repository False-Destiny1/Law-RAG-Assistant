@echo off
chcp 65001 >nul 2>&1
title Law Assistant - FastAPI

echo.
echo  ========================================
echo    Law Assistant - Starting...
echo  ========================================
echo.

set "PYTHON_EXE=E:\anaconda3\envs\pytorch\python.exe"
set "PROJECT_DIR=%~dp0"
set "HOST=127.0.0.1"
set "PORT=8080"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please check Anaconda pytorch environment
    pause
    exit /b 1
)

"%PYTHON_EXE%" --version
echo.

cd /d "%PROJECT_DIR%"

echo Starting FastAPI server...
echo URL: http://%HOST%:%PORT%
echo Press Ctrl+C to stop
echo.
echo ========================================
echo.

start "" /min cmd /c "timeout /t 3 >nul & start http://%HOST%:%PORT%/login"

"%PYTHON_EXE%" -m uvicorn app:app --host %HOST% --port %PORT%

pause
