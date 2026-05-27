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
set HOST=127.0.0.1
set PORT=8080
set "REDIS_EXE=E:\Redis\redis-server.exe"
set "NEO4J_HOME=E:\neo4j-chs-community-5.26.2-windows"

if not exist %PYTHON_EXE% (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please check Anaconda pytorch environment
    pause
    exit /b 1
)

%PYTHON_EXE% --version
echo.

cd /d "%PROJECT_DIR%"

:: Auto-start Redis
echo Checking Redis...
netstat -an | findstr "6379" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo Redis is already running on port 6379
) else (
    if not exist %REDIS_EXE% (
        echo [WARN] Redis not found: %REDIS_EXE%
        echo Redis cache will be unavailable, falling back to local cache
    ) else (
        echo Starting Redis...
        start "" /min %REDIS_EXE%
        timeout /t 2 >nul
        netstat -an | findstr "6379" | findstr "LISTENING" >nul 2>&1
        if %errorlevel%==0 (
            echo Redis started successfully
        ) else (
            echo [WARN] Redis failed to start, falling back to local cache
        )
    )
)
echo.

:: Auto-start Neo4j
echo Checking Neo4j...
netstat -an | findstr "7687" | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo Neo4j is already running on port 7687
) else (
    if not exist "%NEO4J_HOME%\bin\neo4j.bat" (
        echo [WARN] Neo4j not found: %NEO4J_HOME%
        echo Knowledge graph will be unavailable, falling back to vector+BM25
    ) else (
        echo Starting Neo4j...
        start "Neo4j" cmd /c ""%NEO4J_HOME%\bin\neo4j.bat" console"
        echo Waiting for Neo4j to start...
        timeout /t 15 >nul
        netstat -an | findstr "7687" | findstr "LISTENING" >nul 2>&1
        if %errorlevel%==0 (
            echo Neo4j started successfully
        ) else (
            echo [WARN] Neo4j failed to start, falling back to vector+BM25
        )
    )
)
echo.

echo Starting FastAPI server...
echo URL: http://%HOST%:%PORT%
echo Press Ctrl+C to stop
echo.
echo ========================================
echo.

start "" /min cmd /c "timeout /t 3 >nul & start http://%HOST%:%PORT%/login"

%PYTHON_EXE% -m uvicorn app:app --host %HOST% --port %PORT%

pause
