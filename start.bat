@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title Law Assistant - FastAPI

echo.
echo  ========================================
echo    Law Assistant - Starting...
echo  ========================================
echo.

set "PROJECT_DIR=%~dp0"
set HOST=127.0.0.1
set PORT=8080

:: Resolve Python — prefer conda env, fallback to PATH
if defined PYTHON_EXE goto :python_found
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "delims=" %%i in ('where python') do set "PYTHON_EXE=%%i" & goto :python_found
)
echo [ERROR] Python not found in PATH. Please activate your conda environment or set PYTHON_EXE.
pause
exit /b 1
:python_found

:: Resolve Redis — prefer REDIS_EXE env var, fallback to PATH
if not defined REDIS_EXE (
    where redis-server >nul 2>&1
    if %errorlevel%==0 (
        for /f "delims=" %%i in ('where redis-server') do set "REDIS_EXE=%%i"
    )
)

:: Resolve Neo4j — prefer NEO4J_HOME env var, then common install paths
if not defined NEO4J_HOME (
    if exist "C:\neo4j" set "NEO4J_HOME=C:\neo4j"
    if not defined NEO4J_HOME if exist "E:\neo4j-chs-community-5.26.2-windows" set "NEO4J_HOME=E:\neo4j-chs-community-5.26.2-windows"
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
set "NEO4J_NEED_WAIT=0"
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
        set "NEO4J_NEED_WAIT=1"
        echo Waiting for Neo4j to start...
        set NEO4J_WAIT=0
    )
)
:wait_neo4j
if "!NEO4J_NEED_WAIT!"=="1" (
    timeout /t 5 >nul
    set /a NEO4J_WAIT+=5
    netstat -an | findstr "7687" | findstr "LISTENING" >nul 2>&1
    if !errorlevel!==0 (
        echo Neo4j started successfully
        set "NEO4J_NEED_WAIT=0"
    ) else (
        if !NEO4J_WAIT! LSS 60 goto wait_neo4j
        echo [WARN] Neo4j failed to start within 60 seconds
        set "NEO4J_NEED_WAIT=0"
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
