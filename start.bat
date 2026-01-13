@echo off
REM ============================================
REM Podcast Transcript Downloader - Startup
REM Supports Docker Compose full stack
REM ============================================

cd /d %~dp0

:menu
echo.
echo  ============================================
echo   Podcast Transcript Downloader
echo  ============================================
echo.
echo   [1] Start all services (Docker Compose)
echo   [2] Stop all services
echo   [3] Restart all services
echo   [4] View logs
echo   [5] Show status
echo   [6] Dev mode (local backend + frontend)
echo   [0] Exit
echo.
set /p choice="Select option: "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto restart
if "%choice%"=="4" goto logs
if "%choice%"=="5" goto status
if "%choice%"=="6" goto dev
if "%choice%"=="0" goto end
echo Invalid option. Try again.
goto menu

REM ============================================
:start
REM ============================================
echo.
echo [Starting] Docker Compose services...
docker compose up -d
echo.
echo  Services started!
echo  Frontend:    http://localhost:5173
echo  Backend:     http://localhost:8000
echo  Whisper API: http://localhost:8207
echo  Ollama:      http://localhost:11434
echo.
echo  First time? Run: docker compose exec ollama ollama pull qwen3:8b
echo.
pause
goto menu

REM ============================================
:stop
REM ============================================
echo.
echo [Stopping] Docker Compose services...
docker compose down
echo.
echo  All services stopped.
echo.
pause
goto menu

REM ============================================
:restart
REM ============================================
echo.
echo [Restarting] Docker Compose services...
docker compose down
docker compose up -d
echo.
echo  Services restarted!
echo.
pause
goto menu

REM ============================================
:logs
REM ============================================
echo.
echo [Logs] Following all service logs (Ctrl+C to stop)...
echo.
docker compose logs -f
goto menu

REM ============================================
:status
REM ============================================
echo.
echo [Status] Docker Compose services:
echo.
docker compose ps
echo.
echo [Health Check]
echo.
echo Whisper API:
curl -s http://localhost:8207/health 2>nul && echo  OK || echo  Not running
echo.
echo Backend:
curl -s http://localhost:8000/ 2>nul && echo  OK || echo  Not running
echo.
echo Ollama:
curl -s http://localhost:11434/api/tags 2>nul | findstr "models" >nul && echo  OK || echo  Not running
echo.
pause
goto menu

REM ============================================
:dev
REM ============================================
echo.
echo [Dev Mode] Starting Whisper container + local backend/frontend...
echo.

REM Start Whisper API container only
docker compose up -d whisper-api
echo.
echo Waiting for Whisper API to initialize...
timeout /t 5 /nobreak >nul

REM Set CUDA environment for backend
set CUDA_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cudnn\bin
set CUBLAS_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cublas\bin
set PATH=%CUDA_PATH%;%CUBLAS_PATH%;%PATH%

REM Start Backend
cd /d %~dp0backend
start "Backend" cmd /k "call venv\Scripts\activate && python main.py"

REM Start Frontend
cd /d %~dp0frontend
start "Frontend" cmd /k "npm run dev"

cd /d %~dp0
echo.
echo  Dev mode started!
echo  Frontend:    http://localhost:5173
echo  Backend:     http://localhost:8000
echo  Whisper API: http://localhost:8207
echo.
pause
goto menu

REM ============================================
:end
REM ============================================
echo.
echo Goodbye!
exit /b 0
