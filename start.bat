@echo off
REM ============================================
REM Podcast Transcript Downloader - Startup
REM New Architecture: Whisper API + Ollama
REM ============================================

echo.
echo  ============================================
echo   Podcast Transcript Downloader
echo   Starting services...
echo  ============================================
echo.

cd /d %~dp0

REM ============================================
REM Check Docker
REM ============================================
echo [1/4] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo  [!] Docker not found!
    echo      Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    echo.
    echo      The Whisper API runs in a Docker container with GPU support.
    echo.
    pause
    exit /b 1
)
echo      Docker found.

REM ============================================
REM Check Ollama (optional but recommended)
REM ============================================
echo [2/4] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo  [!] Ollama not running or not installed.
    echo      For LLM text polishing, install Ollama:
    echo      https://ollama.com/download/windows
    echo.
    echo      Then pull a model: ollama pull qwen3:8b
    echo      See OLLAMA_SETUP.md for details.
    echo.
    echo      Transcription will still work without LLM polishing.
    echo.
) else (
    echo      Ollama is running.
)

REM ============================================
REM Start Whisper API Container
REM ============================================
echo [3/4] Starting Whisper API container...

REM Check if container is already running
docker ps --filter "name=whisper-api" --format "{{.Names}}" | findstr "whisper-api" >nul 2>&1
if not errorlevel 1 (
    echo      Whisper API container already running.
) else (
    REM Build and start container
    echo      Building and starting Whisper API...
    docker-compose up -d whisper-api
    if errorlevel 1 (
        echo  [!] Failed to start Whisper API container.
        echo      Run 'docker-compose logs whisper-api' for details.
        pause
        exit /b 1
    )
    echo      Whisper API container started.

    REM Wait for container to be ready
    echo      Waiting for Whisper API to initialize...
    timeout /t 10 /nobreak >nul

    REM Check health
    curl -s http://localhost:8207/health >nul 2>&1
    if errorlevel 1 (
        echo      Container still initializing (model loading)...
        echo      This may take 1-2 minutes on first run.
    ) else (
        echo      Whisper API is ready.
    )
)

REM ============================================
REM Start Backend and Frontend
REM ============================================
echo [4/4] Starting Backend and Frontend...

REM Set CUDA environment for backend (fallback transcriber)
set CUDA_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cudnn\bin
set CUBLAS_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cublas\bin
set PATH=%CUDA_PATH%;%CUBLAS_PATH%;%PATH%

REM Start Backend
cd /d %~dp0backend
start "Backend" cmd /k "call venv\Scripts\activate && python main.py"

REM Start Frontend
cd /d %~dp0frontend
start "Frontend" cmd /k "npm run dev"

echo.
echo ============================================
echo  [OK] All services started!
echo.
echo  Frontend:    http://localhost:5173
echo  Backend:     http://localhost:8000
echo  Whisper API: http://localhost:8207
echo  Ollama:      http://localhost:11434
echo ============================================
echo.
echo Press any key to close this window (services will keep running)...
pause > nul
