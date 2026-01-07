@echo off
REM ============================================
REM Podcast Transcript Downloader - Startup
REM ============================================

echo.
echo  ============================================
echo   Podcast Transcript Downloader
echo   Starting...
echo  ============================================
echo.

REM Check for LLM model
if not exist "%~dp0backend\models\qwen2.5-3b-instruct-q4_k_m.gguf" (
    echo  [!] LLM Model not found!
    echo      For text polishing, download the model:
    echo      https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
    echo      Place qwen2.5-3b-instruct-q4_k_m.gguf in backend/models/
    echo.
    echo      Transcription will still work without LLM polishing.
    echo.
)

REM Set CUDA environment
set CUDA_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cudnn\bin
set CUBLAS_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cublas\bin
set PATH=%CUDA_PATH%;%CUBLAS_PATH%;%PATH%

echo [1/2] Starting Backend (FastAPI + Whisper)...
cd /d %~dp0backend
start "Backend" cmd /k "call venv\Scripts\activate && python main.py"

echo [2/2] Starting Frontend (React + Vite)...
cd /d %~dp0frontend
start "Frontend" cmd /k "npm run dev"

echo.
echo ============================================
echo  [OK] Started!
echo.
echo  Frontend: http://localhost:5173
echo  Backend:  http://localhost:8000
echo ============================================
echo.
echo Press any key to close this window (services will keep running)...
pause > nul
