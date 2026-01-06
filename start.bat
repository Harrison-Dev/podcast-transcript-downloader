@echo off
REM ============================================
REM Podcast Transcript Downloader - 一鍵啟動
REM ============================================

echo.
echo  ╔═══════════════════════════════════════════╗
echo  ║   Podcast Transcript Downloader           ║
echo  ║   啟動中...                               ║
echo  ╚═══════════════════════════════════════════╝
echo.

REM 設定 CUDA 環境
set CUDA_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cudnn\bin
set CUBLAS_PATH=%~dp0backend\venv\Lib\site-packages\nvidia\cublas\bin
set PATH=%CUDA_PATH%;%CUBLAS_PATH%;%PATH%

echo [1/2] 啟動 Backend (FastAPI + Whisper)...
cd /d %~dp0backend
start "Backend" cmd /k "call venv\Scripts\activate && python main.py"

echo [2/2] 啟動 Frontend (React + Vite)...
cd /d %~dp0frontend
start "Frontend" cmd /k "npm run dev"

echo.
echo ============================================
echo  ✅ 已啟動！
echo.
echo  Frontend: http://localhost:5173
echo  Backend:  http://localhost:8000
echo ============================================
echo.
echo 按任意鍵關閉此視窗（不會關閉服務）...
pause > nul
