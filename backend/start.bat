@echo off
REM Podcast Transcript Downloader - Backend Startup Script
REM This script sets up the correct PATH for CUDA/cuDNN libraries

echo Setting up CUDA environment...

REM Add NVIDIA cuDNN and cuBLAS to PATH
set PATH=%~dp0venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%
set PATH=%~dp0venv\Lib\site-packages\nvidia\cublas\bin;%PATH%

echo Starting backend server...
echo.

REM Activate venv and start
call %~dp0venv\Scripts\activate
python main.py
