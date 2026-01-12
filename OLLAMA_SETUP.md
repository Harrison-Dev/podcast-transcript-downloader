# Ollama Setup Guide

This guide covers installing and configuring Ollama for LLM text polishing.

## Prerequisites

- Windows 10/11 with NVIDIA GPU (8GB+ VRAM recommended)
- NVIDIA GPU drivers installed

## Step 1: Install Ollama

### Windows

1. Download Ollama from: https://ollama.com/download/windows
2. Run the installer (OllamaSetup.exe)
3. Follow the installation wizard
4. Ollama will run as a background service

### Verify Installation

Open PowerShell or Command Prompt:
```powershell
ollama --version
```

## Step 2: Pull the Model

The default model is `qwen3:8b` (requires ~6GB VRAM).

```powershell
# Pull the default model
ollama pull qwen3:8b
```

### Alternative Models

For lower VRAM usage:
```powershell
# 4B model (~4GB VRAM)
ollama pull qwen3:4b
```

For better quality (requires more VRAM):
```powershell
# 14B model (~10GB VRAM)
ollama pull qwen3:14b
```

Update `backend/.env` with your chosen model:
```
OLLAMA_MODEL=qwen3:4b
```

## Step 3: Verify Ollama is Running

```powershell
# Check if Ollama API is accessible
curl http://localhost:11434/api/tags
```

Expected response:
```json
{"models":[{"name":"qwen3:8b",...}]}
```

## Step 4: Test LLM Generation

```powershell
# Quick test
curl -X POST http://localhost:11434/api/generate -d "{\"model\":\"qwen3:8b\",\"prompt\":\"Hello\",\"stream\":false}"
```

## Troubleshooting

### Ollama Service Not Running

```powershell
# Start Ollama service (Windows)
ollama serve
```

Or restart from system tray icon.

### GPU Not Detected

1. Update NVIDIA drivers to latest version
2. Restart Ollama service
3. Check CUDA availability:
```powershell
nvidia-smi
```

### Model Too Large for VRAM

Switch to a smaller model:
```powershell
ollama pull qwen3:4b
```

Update `backend/.env`:
```
OLLAMA_MODEL=qwen3:4b
```

### API Connection Refused

1. Check if Ollama is running: Look for Ollama icon in system tray
2. Check firewall settings
3. Verify port 11434 is not blocked

## Configuration

The backend uses these environment variables (in `backend/.env`):

```env
# Ollama Configuration
OLLAMA_API=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_TEMPERATURE=0.3
OLLAMA_NUM_PREDICT=4096

# LLM Processing
LLM_ENABLED=true
LLM_TIMEOUT=300
BATCH_SIZE_PASS1=8
BATCH_SIZE_PASS2=20
```

## Memory Usage Tips

| Model | VRAM Required | Quality |
|-------|--------------|---------|
| qwen3:4b | ~4GB | Good |
| qwen3:8b | ~6GB | Better |
| qwen3:14b | ~10GB | Best |

For systems with limited VRAM, use smaller models or disable LLM polishing:
```env
LLM_ENABLED=false
```

## Running Without Ollama

If you don't want to use LLM polishing:

1. Set in `backend/.env`:
```env
LLM_ENABLED=false
```

2. Whisper transcription will still work, just without the two-pass text polishing.
