# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Podcast Transcript Downloader is a full-stack application that downloads podcasts from RSS feeds and transcribes them using faster-whisper (GPU-accelerated Whisper AI). It features optional LLM-based text polishing via Ollama.

## Commands

### Quick Start (Windows)
```bash
./start.bat   # Starts both backend and frontend
```

### Backend
```bash
cd backend
python -m venv venv && venv\Scripts\activate  # Create/activate venv
pip install -r requirements.txt               # Install dependencies
python main.py                                # Run server (port 8000)
```

### Frontend
```bash
cd frontend
npm install      # Install dependencies
npm run dev      # Run dev server (port 5173)
npm run build    # Production build
```

## Architecture

### Backend (FastAPI + Python)

```
backend/
├── main.py          # FastAPI app entry point with CORS config
├── config.py        # Settings via pydantic-settings (env vars/.env support)
├── models.py        # Pydantic models (TranscriptionRequest, JobProgress, etc.)
├── routers/
│   └── transcription.py  # API endpoints + WebSocket for real-time progress
└── services/
    ├── pipeline.py       # PipelineOrchestrator: job management, download→transcribe flow
    ├── transcriber.py    # Whisper integration, Traditional Chinese conversion (OpenCC)
    ├── llm_processor.py  # Ollama-based text polishing (punctuation, cleanup)
    ├── downloader.py     # Async audio download with retry
    ├── rss_parser.py     # RSS/Atom feed parsing
    └── file_manager.py   # Transcript file organization
```

**Key flow:** `TranscriptionRequest` → `PipelineOrchestrator.create_job()` → `_process_job()` → for each episode: download → transcribe (Whisper) → polish (LLM) → save

### Frontend (React + Vite)
Simple React UI in `frontend/src/App.jsx` - communicates with backend via REST API and WebSocket for progress updates.

## Key Technical Details

- **GPU Memory Management:** Whisper and LLM models are loaded/unloaded sequentially to share GPU memory (`release_transcriber_gpu()`, `release_polisher_gpu()`)
- **Traditional Chinese:** Auto-converts Simplified to Traditional using OpenCC when Chinese is detected
- **VAD Parameters:** Configurable in `config.py` for better paragraph segmentation (min silence duration affects paragraph breaks)
- **WebSocket Progress:** Real-time updates via `/api/ws/progress/{job_id}`

## Environment Variables (backend/.env)

```
WHISPER_MODEL=large-v3        # tiny/base/small/medium/large-v3
WHISPER_DEVICE=cuda           # cuda or cpu
LLM_ENABLED=true
LLM_MODEL=qwen3:8b            # Ollama model name
LLM_OLLAMA_HOST=http://localhost:11434
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/transcribe` | POST | Start transcription job |
| `/api/jobs/{id}` | GET | Get job status |
| `/api/jobs/{id}/cancel` | POST | Cancel job |
| `/api/preview` | POST | Preview episodes without processing |
| `/api/ws/progress/{id}` | WS | Real-time progress |
