# Podcast Transcript Downloader

Download podcasts from RSS feeds and automatically transcribe them using **Whisper AI** with GPU acceleration.

![Demo](https://img.shields.io/badge/Status-Ready-green) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- 📡 **RSS Feed Support** - Input any podcast RSS/Atom feed URL
- 🎙️ **AI Transcription** - Uses faster-whisper with CUDA GPU acceleration
- 🧠 **LLM Text Polishing** - Local LLM adds punctuation, removes filler words
- 🚀 **Pipeline Processing** - Parallel downloads, efficient transcription
- 📁 **Smart File Management** - Auto-organize by show/episode with date
- ⏭️ **Skip Existing** - Automatically skip already transcribed episodes
- 🌐 **Web Interface** - Modern React UI with real-time progress updates


## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python |
| AI Model | faster-whisper (large-v3) |
| Frontend | React + Vite |
| GPU | CUDA (RTX 4070 Super optimized) |

## 📦 Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install CUDA libraries for faster-whisper
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# Install LLM for text polishing (optional)
pip install llama-cpp-python
```

### LLM Model Setup (Optional)

For text polishing with punctuation and grammar fixes:

1. Download a GGUF model from [Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
2. Place `qwen2.5-3b-instruct-q4_k_m.gguf` in `backend/models/`

The app will auto-detect and use the model. Without it, transcription still works but without text polishing.

### Frontend Setup

```bash
cd frontend
npm install
```

## 🚀 Running the Application

### Start Backend Server

```bash
cd backend
python main.py
# Or use: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Start Frontend Dev Server

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

## 📖 Usage

1. Open the web interface at `http://localhost:5173`
2. Paste your podcast RSS feed URL
3. Configure options:
   - **Episode count**: How many recent episodes to process
   - **Output directory**: Where to save transcripts
   - **Language**: Optional, auto-detects by default
4. Click **Start Transcription**
5. Watch real-time progress as episodes are downloaded and transcribed

## 📁 Output Structure

Transcripts are organized as:

```
transcripts/
└── Show Name/
    ├── 001_Episode_Title_2024-01-15.txt
    ├── 002_Another_Episode_2024-01-22.txt
    └── ...
```

Each transcript includes metadata header:

```
# Episode Title
# Show: Podcast Name
# Date: 2024-01-15
# Duration: 45 minutes

--------------------------------------------------

[Transcript content...]
```

## ⚙️ Configuration

Create a `.env` file in the `backend` directory:

```env
# Whisper model settings
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Output paths
OUTPUT_BASE_DIR=./transcripts
TEMP_AUDIO_DIR=./temp_audio

# Server
HOST=0.0.0.0
PORT=8000
```

### Model Options

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| `tiny` | ⚡⚡⚡⚡⚡ | ⭐ | ~1GB |
| `base` | ⚡⚡⚡⚡ | ⭐⭐ | ~1GB |
| `small` | ⚡⚡⚡ | ⭐⭐⭐ | ~2GB |
| `medium` | ⚡⚡ | ⭐⭐⭐⭐ | ~5GB |
| `large-v3` | ⚡ | ⭐⭐⭐⭐⭐ | ~10GB |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/transcribe` | POST | Start a transcription job |
| `/api/jobs/{id}` | GET | Get job status |
| `/api/jobs/{id}/cancel` | POST | Cancel running job |
| `/api/preview` | POST | Preview episodes without downloading |
| `/api/ws/progress/{id}` | WebSocket | Real-time progress updates |

## 📝 License

MIT License - feel free to use and modify!
