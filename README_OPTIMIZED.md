# Class Transcriber - Optimized Web Application

A high-performance transcription service optimized for CPU/GPU acceleration, designed for Minerva University class recordings with AI-powered analysis capabilities.

## 🚀 Quick Start (Recommended)

```bash
npm start
```

This automatically:
- Detects available hardware (CPU/GPU)  
- Optimizes PyTorch for your system
- Starts the web application on http://localhost:8888

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js (for npm scripts)
- FFmpeg

### One-Command Setup
```bash
# Install all dependencies and start optimized
npm install && npm start
```

### Manual Setup (Alternative)
```bash
# 1. Install Python dependencies with CPU optimization
pip install -r requirements.txt

# 2. Start with hardware detection
python app.py
```

## ⚡ Performance Optimizations

This application is pre-configured for optimal performance:

### CPU Optimization
- **PyTorch CPU-only builds** for faster startup on CPU-only systems
- **Intel MKL** acceleration when available
- **Multi-threading** for parallel audio processing

### GPU Acceleration (Auto-detected)
- **CUDA support** for NVIDIA GPUs
- **Metal Performance Shaders** for Apple Silicon
- **ROCm** for AMD GPUs

### Memory Management
- **Chunked processing** for large audio files
- **Automatic memory cleanup** during transcription
- **Optimized buffer sizes** based on available RAM

## 🖥️ System Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 2GB free space

### Recommended
- **CPU**: 8+ cores or dedicated GPU
- **RAM**: 16GB+
- **GPU**: 4GB+ VRAM (optional but recommended)

## 🛠️ Available Commands

```bash
# Development
npm start                 # Start optimized application
npm run install         # Install Python dependencies  
npm run docker:start    # Start with Docker (production-ready)

# Docker Commands  
npm run docker:build    # Build containers
npm run docker:up       # Start existing containers
npm run docker:stop     # Stop all containers

# Utilities
npm run health          # Check system optimization status
```

## 🎯 Features

### Core Transcription
- **Hardware-accelerated** audio processing
- **Multiple input sources**: File upload, URLs, Google Drive
- **Real-time progress** tracking with ETA
- **Privacy modes**: Names, IDs, or both versions
- **Professional outputs**: PDF and CSV formats

### AI Analysis (GPT-4 Powered)
- **Intelligent transcript analysis** 
- **Multi-file support** (PDF/CSV)
- **Educational insights** for teaching improvement
- **Interactive chat** with transcript content
- **One-click comprehensive reports**

### Performance Features
- **Automatic GPU detection** and utilization
- **Optimized model loading** for faster startup
- **Memory-efficient processing** for large files
- **Concurrent transcription** support
- **Smart caching** for repeated operations

## 🔧 Configuration

### Environment Variables
```bash
# Required for AI features
OPENAI_API_KEY=your_api_key_here

# Performance tuning
TORCH_THREADS=auto              # Auto-detect optimal thread count
WHISPER_DEVICE=auto            # Auto-select best device (cpu/cuda/mps)
MAX_WORKERS=4                  # Concurrent transcription jobs
MEMORY_LIMIT=8192              # Memory limit in MB

# Development
DEV_MODE=false                 # Enable development features
PORT=8888                      # Application port
```

### Hardware Detection
The application automatically detects and optimizes for:
- **Apple Silicon** (M1/M2/M3 chips) → Metal Performance Shaders
- **Intel/AMD CPUs** → MKL-DNN acceleration  
- **NVIDIA GPUs** → CUDA acceleration
- **AMD GPUs** → ROCm support

## 📊 Performance Benchmarks

Typical transcription speeds (1-hour recording):

| Hardware | Processing Time | Notes |
|----------|----------------|-------|
| M1 MacBook Pro | 3-5 minutes | Metal acceleration |
| Intel i7 + RTX 3080 | 2-4 minutes | CUDA acceleration |
| Intel i5 CPU-only | 8-12 minutes | CPU optimization |
| Cloud CPU (4 cores) | 15-20 minutes | Standard performance |

## 🐳 Docker Deployment

For production environments:

```bash
# Quick production start
npm run docker:start

# Or manual Docker
docker-compose up --build
```

Docker includes:
- **Multi-stage builds** for smaller images
- **Health checks** and auto-restart
- **Volume mounts** for persistent storage
- **Production WSGI** server (Gunicorn)

## 🔍 Usage Guide

### 1. Forum Integration
- Open Chrome DevTools (F12) on Forum class page
- Network tab → Copy any request as cURL
- Paste into application for automatic metadata

### 2. Audio Sources
- **File Upload**: Drag & drop or browse
- **Direct URL**: Paste media URL
- **Google Drive**: Enter mounted drive path

### 3. AI Analysis
- Upload generated transcripts (PDF/CSV)
- Click "Initial Analysis" for comprehensive insights
- Ask specific questions about class content

## 🚨 Troubleshooting

### Performance Issues
```bash
# Check system optimization
npm run health

# Monitor resource usage
htop  # or Activity Monitor on macOS

# Clear cache and restart
rm -rf __pycache__ .whisper_cache
npm start
```

### Common Fixes
- **Slow transcription**: Verify GPU acceleration is enabled
- **Memory errors**: Reduce file size or increase swap
- **CUDA errors**: Update GPU drivers
- **Audio format**: Convert to WAV/MP3 if unsupported

## 🔐 Security & Privacy

- **Local processing** - no data sent to external servers (except AI analysis)
- **Temporary file cleanup** after processing
- **Minerva email authentication** required
- **Secure session management** 
- **No persistent storage** of sensitive audio

## 📈 Monitoring

Built-in monitoring includes:
- **Real-time progress** with detailed status
- **Resource utilization** tracking
- **Error logging** with timestamps
- **Performance metrics** per transcription

## 🤝 Support

For issues or optimization questions:
1. Check logs: `docker-compose logs` 
2. Verify hardware detection in startup messages
3. Monitor resource usage during processing
4. Report performance issues with system specs

---

**Optimized for**: Apple Silicon, Intel/AMD CPUs, NVIDIA/AMD GPUs, Cloud deployment

**Accuracy Notice**: Always manually verify critical information from automated transcriptions.