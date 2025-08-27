# 🚀 Universal Cross-Platform Transcriber

A smart transcription system that **automatically detects and uses the best available hardware** across different platforms:

- **🍎 Apple Silicon (M1/M2/M3)**: Uses MPS acceleration  
- **🖥️ NVIDIA GPUs**: Uses CUDA acceleration  
- **💻 CPU Fallback**: Works on any system  

## 🎯 Key Features

✅ **Automatic Hardware Detection** - No manual configuration needed  
✅ **Cross-Platform Compatibility** - Works on macOS, Linux, Windows  
✅ **Optimal Performance** - Uses the fastest available processing method  
✅ **Simple Command Line Interface** - Easy to use for anyone  
✅ **Multiple File Formats** - Supports MP4, MP3, WAV, M4A, AAC, OGG, URLs  

## 🚀 Quick Start

### 1. Setup (One-time)
```bash
# Clone or download the files, then run:
python setup_requirements.py
```

### 2. Check Your Hardware
```bash
python universal_transcriber.py --device-info
```

### 3. Transcribe Audio
```bash
# Basic usage
python universal_transcriber.py --audio lecture.mp4

# Advanced usage
python universal_transcriber.py --audio lecture.mp4 --model large --output ./transcripts

# From URL
python universal_transcriber.py --audio "https://example.com/lecture.mp3"

# Or use the convenience script
./run_transcriber.sh lecture.mp4
```

## 🖥️ Performance Comparison

| Platform | Hardware | Expected Speed | Example (90min lecture) |
|----------|----------|----------------|-------------------------|
| **Apple Silicon M3** | MPS | ⚡ Very Fast | ~8-15 minutes |
| **Apple Silicon M2** | MPS | ⚡ Very Fast | ~10-18 minutes |
| **Apple Silicon M1** | MPS | ⚡ Fast | ~15-25 minutes |
| **NVIDIA RTX 4090** | CUDA | ⚡⚡ Extremely Fast | ~5-12 minutes |
| **NVIDIA RTX 3080** | CUDA | ⚡ Very Fast | ~8-15 minutes |
| **NVIDIA GTX 1080** | CUDA | ⚡ Fast | ~15-30 minutes |
| **Intel/AMD CPU** | CPU | 🐌 Slow | ~2-8 hours |

## 📋 Command Line Options

```bash
python universal_transcriber.py [OPTIONS]

Required:
  --audio, -a          Audio/video file path or URL

Optional:
  --output, -o         Output directory (default: current directory)
  --class-id, -c       Class identifier (default: auto-generated)
  --model, -m          Whisper model: tiny/base/small/medium/large (default: medium)
  --device-info        Show hardware information and exit

Examples:
  python universal_transcriber.py -a lecture.mp4 -m large -o ./output
  python universal_transcriber.py -a "https://site.com/audio.mp3" -c CS101
```

## 🔧 Model Selection Guide

| Model | Speed | Quality | VRAM Usage | Best For |
|-------|-------|---------|------------|----------|
| `tiny` | ⚡⚡⚡ Fastest | Basic | ~1GB | Quick drafts, testing |
| `base` | ⚡⚡ Very Fast | Good | ~1GB | Fast transcripts |
| `small` | ⚡ Fast | Better | ~2GB | Balanced speed/quality |
| `medium` | 📋 Balanced | Great | ~5GB | **Recommended default** |
| `large` | 🎯 Slower | Best | ~10GB | Highest accuracy needed |

## 🛠️ Installation Details

### Automatic Setup (Recommended)
```bash
python setup_requirements.py
```
This script automatically:
- Detects your platform (Apple Silicon, NVIDIA GPU, CPU-only)
- Installs the correct PyTorch version
- Installs all dependencies
- Tests the installation
- Creates convenience scripts

### Manual Setup
If you prefer manual installation:

**For Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision torchaudio
pip install openai-whisper pydub requests iso8601 reportlab psutil tqdm
brew install ffmpeg  # if not already installed
```

**For NVIDIA GPU systems:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install openai-whisper pydub requests iso8601 reportlab psutil tqdm
# Install ffmpeg for your system
```

**For CPU-only systems:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper pydub requests iso8601 reportlab psutil tqdm
# Install ffmpeg for your system
```

## 🔍 Troubleshooting

### "No module named 'device_manager'"
Make sure you're running the script from the directory containing all the files:
```bash
cd /path/to/transcriber/files
python universal_transcriber.py --device-info
```

### "MPS available but not functional"
This can happen with some PyTorch versions. The system will automatically fall back to CPU:
```bash
# Try updating PyTorch
pip install --upgrade torch torchvision torchaudio
```

### "CUDA available but not functional" 
Check your CUDA installation:
```bash
nvidia-smi  # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow transcription on Apple Silicon
Make sure you're using the MPS-enabled PyTorch:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
# Should show: MPS available: True
```

### FFmpeg not found
**macOS:** `brew install ffmpeg`  
**Linux:** `sudo apt install ffmpeg` (Ubuntu) or equivalent  
**Windows:** Download from https://ffmpeg.org or use `choco install ffmpeg`

## 📁 File Structure

```
universal-transcriber/
├── device_manager.py          # Hardware detection & optimization
├── universal_transcriber.py   # Main transcription script  
├── setup_requirements.py      # Automated setup script
├── run_transcriber.sh         # Convenience script (created by setup)
└── README_UNIVERSAL.md        # This file
```

## 🎓 For Developers

### Adding New Device Support
The `DeviceManager` class in `device_manager.py` can be extended to support new hardware:

```python
class DeviceManager:
    def _detect_best_device(self):
        # Add your device detection logic here
        if your_custom_device_available():
            return "your_device"
        # ... existing logic
```

### Integration with Existing Projects
You can easily integrate the universal transcriber into your existing projects:

```python
from universal_transcriber import UniversalTranscriptionProcessor

# Initialize with automatic device detection
transcriber = UniversalTranscriptionProcessor(model_name="medium")

# Transcribe audio
transcript_path = transcriber.transcribe(
    audio_path="lecture.mp4", 
    class_id="CS101", 
    output_dir="./output"
)
```

## 🤝 Contributing

Feel free to contribute improvements:
- Add support for new hardware platforms
- Optimize performance for specific devices  
- Add new audio format support
- Improve error handling

## 📄 License

This project builds upon OpenAI's Whisper model and other open-source libraries. Please respect their respective licenses.