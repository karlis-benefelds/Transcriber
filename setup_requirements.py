"""
Cross-platform setup script for Universal Transcriber
Automatically installs the correct PyTorch version for each platform
"""

import subprocess
import sys
import platform
import os

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"⚙️  {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def detect_platform():
    """Detect platform and architecture"""
    system = platform.system()
    machine = platform.machine()
    
    print(f"🖥️  Platform: {system} {machine}")
    
    if system == "Darwin" and machine == "arm64":
        return "apple_silicon"
    elif system == "Darwin":
        return "intel_mac"
    elif system == "Linux":
        return "linux"
    elif system == "Windows":
        return "windows"
    else:
        return "unknown"

def check_cuda_availability():
    """Check if NVIDIA CUDA is available"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            return True
        else:
            print("ℹ️  No NVIDIA GPU detected")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("ℹ️  nvidia-smi not found - no NVIDIA GPU")
        return False

def install_base_requirements():
    """Install base requirements common to all platforms"""
    requirements = [
        "openai-whisper",
        "pydub",
        "requests",
        "iso8601",
        "reportlab",
        "psutil",
        "tqdm"
    ]
    
    cmd = [sys.executable, "-m", "pip", "install"] + requirements
    return run_command(cmd, "Installing base requirements...")

def install_pytorch_for_platform(platform_type, has_cuda=False):
    """Install PyTorch optimized for the specific platform"""
    
    if platform_type == "apple_silicon":
        # Apple Silicon - use default PyTorch with MPS support
        cmd = [sys.executable, "-m", "pip", "install", 
               "torch", "torchvision", "torchaudio"]
        return run_command(cmd, "Installing PyTorch with MPS support for Apple Silicon...")
    
    elif platform_type in ["linux", "windows"] and has_cuda:
        # NVIDIA CUDA systems
        cmd = [sys.executable, "-m", "pip", "install",
               "torch", "torchvision", "torchaudio", 
               "--index-url", "https://download.pytorch.org/whl/cu121"]
        return run_command(cmd, "Installing PyTorch with CUDA support...")
    
    else:
        # CPU-only systems (Intel Mac, systems without CUDA)
        cmd = [sys.executable, "-m", "pip", "install",
               "torch", "torchvision", "torchaudio", 
               "--index-url", "https://download.pytorch.org/whl/cpu"]
        return run_command(cmd, "Installing CPU-only PyTorch...")

def install_system_dependencies(platform_type):
    """Install system-level dependencies"""
    
    if platform_type in ["apple_silicon", "intel_mac"]:
        # macOS - check for ffmpeg via Homebrew
        try:
            result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ ffmpeg already installed")
            else:
                print("📦 Please install ffmpeg using: brew install ffmpeg")
                print("   If you don't have Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        except:
            print("📦 Please install ffmpeg using: brew install ffmpeg")
    
    elif platform_type == "linux":
        print("📦 Please ensure ffmpeg is installed:")
        print("   Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("   CentOS/RHEL: sudo yum install ffmpeg")
        print("   Arch: sudo pacman -S ffmpeg")
    
    elif platform_type == "windows":
        print("📦 Please install ffmpeg:")
        print("   1. Download from https://ffmpeg.org/download.html#build-windows")
        print("   2. Add ffmpeg.exe to your PATH")
        print("   3. Or use chocolatey: choco install ffmpeg")

def test_installation():
    """Test that everything is installed correctly"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test PyTorch import
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        
        # Test device detection
        if torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon GPU) support available")
        elif torch.cuda.is_available():
            print(f"✅ CUDA support available - {torch.cuda.get_device_name()}")
        else:
            print("ℹ️  CPU-only mode (no GPU acceleration)")
        
        # Test Whisper import
        import whisper
        print("✅ OpenAI Whisper imported successfully")
        
        # Test other dependencies
        import pydub, requests, reportlab
        print("✅ All dependencies imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def create_run_script():
    """Create a convenient run script"""
    
    script_content = '''#!/bin/bash
# Universal Transcriber Runner
# This script runs the transcriber with optimal settings for your system

echo "🚀 Universal Transcriber"
echo "======================="

# Check if audio file is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: ./run_transcriber.sh <audio_file> [options]"
    echo ""
    echo "Examples:"
    echo "  ./run_transcriber.sh my_lecture.mp4"
    echo "  ./run_transcriber.sh https://example.com/audio.mp3 --model large"
    echo "  ./run_transcriber.sh lecture.wav --output ./transcripts --class-id CS101"
    echo ""
    echo "Options:"
    echo "  --model         Whisper model: tiny, base, small, medium, large (default: medium)"
    echo "  --output        Output directory (default: current directory)"
    echo "  --class-id      Class identifier (default: auto-generated)"
    echo "  --device-info   Show device information"
    exit 1
fi

# Run the universal transcriber
python3 universal_transcriber.py "$@"
'''
    
    with open("run_transcriber.sh", "w") as f:
        f.write(script_content)
    
    # Make it executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("run_transcriber.sh", 0o755)
        print("✅ Created run_transcriber.sh (executable)")
    else:
        print("✅ Created run_transcriber.sh")

def main():
    """Main setup function"""
    print("🔧 Universal Transcriber Setup")
    print("=" * 40)
    
    # Detect platform
    platform_type = detect_platform()
    has_cuda = check_cuda_availability() if platform_type in ["linux", "windows"] else False
    
    print(f"\n📋 Setup Plan:")
    print(f"   Platform: {platform_type}")
    if platform_type == "apple_silicon":
        print(f"   GPU: Apple Silicon (MPS)")
    elif has_cuda:
        print(f"   GPU: NVIDIA CUDA")
    else:
        print(f"   GPU: None (CPU only)")
    
    # Install dependencies
    print(f"\n🔽 Installing Dependencies...")
    
    success = True
    
    # Install base requirements
    if not install_base_requirements():
        success = False
    
    # Install PyTorch for platform
    if not install_pytorch_for_platform(platform_type, has_cuda):
        success = False
    
    # Install system dependencies
    install_system_dependencies(platform_type)
    
    if success:
        # Test installation
        if test_installation():
            create_run_script()
            
            print(f"\n🎉 Setup completed successfully!")
            print(f"=" * 40)
            print(f"📋 Next steps:")
            print(f"   1. Test device detection: python3 universal_transcriber.py --device-info")
            print(f"   2. Run a transcription: python3 universal_transcriber.py --audio your_file.mp4")
            print(f"   3. Or use the convenience script: ./run_transcriber.sh your_file.mp4")
            
        else:
            print(f"\n❌ Setup completed with errors - please check the installation")
    else:
        print(f"\n❌ Setup failed - please check the errors above")

if __name__ == "__main__":
    main()