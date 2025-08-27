#!/usr/bin/env python3
"""
Quick script to check the webapp's runtime environment
"""
import torch
import sys
import os
from pathlib import Path

def check_runtime():
    print("=== RUNTIME ENVIRONMENT CHECK ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    print("=== GPU/CUDA STATUS ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"GPU memory test: SUCCESS")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU memory test: FAILED - {e}")
    else:
        print("No CUDA GPU detected - running on CPU")
        print("⚠️  This will make transcription 10-20x slower!")
    
    print()
    print("=== WHISPER MODEL STATUS ===")
    try:
        import whisper
        print(f"Whisper version: {whisper.__version__ if hasattr(whisper, '__version__') else 'Unknown'}")
        
        # Test loading a small model
        print("Testing Whisper model loading...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("tiny").to(device)
        print(f"✅ Whisper model loaded successfully on {device}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Whisper model loading failed: {e}")
    
    print()
    print("=== SYSTEM RESOURCES ===")
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"CPU usage: {cpu_percent}%")
        print(f"RAM: {memory.percent}% used ({memory.used / 1024**3:.1f} / {memory.total / 1024**3:.1f} GB)")
    except ImportError:
        print("psutil not available - install with: pip install psutil")
    except Exception as e:
        print(f"Resource check failed: {e}")

if __name__ == "__main__":
    check_runtime()