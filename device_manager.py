"""
Cross-platform device detection and optimization for transcription
Supports M1/M2/M3 (MPS), NVIDIA GPUs (CUDA), and CPU fallback
"""

import torch
import platform
import subprocess
import sys
from typing import Tuple, Dict, Any

class DeviceManager:
    """Manages device detection and optimization across different hardware"""
    
    def __init__(self):
        self.device = self._detect_best_device()
        self.device_info = self._get_device_info()
        self._apply_optimizations()
    
    def _detect_best_device(self) -> str:
        """Detect the best available device for transcription"""
        
        # Priority order: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if self._is_apple_silicon() and torch.backends.mps.is_available():
            if self._test_mps_functionality():
                return "mps"
            else:
                print("⚠️  MPS available but not fully functional, falling back...")
        
        if torch.cuda.is_available():
            if self._test_cuda_functionality():
                return "cuda"
            else:
                print("⚠️  CUDA available but not functional, falling back...")
        
        print("ℹ️  Using CPU - transcription will be slower")
        return "cpu"
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon (M1/M2/M3)"""
        if platform.system() != "Darwin":
            return False
        
        try:
            # Check for Apple Silicon architecture
            result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
            arch = result.stdout.strip()
            return arch == "arm64"
        except:
            return False
    
    def _test_mps_functionality(self) -> bool:
        """Test basic MPS functionality"""
        try:
            # Test basic tensor operations on MPS
            test_tensor = torch.randn(100, 100, device='mps')
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            return True
        except Exception as e:
            print(f"MPS test failed: {e}")
            return False
    
    def _test_cuda_functionality(self) -> bool:
        """Test basic CUDA functionality"""
        try:
            # Test basic tensor operations on CUDA
            test_tensor = torch.randn(100, 100, device='cuda')
            result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, result
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"CUDA test failed: {e}")
            return False
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed information about the selected device"""
        info = {
            'device': self.device,
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__
        }
        
        if self.device == "mps":
            info.update({
                'device_name': self._get_apple_chip_name(),
                'memory_info': 'Unified Memory Architecture',
                'supports_fp16': False,  # MPS has limited FP16 support
                'optimal_batch_size': 'auto'
            })
        
        elif self.device == "cuda":
            gpu_id = 0  # Using primary GPU
            info.update({
                'device_name': torch.cuda.get_device_name(gpu_id),
                'memory_total': f"{torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB",
                'cuda_version': torch.version.cuda,
                'supports_fp16': True,
                'optimal_batch_size': 'large'
            })
        
        else:  # CPU
            info.update({
                'device_name': 'CPU',
                'cores': torch.get_num_threads(),
                'supports_fp16': False,
                'optimal_batch_size': 'small'
            })
        
        return info
    
    def _get_apple_chip_name(self) -> str:
        """Get the specific Apple chip name (M1, M2, M3, etc.)"""
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            brand = result.stdout.strip()
            
            # Extract chip name from brand string
            if 'Apple M' in brand:
                # Extract M1, M2, M3, etc.
                import re
                match = re.search(r'Apple (M\d+[^\s]*)', brand)
                if match:
                    return match.group(1)
            
            return "Apple Silicon"
        except:
            return "Apple Silicon"
    
    def _apply_optimizations(self):
        """Apply device-specific optimizations"""
        
        if self.device == "mps":
            # MPS-specific optimizations
            # No specific backend settings needed for MPS
            print(f"🚀 Optimized for {self.device_info['device_name']}")
        
        elif self.device == "cuda":
            # CUDA-specific optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            try:
                # Set memory fraction - be more conservative on unknown GPUs
                torch.cuda.set_per_process_memory_fraction(0.85)
            except:
                pass
            
            print(f"🚀 Optimized for {self.device_info['device_name']}")
        
        else:  # CPU
            # CPU optimizations
            torch.set_num_threads(torch.get_num_threads())  # Use all available cores
            print(f"🚀 Optimized for CPU with {self.device_info['cores']} threads")
    
    def get_autocast_context(self):
        """Get the appropriate autocast context for the device"""
        if self.device == "cuda":
            return torch.amp.autocast("cuda")
        elif self.device == "mps":
            # MPS doesn't need autocast for most operations
            from contextlib import nullcontext
            return nullcontext()
        else:  # CPU
            from contextlib import nullcontext
            return nullcontext()
    
    def cleanup_memory(self):
        """Clean up device memory after operations"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # MPS memory is managed automatically, but we can trigger cleanup
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        # CPU doesn't need explicit cleanup
    
    def get_recommended_precision(self) -> bool:
        """Get recommended precision setting for fp16"""
        return self.device_info.get('supports_fp16', False)
    
    def print_device_summary(self):
        """Print a summary of the detected device and optimizations"""
        print("=" * 60)
        print("🖥️  DEVICE DETECTION SUMMARY")
        print("=" * 60)
        print(f"Selected Device: {self.device.upper()}")
        print(f"Device Name: {self.device_info.get('device_name', 'Unknown')}")
        print(f"Platform: {self.device_info['platform']} ({self.device_info['architecture']})")
        
        if self.device == "cuda":
            print(f"GPU Memory: {self.device_info.get('memory_total', 'Unknown')}")
            print(f"CUDA Version: {self.device_info.get('cuda_version', 'Unknown')}")
        elif self.device == "mps":
            print(f"Memory: {self.device_info.get('memory_info', 'Unknown')}")
        else:
            print(f"CPU Threads: {self.device_info.get('cores', 'Unknown')}")
        
        print(f"FP16 Support: {'✅' if self.device_info.get('supports_fp16') else '❌'}")
        print(f"PyTorch Version: {self.device_info['pytorch_version']}")
        print("=" * 60)

# Convenience function for quick device detection
def get_optimal_device() -> DeviceManager:
    """Quick function to get optimally configured device manager"""
    return DeviceManager()

if __name__ == "__main__":
    # Test device detection
    device_manager = get_optimal_device()
    device_manager.print_device_summary()