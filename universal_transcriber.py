"""
Universal Cross-Platform Transcriber
Automatically detects and uses optimal hardware: M1/M2/M3 (MPS), NVIDIA GPUs (CUDA), or CPU
"""

import gc
import whisper
import json
import datetime
from pathlib import Path
import torch
from pydub import AudioSegment
import numpy as np
import requests
import re
import iso8601
import csv
import subprocess
from datetime import timedelta
import tempfile
from contextlib import nullcontext
from device_manager import DeviceManager
import sys
import argparse
import os
from typing import Optional

class UniversalTranscriptionProcessor:
    """Cross-platform transcription processor that adapts to available hardware"""
    
    def __init__(self, segment_length=14400, model_name="medium", progress_callback=None):
        # Initialize device manager for cross-platform support
        self.device_manager = DeviceManager()
        self.device = self.device_manager.device
        self.device_info = self.device_manager.device_info
        
        self.progress_callback = progress_callback
        self.segment_length = int(segment_length)
        
        # Load and configure model
        print(f"Loading Whisper '{model_name}' model on {self.device.upper()}...")
        self.model = whisper.load_model(model_name).to(self.device)
        
        # Apply precision optimization based on device capabilities
        if self.device_manager.get_recommended_precision():
            self.model = self.model.half()
            print("✅ Using FP16 precision for faster inference")
        else:
            print("ℹ️  Using FP32 precision (FP16 not supported/recommended)")

    def transcribe(self, audio_path: str, class_id: str, output_dir: str = ".") -> str:
        """Universal transcription that works across all supported devices"""
        
        print(f"🎵 Processing audio: {audio_path}")
        
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000.0  # Convert to seconds
            
            print(f"⏱️  Audio duration: {timedelta(seconds=int(total_duration))}")
            print(f"🔄 Processing in {self._calculate_segments(total_duration)} segments...")
            
            all_segments = []
            segment_times = range(0, int(total_duration), self.segment_length)
            
            for i, start_time in enumerate(segment_times):
                # Progress callback
                if self.progress_callback:
                    progress = (i / len(segment_times)) * 100
                    self.progress_callback(progress)
                
                print(f"📝 Processing segment {i+1}/{len(segment_times)} "
                      f"({self._format_time(start_time)}-{self._format_time(min(start_time + self.segment_length, total_duration))})")
                
                # Extract audio segment
                remaining = total_duration - start_time
                duration = min(self.segment_length, remaining)
                
                start_ms = int(start_time * 1000)
                end_ms = int((start_time + duration) * 1000)
                segment = audio[start_ms:end_ms]
                
                # Create temporary file for this segment
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    segment.export(temp_path, format="wav")
                
                try:
                    # Use appropriate autocast context for device
                    with self.device_manager.get_autocast_context():
                        result = self.model.transcribe(
                            temp_path,
                            word_timestamps=True,
                            language="en",
                            task="transcribe",
                            fp16=self.device_manager.get_recommended_precision(),
                            condition_on_previous_text=True,
                            initial_prompt="This is a university lecture."
                        )
                    
                    # Process segments with time offset
                    for seg in result.get("segments", []):
                        seg_start = float(seg.get("start", 0.0)) + start_time
                        seg_end = float(seg.get("end", 0.0)) + start_time
                        
                        # Process word timestamps
                        words = []
                        for w in seg.get("words", []) or []:
                            words.append({
                                "word": str(w.get("word", "")).strip(),
                                "start": float(w.get("start", 0.0)) + start_time,
                                "end": float(w.get("end", 0.0)) + start_time
                            })
                        
                        all_segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": self._normalize_text(str(seg.get("text", "")).strip()),
                            "words": words
                        })
                
                except Exception as segment_error:
                    print(f"⚠️  Error processing segment {i+1}: {segment_error}")
                    continue
                
                finally:
                    # Cleanup
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except:
                        pass
                    
                    # Device-specific memory cleanup
                    self.device_manager.cleanup_memory()
                    gc.collect()
            
            if not all_segments:
                raise RuntimeError("❌ No segments were successfully transcribed!")
            
            # Save transcript
            transcript_path = str(Path(output_dir) / f"session_{class_id}_transcript.json")
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump({
                    "segments": sorted(all_segments, key=lambda x: x["start"]),
                    "device_info": self.device_info,
                    "processing_time": str(datetime.datetime.now()),
                }, f, indent=2)
            
            print(f"✅ Transcript saved: {transcript_path}")
            return transcript_path
        
        except Exception as e:
            print(f"❌ Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def _calculate_segments(self, duration: float) -> int:
        """Calculate number of segments needed"""
        return max(1, int(duration // self.segment_length) + (1 if duration % self.segment_length > 0 else 0))
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text spacing and punctuation"""
        if not text:
            return text
        
        # Remove zero-width characters
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        text = text.replace('\u00A0', ' ')
        
        # Normalize spacing
        text = re.sub(r'\s*\n+\s*', ' ', text)
        text = re.sub(r'(\.\.\.)(?=\S)', r'\1 ', text)
        text = re.sub(r'(?<!\.)([.!?])(?=([""\'(\[]?[A-Za-z]))', r'\1 ', text)
        text = re.sub(r'([:;])(?=([""\'(\[]?[A-Za-z]))', r'\1 ', text)
        text = re.sub(r'([.!?][""\')\]])(?=\S)', r'\1 ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()

class UniversalAudioPreprocessor:
    """Cross-platform audio preprocessing"""
    
    @staticmethod
    def validate_and_process(file_path: str) -> str:
        """Validate and preprocess audio files for optimal transcription"""
        
        print(f"🔍 Validating audio file: {file_path}")
        
        # Handle URL downloads
        if file_path.startswith(('http://', 'https://')):
            file_path = UniversalAudioPreprocessor._download_from_url(file_path)
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"❌ Audio file not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.mp4':
                print("🎬 Converting MP4 to audio...")
                return UniversalAudioPreprocessor._convert_video_to_audio(file_path)
            elif file_ext in ['.mp3', '.m4a', '.aac', '.ogg']:
                print("🎵 Converting audio to WAV...")
                return UniversalAudioPreprocessor._convert_to_wav(file_path)
            elif file_ext == '.wav':
                print("✅ WAV file detected, using directly")
                return file_path
            else:
                raise ValueError(f"❌ Unsupported format: {file_ext}")
        
        except Exception as e:
            print(f"❌ Audio processing error: {str(e)}")
            raise RuntimeError(f"Audio processing failed: {str(e)}")
    
    @staticmethod
    def _download_from_url(url: str) -> str:
        """Download audio file from URL"""
        print(f"⬇️  Downloading from URL...")
        
        base = url.split('?', 1)[0]
        suffix = Path(base).suffix or ".mp4"
        local_path = tempfile.mktemp(suffix=suffix)
        
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            
            print(f"✅ Downloaded to: {local_path}")
            return local_path
        
        except Exception as e:
            raise RuntimeError(f"Download failed: {str(e)}")
    
    @staticmethod
    def _convert_video_to_audio(video_path: str) -> str:
        """Convert video file to audio"""
        audio_path = video_path.rsplit('.', 1)[0] + '.mp3'
        
        cmd = [
            'ffmpeg', '-y', '-v', 'warning',
            '-i', video_path, '-vn',
            '-acodec', 'libmp3lame', '-ar', '44100', '-ab', '192k',
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 or not Path(audio_path).exists():
            raise RuntimeError(f"Video conversion failed: {result.stderr}")
        
        return UniversalAudioPreprocessor._convert_to_wav(audio_path)
    
    @staticmethod
    def _convert_to_wav(audio_path: str) -> str:
        """Convert audio to WAV format optimized for Whisper"""
        wav_path = audio_path.rsplit('.', 1)[0] + '_whisper.wav'
        
        cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 or not Path(wav_path).exists():
            raise RuntimeError(f"WAV conversion failed: {result.stderr}")
        
        return wav_path

def main():
    """Main entry point for universal transcription"""
    
    parser = argparse.ArgumentParser(description="Universal Cross-Platform Transcriber")
    parser.add_argument("--audio", "-a", required=True, help="Audio/video file path or URL")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    parser.add_argument("--class-id", "-c", default=None, help="Class ID (auto-generated if not provided)")
    parser.add_argument("--model", "-m", default="medium", choices=["tiny", "base", "small", "medium", "large"], 
                       help="Whisper model size")
    parser.add_argument("--device-info", action="store_true", help="Show device information and exit")
    
    args = parser.parse_args()
    
    # Show device info if requested
    if args.device_info:
        device_manager = DeviceManager()
        device_manager.print_device_summary()
        return
    
    # Generate class ID if not provided
    class_id = args.class_id or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        print("🚀 Starting Universal Transcription...")
        print("=" * 60)
        
        # Initialize device manager and show info
        device_manager = DeviceManager()
        device_manager.print_device_summary()
        
        # Preprocess audio
        print("\n📁 Processing Audio File...")
        audio_processor = UniversalAudioPreprocessor()
        processed_audio = audio_processor.validate_and_process(args.audio)
        
        # Transcribe
        print(f"\n🎙️  Starting Transcription with {args.model} model...")
        
        def progress_callback(progress):
            print(f"Progress: {progress:.1f}%")
        
        transcription_processor = UniversalTranscriptionProcessor(
            model_name=args.model,
            progress_callback=progress_callback
        )
        
        transcript_path = transcription_processor.transcribe(
            processed_audio, 
            class_id, 
            args.output
        )
        
        print("\n" + "=" * 60)
        print("🎉 TRANSCRIPTION COMPLETED!")
        print("=" * 60)
        print(f"📄 Output file: {transcript_path}")
        print(f"🖥️  Device used: {device_manager.device.upper()}")
        print(f"🔧 Model: {args.model}")
        
        # Cleanup temporary files
        if processed_audio != args.audio and Path(processed_audio).exists():
            try:
                Path(processed_audio).unlink()
                print(f"🧹 Cleaned up temporary file: {processed_audio}")
            except:
                pass
        
    except KeyboardInterrupt:
        print("\n❌ Transcription cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Transcription failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()