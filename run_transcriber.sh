#!/bin/bash
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
