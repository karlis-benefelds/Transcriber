# Class Transcriber Web Application

A web-based transcription service for Minerva University class recordings. Transcribes audio/video files, fetches Forum metadata, and generates polished PDF and CSV outputs with attendance and class events.

## Features

### Core Transcription
- **Multiple Input Sources**: Upload files, direct URLs, or Google Drive paths
- **Forum Integration**: Automatically fetches class metadata, attendance, and timeline events
- **Privacy Options**: Choose between showing names, anonymizing to IDs, or generating both versions
- **Professional Output**: Generates formatted PDF and CSV transcripts
- **Progress Tracking**: Real-time progress updates during transcription
- **Responsive Design**: Works on desktop and mobile devices

### AI Analysis (NEW!)
- **Intelligent Analysis**: Chat with GPT-4 about your class transcripts
- **File Upload**: Support for both PDF and CSV transcript files (multiple files allowed)
- **Initial Analysis**: One-click comprehensive analysis including:
  - Session overview and key topics discussed
  - Student participation patterns
  - Teaching methods and learning outcomes
  - Discussion quality assessment
  - Areas for improvement and notable moments
- **Interactive Chat**: Ask specific questions about transcript content
- **Educational Insights**: Tailored for professors to improve teaching effectiveness

## Quick Start

### Using npm (Recommended)

1. Clone or download the application files
2. Install dependencies and run:

```bash
npm run install
npm start
```

3. Open your browser to `http://localhost:8888`

### Using Docker

1. Clone or download the application files
2. Build and run with Docker Compose:

```bash
npm run docker:start
```
*or alternatively:*
```bash
docker-compose up --build
```

3. Open your browser to `http://localhost:5000`

### Manual Installation

1. Install Python 3.9+ and required system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip ffmpeg

# macOS (with Homebrew)
brew install python ffmpeg
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Configure OpenAI API (for AI Analysis feature):

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
```

4. Run the application:

```bash
python app.py
```

5. Open your browser to `http://localhost:8888`

### Available npm Scripts

- `npm start` - Start the application
- `npm run install` - Install Python dependencies
- `npm run docker:start` - Build and start with Docker
- `npm run docker:up` - Start existing Docker containers
- `npm run docker:build` - Build Docker containers only
- `npm run docker:stop` - Stop Docker containers

## Usage

1. **Get Forum cURL**: 
   - Open Chrome DevTools (F12) on a Forum class page
   - Go to Network tab and refresh the page
   - Find any API request and right-click → "Copy as cURL"
   - Paste this into the cURL field

2. **Choose Audio Source**:
   - **Upload**: Select a local audio/video file
   - **Direct URL**: Provide a direct link to the media file
   - **Google Drive**: Enter the full path to a file in mounted Drive

3. **Select Privacy Mode**:
   - **Names**: Show actual student names
   - **IDs**: Anonymize students to ID numbers
   - **Both**: Generate separate files for each option

4. **Start Transcription**: Click "Start Transcription" and wait for completion

5. **Download Results**: Download the generated PDF and CSV files

### Using AI Analysis

1. **Click "AI Analysis"** in the header to open the analysis interface

2. **Upload Transcript Files**:
   - Drag and drop PDF or CSV transcript files
   - Or click to browse and select files
   - Multiple files supported

3. **Get Initial Analysis**:
   - Click "Initial Analysis" for comprehensive automated insights
   - Includes participation patterns, teaching methods, and improvement suggestions

4. **Chat with AI**:
   - Type specific questions about your transcripts
   - Get detailed answers with references to transcript content
   - Examples: "Which students participated most?", "What were the main discussion topics?", "How can I improve student engagement?"

5. **Clear Chat**: Use "Clear Chat" to start fresh while keeping uploaded files

## Supported File Formats

- **Audio**: MP3, WAV, M4A, AAC, OGG
- **Video**: MP4 (audio will be extracted)

## System Requirements

- **Memory**: 8GB RAM minimum (16GB recommended for longer recordings)
- **GPU**: CUDA-compatible GPU recommended for faster transcription
- **Storage**: Sufficient space for temporary files during processing
- **Network**: Internet connection for Forum API access

## Deployment Considerations

### Production Deployment

For production use, consider:

1. **Reverse Proxy**: Use nginx or similar for SSL termination
2. **Authentication**: Add proper user authentication
3. **File Cleanup**: Implement automatic cleanup of temporary files
4. **Monitoring**: Add logging and monitoring
5. **Scaling**: Use a proper job queue system (Redis + Celery) for multiple concurrent transcriptions

### Environment Variables

- `FLASK_ENV`: Set to `production` for production deployment
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 500MB)

### Security Notes

- This application processes sensitive academic content
- Ensure proper network security and access controls
- Consider data retention policies for uploaded files
- The application currently stores job status in memory (not persistent)

## Troubleshooting

### Common Issues

1. **Upload Fails**: Check file size limits and format support
2. **cURL Invalid**: Ensure you're copying from a logged-in Forum session
3. **Transcription Errors**: Check audio quality and file corruption
4. **Memory Issues**: Reduce file size or increase system memory

### Logs

Check application logs for detailed error information:

```bash
docker-compose logs transcriber
```

## Accuracy Notice

**Important**: Do not rely solely on generated transcripts. Always manually verify key information for accuracy. Automated transcription may contain errors, especially with:

- Multiple speakers
- Technical terminology
- Poor audio quality
- Background noise
- Accented speech

## License

This application is designed specifically for Minerva University internal use.