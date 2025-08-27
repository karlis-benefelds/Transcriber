# Use Python 3.11 slim base image for Cloud Run
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn for production deployment
RUN pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/transcriber_uploads /tmp/transcriber_outputs \
    && chmod 755 /tmp/transcriber_uploads /tmp/transcriber_outputs

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose Cloud Run port
EXPOSE 8080

# Run with gunicorn for better performance and concurrency
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --max-requests 1000 --max-requests-jitter 50 app:app