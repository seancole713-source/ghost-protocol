# Ghost Trading System - Production Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data /tmp/prom_multiproc

# Environment defaults
ENV SIM_MODE=0 \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prom_multiproc

# Expose port (Railway assigns PORT dynamically)
EXPOSE 8080

# Health check (use Railway's PORT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8080}/ui/health || exit 1

# Run the application with uvicorn (Railway PORT)
CMD uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}
