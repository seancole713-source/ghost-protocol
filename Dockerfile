# Ghost Trading System - Production Dockerfile
# Force rebuild: Phase 5 deployment
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

# Create data directories with proper permissions for Railway volume
# /app/data will be mounted as Railway persistent volume
RUN mkdir -p /app/data /data /tmp/prom_multiproc && \
    chmod -R 777 /app/data /data /tmp/prom_multiproc

# Environment defaults
ENV SIM_MODE=0 \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prom_multiproc \
    GHOST_PREDICT_DB=/app/data/ghost_predictions.db

# Expose port (Railway assigns PORT dynamically)
EXPOSE 8080

# No HEALTHCHECK in Dockerfile - Railway has its own health check system
# Railway will hit /health endpoint directly via its load balancer

# Run startup script which initializes databases then starts server
CMD ["./start.sh"]
