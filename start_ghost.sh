#!/bin/bash
# Ghost Server Startup Script with API Keys

# Load secrets
set -a
source /workspaces/GHOST/secrets.env 2>/dev/null || true
set +a

# Activate venv
source /workspaces/GHOST/.venv/bin/activate

# Set Prometheus dir
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Start server
exec uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload
