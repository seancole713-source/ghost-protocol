#!/bin/bash
cd /workspaces/GHOST
source .venv/bin/activate
export SIM_MODE=1
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
echo "Starting Ghost in SIMULATION MODE..."
echo "SIM_MODE=$SIM_MODE"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload
