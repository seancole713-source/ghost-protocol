#!/bin/bash
# Ghost Protocol - Production Startup Script
# Starts server immediately - initialization runs in background

set -e  # Exit on error

echo "🚀 Ghost Protocol - Starting production server..."
echo "📍 Working directory: $(pwd)"

# DISABLED: Init script runs BEFORE server starts, blocking healthcheck
# Railway healthcheck needs server responding within 100s
# Initialization moved to background task in wolf_app.py startup
#
# echo "📋 Force-initializing watchlists and goals..."
# python3 scripts/force_init_watchlists.py 2>&1 || echo "⚠️  Init script failed (non-fatal, using defaults)"

echo "✅ Starting server immediately (init runs in background)..."
echo ""

# Start uvicorn with Railway port (default 8080)
exec uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}
