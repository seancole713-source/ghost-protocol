#!/bin/bash
# Ghost Protocol - Production Startup Script
# Initializes databases and starts server

set -e  # Exit on error

echo "🚀 Ghost Protocol - Starting production server..."

# ALWAYS run force init (idempotent - skips existing symbols)
echo "📋 Force-initializing watchlists..."
python3 scripts/force_init_watchlists.py || echo "⚠️  Watchlist init failed (non-fatal, using defaults)"

echo "✅ Initialization complete, starting server..."
echo ""

# Start uvicorn with Railway port (default 8080)
exec uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}
