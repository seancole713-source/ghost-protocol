#!/bin/bash
# Ghost Protocol - Production Startup Script
# Initializes databases and starts server

set -e  # Exit on error

echo "🚀 Ghost Protocol - Starting production server..."

# Initialize watchlists if databases don't exist or are empty
if [ ! -f /app/data/smart_watcher.db ] || [ $(sqlite3 /app/data/smart_watcher.db "SELECT COUNT(*) FROM watchlist;" 2>/dev/null || echo "0") -eq 0 ]; then
    echo "📋 Initializing Smart Watcher (25 symbols)..."
    python3 scripts/init_smart_watcher.py || echo "⚠️  Smart Watcher init failed (non-fatal)"
else
    echo "✓ Smart Watcher already initialized"
fi

if [ ! -f watchlist.db ] || [ $(sqlite3 watchlist.db "SELECT COUNT(*) FROM watchlist;" 2>/dev/null || echo "0") -eq 0 ]; then
    echo "📋 Initializing Watchlist Manager (82 symbols)..."
    python3 scripts/init_watchlist.py || echo "⚠️  Watchlist Manager init failed (non-fatal)"
else
    echo "✓ Watchlist Manager already initialized"
fi

echo "✅ Initialization complete, starting server..."
echo ""

# Start uvicorn with Railway port (default 8080)
exec uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}
