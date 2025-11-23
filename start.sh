#!/bin/bash
# Ghost Protocol - Production Startup Script
# Initializes databases and starts server

set -e  # Exit on error

echo "🚀 Ghost Protocol - Starting production server..."

# Detect environment (Railway uses /app, local uses current dir)
if [ -d "/app/data" ]; then
    SMART_WATCHER_DB="/app/data/smart_watcher.db"
    DATA_DIR="/app/data"
else
    SMART_WATCHER_DB="data/smart_watcher.db"
    DATA_DIR="data"
fi

WATCHLIST_DB="watchlist.db"

# Initialize Smart Watcher if empty or missing
SMART_COUNT=$(sqlite3 "$SMART_WATCHER_DB" "SELECT COUNT(*) FROM watchlist;" 2>/dev/null || echo "0")
if [ "$SMART_COUNT" -eq 0 ]; then
    echo "📋 Initializing Smart Watcher (25 symbols)..."
    python3 scripts/init_smart_watcher.py || echo "⚠️  Smart Watcher init failed (non-fatal)"
else
    echo "✓ Smart Watcher already initialized ($SMART_COUNT symbols)"
fi

# Initialize Watchlist Manager if empty or missing
WATCHLIST_COUNT=$(sqlite3 "$WATCHLIST_DB" "SELECT COUNT(*) FROM watchlist;" 2>/dev/null || echo "0")
if [ "$WATCHLIST_COUNT" -eq 0 ]; then
    echo "📋 Initializing Watchlist Manager (82 symbols)..."
    python3 scripts/init_watchlist.py || echo "⚠️  Watchlist Manager init failed (non-fatal)"
else
    echo "✓ Watchlist Manager already initialized ($WATCHLIST_COUNT symbols)"
fi

echo "✅ Initialization complete, starting server..."
echo ""

# Start uvicorn with Railway port (default 8080)
exec uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}
