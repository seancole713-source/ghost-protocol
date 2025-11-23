#!/bin/bash
# Ghost Protocol - Production Startup Script
# Initializes databases and starts server

set -e  # Exit on error

echo "🚀 Ghost Protocol - Starting production server..."
echo "📍 Working directory: $(pwd)"
echo "📁 Data directory: $(ls -la data/ 2>/dev/null | head -5 || echo 'data/ not found')"

# ALWAYS run force init (idempotent - skips existing symbols)
echo "📋 Force-initializing watchlists and goals..."
python3 scripts/force_init_watchlists.py 2>&1 || echo "⚠️  Init script failed (non-fatal, using defaults)"

echo ""
echo "📊 Post-init verification:"
echo "   Smart Watcher: $(sqlite3 data/smart_watcher.db 'SELECT COUNT(*) FROM watchlist;' 2>/dev/null || echo '0') symbols"
echo "   Watchlist Manager: $(sqlite3 watchlist.db 'SELECT COUNT(*) FROM watchlist;' 2>/dev/null || echo '0') symbols"
echo "   Goals: $(sqlite3 data/goals.db 'SELECT COUNT(*) FROM goals;' 2>/dev/null || echo '0') configured"

echo ""
echo "✅ Initialization complete, starting server..."
echo ""

# Start uvicorn with Railway port (default 8080)
exec uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}
