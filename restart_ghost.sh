#!/bin/bash
#
# Ghost Server Restart Script
# Applies all fixes and restarts with correct configuration
#

set -e

echo "🔄 GHOST SERVER RESTART"
echo "========================================"
echo ""

# Stop current server
echo "1️⃣  Stopping current server..."
pkill -f "uvicorn wolf_app" 2>/dev/null || true
sleep 2
echo "   ✅ Stopped"
echo ""

# Add WOLF to watchlist
echo "2️⃣  Adding WOLF to watchlist..."
cd /workspaces/GHOST
source .venv/bin/activate
python add_wolf_to_watchlist.py || echo "   ⚠️  Note: May already be added"
echo ""

# Start server
echo "3️⃣  Starting Ghost server..."
cd /workspaces/GHOST
source .venv/bin/activate

export SIM_MODE=0
export USE_PLACEHOLDERS=0
export PORTFOLIO_PERSISTENCE_ENABLED=1
export ALERT_SCHEDULE_OPEN_CLOSE=1
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload > ghost_server.out 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > ghost_server.pid

echo "   ✅ Started (PID: $SERVER_PID)"
echo ""

# Wait for startup
echo "4️⃣  Waiting for server to initialize..."
sleep 8
echo "   ✅ Ready"
echo ""

# Verify
echo "5️⃣  Verifying..."
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "   ✅ Server running (PID: $SERVER_PID)"
else
    echo "   ❌ Server failed to start - check ghost_server.out"
    exit 1
fi

# Check for position restored log
if grep -q "position_restored_from_db" ghost_server.out 2>/dev/null; then
    QTY=$(grep "position_restored_from_db" ghost_server.out | tail -1 | grep -oP 'qty[":]*\K[0-9.]+' || echo "?")
    AVG=$(grep "position_restored_from_db" ghost_server.out | tail -1 | grep -oP 'avg[":]*\K[0-9.]+' || echo "?")
    echo "   ✅ Portfolio loaded: $QTY shares @ \$$AVG"
else
    echo "   ⚠️  Position restore log not found yet (may still be starting)"
fi

echo ""
echo "========================================"
echo "✅ GHOST SERVER RESTARTED SUCCESSFULLY"
echo ""
echo "📊 Access UI: https://crispy-happiness-q7gp6xvxr9r62xv9v-5000.app.github.dev/"
echo "📱 Test Telegram: Send /status to your bot"
echo "📝 View logs: tail -f ghost_server.out"
echo ""
echo "🎯 Next: Test Telegram /status to verify 8.41959051 shares!"
echo "========================================"
