#!/bin/bash
# Restart Ghost with 5-second price cache TTL for real-time market data

echo "🔄 Restarting Ghost with real-time price updates..."

# Stop existing server
pkill -f "uvicorn wolf_app" 2>/dev/null || true
sleep 2

# Start with reduced TTL
cd /workspaces/GHOST
source .venv/bin/activate

export SIM_MODE=0
export PORTFOLIO_PERSISTENCE_ENABLED=1
export PRICE_TTL_OPEN_S=5         # ← 5 second cache during market hours
export PRICE_TTL_S=30              # ← 30 second cache when market closed
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload > ghost_server.out 2>&1 &
SERVER_PID=$!

echo "✅ Ghost restarted (PID: $SERVER_PID)"
echo "⏳ Waiting 10 seconds for startup..."
sleep 10

# Verify
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ Server running"
    curl -s http://localhost:5000/health | python3 -c "import json,sys; print(f'Health: {json.load(sys.stdin)}')" 2>/dev/null || echo "Waiting for health..."
else
    echo "❌ Server failed to start"
    tail -20 ghost_server.out
    exit 1
fi

echo ""
echo "🎯 Price cache TTL: 5 seconds (was 45 seconds)"
echo "📊 Prices will now update every 5 seconds during market hours"
echo "🔗 UI: https://crispy-happiness-q7gp6xvxr9r62xv9v-5000.app.github.dev/cockpit"
