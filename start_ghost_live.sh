#!/bin/bash
# === GHOST v10.2 — LIVE START + VERIFY ===
# Complete startup and verification script for GHOST trading system

set -e  # Exit on error

echo "=================================================="
echo "GHOST v10.2 - Live Startup & Verification"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 0) Ensure Python deps
echo -e "${BLUE}[0/6]${NC} Checking Python dependencies..."
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found, creating...${NC}"
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# 1) Live env configuration
echo -e "${BLUE}[1/6]${NC} Configuring environment..."

ENV_FILE=".env.local"
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}✓ Using $ENV_FILE overrides${NC}"
elif [ -f ".env" ]; then
    ENV_FILE=".env"
    echo -e "${YELLOW}⚠️  Using tracked .env file; prefer .env.local for secrets${NC}"
else
    cat <<'MSG'
No environment file detected.

Fetch the canonical values directly from Railway (Project "tender-benevolence" → Service "ghost-protocol")
and write only the overrides you need to `.env.local`:

  railway variables --service ghost-protocol --json > /tmp/ghost_env.json
  # copy the values you actually need into .env.local (which stays ignored)

Then re-run this script.
MSG
    exit 1
fi

# Load environment variables (bash will ignore comments)
set -a
source "$ENV_FILE"
set +a

# Validate critical env vars
if [ -z "${ALPHAVANTAGE_API_KEY:-}" ]; then
    echo -e "${RED}❌ ALPHAVANTAGE_API_KEY missing. Pull it from Railway Variables.${NC}"
    exit 1
fi

if [ -z "${POLYGON_API_KEY:-}" ]; then
    echo -e "${RED}❌ POLYGON_API_KEY missing. Pull it from Railway Variables.${NC}"
    exit 1
fi

if [ -z "${GHOST_API_TOKEN:-}" ]; then
    echo -e "${YELLOW}⚠️  Generating temporary GHOST_API_TOKEN for local use...${NC}"
    export GHOST_API_TOKEN=$(openssl rand -hex 32)
fi

echo -e "${GREEN}✅ Environment configured${NC}"
echo "   ENV_FILE: $ENV_FILE"
echo "   SIM_MODE: ${SIM_MODE:-unset}"
echo "   ALPHAVANTAGE_API_KEY: ${ALPHAVANTAGE_API_KEY:0:6}***"
echo "   POLYGON_API_KEY: ${POLYGON_API_KEY:0:6}***"
echo "   PRICE_TTL_OPEN_S: ${PRICE_TTL_OPEN_S:-60}"
echo ""

# 2) Start FastAPI on port 5000
echo -e "${BLUE}[2/6]${NC} Starting GHOST server..."

# Kill any existing processes
pkill -f "wolf_app.py" || true
pkill -f "uvicorn.*wolf_app" || true
sleep 2

# Check if port 5000 is free
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}❌ Port 5000 is already in use${NC}"
    lsof -i :5000
    exit 1
fi

# Start server in background
echo "Starting uvicorn on http://0.0.0.0:5000"
nohup .venv/bin/python wolf_app.py > ghost_server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > ghost_server.pid
echo -e "${GREEN}✅ Server started (PID: $SERVER_PID)${NC}"
echo ""

# 3) Wait and verify core health
echo -e "${BLUE}[3/6]${NC} Waiting for server to be ready..."
for i in {1..15}; do
    if curl -s http://127.0.0.1:5000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Server is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 15 ]; then
        echo -e "${RED}❌ Server failed to start${NC}"
        echo "Check logs: tail -50 ghost_server.log"
        exit 1
    fi
done
echo ""

# Health checks
echo -e "${BLUE}[3a]${NC} Health Check:"
HEALTH=$(curl -s http://127.0.0.1:5000/health)
echo "$HEALTH" | jq '.'
if echo "$HEALTH" | jq -e '.ok == true' >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}[3b]${NC} Provider Health:"
curl -s http://127.0.0.1:5000/api/price/diagnostics 2>&1 | jq '{price, provider, cache_age_s}' || echo "Provider check unavailable"
echo ""

echo -e "${BLUE}[3c]${NC} Multi-Symbol Quotes:"
curl -s "http://127.0.0.1:5000/api/watchlist/price?symbol=WOLF" 2>&1 | jq '.' || echo "Quotes unavailable"
echo ""

echo -e "${BLUE}[3d]${NC} Cockpit Snapshot:"
COCKPIT=$(curl -s http://127.0.0.1:5000/api/cockpit 2>&1)
if echo "$COCKPIT" | jq -e '.snapshot_id' >/dev/null 2>&1; then
    echo "$COCKPIT" | jq '{snapshot_id, prices, portfolio: {qty, market_value}, forecast_summary}'
    echo -e "${GREEN}✅ Cockpit operational${NC}"
else
    echo "$COCKPIT" | head -20
    echo -e "${YELLOW}⚠️  Cockpit may have issues${NC}"
fi
echo ""

# 4) Force live refresh + check forecast
echo -e "${BLUE}[4/6]${NC} Forcing data refresh..."

echo "Triggering advisor refresh..."
REFRESH=$(curl -s -X POST http://127.0.0.1:5000/api/advisor_refresh 2>&1)
if echo "$REFRESH" | jq -e '.ok == true' >/dev/null 2>&1; then
    echo "$REFRESH" | jq '{ok, result: {forecast_id, price_now, price_pred_mid, confidence}}'
    echo -e "${GREEN}✅ Advisor refresh successful${NC}"
else
    echo "$REFRESH" | head -20
    echo -e "${YELLOW}⚠️  Refresh may have issues${NC}"
fi
echo ""

echo "Checking watchlist/top movers..."
curl -s "http://127.0.0.1:5000/api/top_movers?threshold=7.0&limit=5" 2>&1 | jq '.stocks[0:3]' || echo "Top movers unavailable"
echo ""

echo "Checking 48h forecast..."
curl -s "http://127.0.0.1:5000/forecast/48h/recent?symbol=WOLF&limit=3" 2>&1 | jq '.' || echo "Forecast unavailable"
echo ""

echo "Checking forecast metrics..."
curl -s "http://127.0.0.1:5000/forecast/48h/metrics?symbol=WOLF" 2>&1 | jq '.' || echo "Metrics unavailable"
echo ""

# 5) Prometheus sanity (optional)
echo -e "${BLUE}[5/6]${NC} Checking Prometheus metrics (optional)..."
if curl -sf http://127.0.0.1:5000/metrics | head -5 >/dev/null 2>&1; then
    METRIC_COUNT=$(curl -sf http://127.0.0.1:5000/metrics | grep -c "^ghost_" || echo 0)
    echo -e "${GREEN}✅ Metrics endpoint active ($METRIC_COUNT GHOST metrics)${NC}"
else
    echo -e "${YELLOW}⚠️  Metrics endpoint disabled or unavailable${NC}"
fi
echo ""

# Summary
echo "=================================================="
echo -e "${GREEN}✅ GHOST v10.2 is LIVE!${NC}"
echo "=================================================="
echo ""
echo "🌐 Access URLs:"
echo "   • Local:      http://127.0.0.1:5000"
echo "   • Network:    http://0.0.0.0:5000"
echo "   • Railway:    https://web-production-8e9a0.up.railway.app"
echo ""
echo "📊 Key Endpoints:"
echo "   • Health:     http://127.0.0.1:5000/health"
echo "   • Cockpit:    http://127.0.0.1:5000/api/cockpit"
echo "   • Forecast:   http://127.0.0.1:5000/forecast/48h/recent?symbol=WOLF"
echo "   • API Docs:   http://127.0.0.1:5000/docs"
echo ""
echo "📁 Important Files:"
echo "   • Server Log:  tail -f ghost_server.log"
echo "   • Server PID:  cat ghost_server.pid"
echo "   • Database:    data/wolf.db"
echo ""
echo "🛑 To Stop Server:"
echo "   pkill -f wolf_app.py"
echo "   # OR"
echo "   kill \$(cat ghost_server.pid)"
echo ""

# Check if in Codespaces
if [ -n "$CODESPACE_NAME" ]; then
    echo -e "${YELLOW}📝 Codespaces Detected:${NC}"
    echo "   1. Go to PORTS tab"
    echo "   2. Make port 5000 PUBLIC"
    echo "   3. Click globe icon to open in browser"
    echo ""
fi

# 6) Tag release (optional)
echo -e "${BLUE}[6/6]${NC} Git tagging (optional)..."
read -p "Tag this as v10.2.0 release? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git pull --rebase origin main
    if git tag v10.2.0 2>/dev/null; then
        git push origin v10.2.0
        echo -e "${GREEN}✅ Tagged and pushed v10.2.0${NC}"
    else
        echo -e "${YELLOW}⚠️  Tag v10.2.0 already exists${NC}"
    fi
else
    echo "Skipping git tag"
fi
echo ""

echo "=================================================="
echo "🚀 Startup complete! Monitor logs with:"
echo "   tail -f ghost_server.log"
echo "=================================================="
