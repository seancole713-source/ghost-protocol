# 🚀 GHOST LIVE MODE - QUICK REFERENCE

**Status:** 🟡 OPERATIONAL (87% Pass Rate)\
**Server:** Running on port 5000\
**Mode:** LIVE DATA (SIM_MODE=0)

______________________________________________________________________

## ⚡ QUICK STATUS CHECK

```bash
# Server status
lsof -i :5000

# Health check
curl -s http://localhost:5000/health | jq

# Cockpit snapshot
curl -s http://localhost:5000/api/cockpit | jq '{ticker, mode, gps, forecast: .forecast.points | length}'

# Portfolio
curl -s http://localhost:5000/api/portfolio | jq
```

______________________________________________________________________

## 🔧 KNOWN ISSUES & FIXES

### 1. Risk Status Endpoint (CRITICAL)

**Symptom:** Returns 500 error\
**Cause:** yfinance empty DataFrame, pandas operations fail\
**File:** `wolf_app.py`, line ~8970

**Quick Fix:**

```bash
# Apply patch (already documented in MASTER_LIVE_TEST_REPORT.md)
# Add empty DataFrame check before pandas operations
```

### 2. Price Endpoint 404

**Symptom:** `/api/prices/{symbol}` returns 404\
**Investigation Needed:** Verify route definition

### 3. Market Status Null

**Symptom:** Market status showing null\
**File:** `wolf_app.py`, `_build_market_status_with_indices()`

______________________________________________________________________

## 📱 UI ACCESS

**Main Dashboard:**\
https://crispy-happiness-q7gp6xvxr9r62xv9v-5000.app.github.dev/

**Direct Panels:**

- `/cockpit.html` - Main cockpit
- `/bank.html` - Portfolio manager
- `/markets.html` - Market overview
- `/engine.html` - Engine controls

______________________________________________________________________

## 🧪 TEST COMMANDS

```bash
# Test all endpoints
echo "Health:" && curl -s http://localhost:5000/health
echo "Cockpit:" && curl -s http://localhost:5000/api/cockpit | jq .ticker
echo "Portfolio:" && curl -s http://localhost:5000/api/portfolio | jq .nav
echo "Forecast:" && curl -s http://localhost:5000/predict/48h | jq .horizon_h
echo "News:" && curl -s http://localhost:5000/api/feeds/latest?limit=1 | jq .count
echo "Trade Card:" && curl -s http://localhost:5000/api/trade_card/WOLF | jq .confidence

# SSE Stream (10 seconds)
timeout 10 curl -N http://localhost:5000/api/cockpit/stream

# Add test position (requires auth token)
curl -X POST http://localhost:5000/api/position \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"qty": 100, "avg_cost": 1.20}'
```

______________________________________________________________________

## 🔄 RESTART COMMANDS

```bash
# Stop server
pkill -f uvicorn

# Start in LIVE mode
source /workspaces/GHOST/.venv/bin/activate
export SIM_MODE=0
export USE_PLACEHOLDERS=0
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload &

# Or use task
# (will auto-start with dependency installation)
```

______________________________________________________________________

## 📊 WORKING FEATURES

✅ **Backend:**

- Health checks
- Cockpit API (35 fields)
- Portfolio tracking
- 48h forecast (24 points)
- Watchlist (9 symbols)
- News feed (live Polygon)
- Trade card with AI
- Top movers
- SSE streaming

✅ **UI Panels:**

- Market status (partial)
- 48h forecast chart
- Portfolio overview
- Ghost score heatmap (1 symbol)
- Top movers
- Market outlook
- Live news (10 items)
- Watchlist
- Trade card
- Diagnostics stream

✅ **Integrations:**

- Telegram bot (GhostAlphaSniperBot)
- Yahoo Finance (rate-limited but functional)
- AlphaVantage (configured)
- Polygon API (active)

______________________________________________________________________

## ⚠️ BLOCKERS

❌ **Must Fix Before Production:**

1. Risk status endpoint 500 error
2. Price endpoint 404
3. Market status null fields

______________________________________________________________________

## 📈 METRICS

- **Pass Rate:** 87% (12/14 endpoints)
- **UI Coverage:** 90% (9/10 panels)
- **Uptime:** Stable
- **Data Sources:** All live
- **SSE Streaming:** Sustained

______________________________________________________________________

## 📚 REPORTS

- **MASTER_LIVE_TEST_REPORT.md** - Full test results
- **LIVE_MODE_VALIDATION.md** - Initial validation summary

______________________________________________________________________

**Last Updated:** October 6, 2025 13:45 UTC\
**Next Action:** Apply risk status fix and retest
