# 🚀 Ghost Quick Start Guide

## Prerequisites

✅ Python 3.11+ installed\
✅ Dependencies installed via `requirements.txt`\
✅ Environment variables configured in `secrets.env` (optional)

______________________________________________________________________

## 1. Start the Server

```bash
# Option 1: Using VS Code task
Run Task → "Run Ghost server (:5000)"

# Option 2: Manual start
source .venv/bin/activate
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload
```

Server will be available at: **[http://localhost:5000](http://localhost:5000)**

______________________________________________________________________

## 2. Configure Environment (Optional)

Create `secrets.env`:

```bash
# API Keys (optional for enhanced data)
export ALPHAVANTAGE_API_KEY="$(railway variables get ALPHAVANTAGE_API_KEY)"
export POLYGON_API_KEY="$(railway variables get POLYGON_API_KEY)"

# Security (recommended)
export GHOST_API_TOKEN="$(railway variables get GHOST_API_TOKEN)"

# Alerts (optional)
export TELEGRAM_BOT_TOKEN="$(railway variables get TELEGRAM_BOT_TOKEN)"
export TELEGRAM_CHAT_ID="$(railway variables get TELEGRAM_CHAT_ID)"

# Storage (optional, defaults to SQLite)
export REDIS_URL="redis://localhost:6379/0"
```

Load environment:

```bash
source secrets.env
```

______________________________________________________________________

## 3. Import Your Portfolio

### Via API (Recommended)

```bash
curl -X POST http://localhost:5000/api/positions/import \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "reset": true,
    "apply_to_cash": true,
    "positions": [
      {
        "symbol": "WOLF",
        "qty": 100,
        "price_paid": 120.00
      }
    ]
  }'
```

### Via CSV Upload (UI)

1. Navigate to **[http://localhost:5000/cockpit](http://localhost:5000/cockpit)**
2. Click "Import Positions"
3. Paste CSV or JSON
4. Click "Apply"

**CSV Format**:

```csv
symbol,qty,price_paid,market
WOLF,100,120.00,stock
AAPL,50,180.00,stock
```

______________________________________________________________________

## 4. Verify Live Data

### Check Price Provider

```bash
curl http://localhost:5000/api/cockpit | jq '.prices'
```

**Expected Output**:

```json
{
  "provider": "yahoo",
  "price": 123.45,
  "prev_close": 122.50,
  "change_pct": 0.77
}
```

### Check Diagnostics

```bash
curl http://localhost:5000/diagnostics/summary | jq '.price_diag'
```

**Expected Output**:

```json
{
  "market_open": true,
  "last_fetch_provider": "yahoo",
  "last_fetch_latency_ms": 142,
  "last_good_price_ts": 1696262400,
  "fallback_reason": null,
  "provider_spread": 0.002,
  "quorum_ok": true
}
```

______________________________________________________________________

## 5. Access the Cockpit

Open in browser: **[http://localhost:5000/cockpit](http://localhost:5000/cockpit)**

### Key Panels

1. **Portfolio Overview**

   - Live NAV, PnL, positions
   - Real-time price updates
   - Market value calculations

2. **Forecast Overlay (48h)**

   - Predicted vs actual price lines
   - MAP, RMSE, Bias metrics
   - Auto-pause on anomalies

3. **Diagnostics**

   - Provider status
   - Recent events
   - Price diagnostics
   - Circuit breaker states

4. **Admin Toggles**

   - Live config changes
   - TTL adjustments
   - Provider preferences

______________________________________________________________________

## 6. Runtime Configuration

### View Current Config

```bash
curl http://localhost:5000/api/runtime/config
```

### Update Config (No Restart Required)

```bash
curl -X POST http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "price_ttl_s": 60,
    "price_ttl_open_s": 45,
    "yahoo_first": 1,
    "price_max_deviation_open": 0.08
  }'
```

**Configurable Parameters**:

- `price_ttl_s` - Cache TTL (market closed)
- `price_ttl_open_s` - Cache TTL (market open)
- `news_ttl_s` - News cache TTL
- `yahoo_first` - Provider priority (0=AlphaVantage first, 1=Yahoo first)
- `price_max_deviation_open` - Quorum spread threshold

______________________________________________________________________

## 7. Verify Persistence

### Test Position Persistence

```bash
# 1. Import positions
curl -X POST http://localhost:5000/api/positions/import \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"positions": [{"symbol": "WOLF", "qty": 100, "price_paid": 120.00}]}'

# 2. Check state file
cat /data/wolf_state.json
# or
sqlite3 /data/wolf.db "SELECT value FROM state WHERE key='position';" | jq '.'

# 3. Restart server
pkill -f "uvicorn wolf_app"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 &

# 4. Verify positions restored
sleep 5
curl http://localhost:5000/api/cockpit | jq '.portfolio.rows'
```

**Expected**: Positions array intact after restart.

______________________________________________________________________

## 8. Monitor Forecast Accuracy

### Get Forecast Overlay

```bash
curl http://localhost:5000/api/forecast/overlay?symbol=WOLF | jq '.'
```

**Expected Output**:

```json
{
  "enabled": true,
  "symbol": "WOLF",
  "coverage_h": 48,
  "path_predicted": {
    "mid": [
      {"t": 1696262400, "p": 123.45},
      {"t": 1696269600, "p": 124.12}
    ]
  },
  "path_actual": [
    {"t": 1696262400, "p": 123.50},
    {"t": 1696269600, "p": 124.00}
  ],
  "metrics": {
    "map": 1.85,
    "rmse": 0.98,
    "bias": 0.12,
    "accrual_pct": 92.3
  }
}
```

______________________________________________________________________

## 9. Common Issues & Solutions

### Issue: Prices showing "prev-close" or "unavailable"

**Cause**: All live providers failed

**Solution**:

1. Check diagnostics:
   `curl http://localhost:5000/diagnostics/summary | jq '.price_diag'`
2. Verify API keys in environment
3. Check provider circuit breakers
4. Review recent events for errors

**Fix**:

```bash
# Reset circuit breakers
curl -X POST http://localhost:5000/debug/breaker_reset \
  -H "Authorization: Bearer $GHOST_API_TOKEN"

# Force price refresh
curl http://localhost:5000/api/cockpit
```

### Issue: Positions reset to zero after restart

**Cause**: Persistence not working

**Solution**:

1. Check `WOLF_PERSIST_MODE` env var (should be `auto`, `redis`, `sqlite`, or `file`)
2. Verify storage path writable: `/data/wolf.db` or `/data/wolf_state.json`
3. Check startup logs for `persist_load_failed` error

**Fix**:

```bash
# Ensure data directory exists
mkdir -p /data
chmod 755 /data

# Set explicit persistence mode
export WOLF_PERSIST_MODE=sqlite
export WOLF_SQLITE_PATH=/data/wolf.db
```

### Issue: Forecast showing "paused (anomaly)"

**Cause**: Price anomaly detected (corporate action, extreme move, provider spread)

**Solution**:

1. Check diagnostics for anomaly flags
2. Wait for price to normalize (forecast auto-resumes)
3. Override manually in UI (Resume overlay button)

**Fix**:

```bash
# Check anomaly status
curl http://localhost:5000/api/cockpit | jq '.flags'

# Expected after recovery:
# {
#   "price_anomaly": false,
#   "corp_action_suspected": false
# }
```

### Issue: High MAP/RMSE in forecast metrics

**Cause**: Market volatility or model drift

**Solution**:

1. Review `band_widen_factor` setting (increase for wider bands)
2. Adjust `PRED_SIGMA_DAILY` for volatility
3. Check actual vs predicted lines in UI

**Fix**:

```bash
# Increase prediction bands
curl -X POST http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"band_widen_factor": 1.5}'
```

______________________________________________________________________

## 10. Health Checks

### System Health

```bash
curl http://localhost:5000/health
```

**Expected**: `{"ok": true}`

### Price Health

```bash
curl http://localhost:5000/api/cockpit | jq '.status'
```

**Expected**:

```json
{
  "ok": true,
  "active": true,
  "feeds": {
    "stocks": true,
    "crypto": false,
    "news": true,
    "telegram": true,
    "prices": true
  }
}
```

### API Secrets Health

```bash
curl http://localhost:5000/api/secrets/health
```

**Expected**:

```json
{
  "present": {
    "GHOST_API_TOKEN": true,
    "ALPHAVANTAGE_API_KEY": true,
    "POLYGON_API_KEY": true,
    "TELEGRAM_BOT_TOKEN": true,
    "TELEGRAM_CHAT_ID": true,
    "REDIS_URL": false
  },
  "missing": ["REDIS_URL"]
}
```

______________________________________________________________________

## 11. Testing Provider Chain

### Test Yahoo (Primary)

```bash
curl "http://localhost:5000/debug/yahoo?symbol=WOLF"
```

### Test AlphaVantage

```bash
curl "http://localhost:5000/debug/alphavantage?symbol=WOLF"
```

### Test Polygon

```bash
curl "http://localhost:5000/debug/polygon?symbol=WOLF"
```

### Test Quorum Consensus

```bash
# Get all provider quotes
curl http://localhost:5000/diagnostics/summary | jq '.price_diag.providers'

# Expected: Array of [provider, price] pairs
# [
#   ["yahoo", 123.45],
#   ["alphavantage", 123.42],
#   ["yfinance", 123.47]
# ]
```

______________________________________________________________________

## 12. Backup & Recovery

### Backup State

```bash
# SQLite
cp /data/wolf.db /backup/wolf.db.$(date +%Y%m%d)

# File
cp /data/wolf_state.json /backup/wolf_state.json.$(date +%Y%m%d)
```

### Restore State

```bash
# SQLite
cp /backup/wolf.db.20251002 /data/wolf.db

# File
cp /backup/wolf_state.json.20251002 /data/wolf_state.json

# Restart server
pkill -f "uvicorn wolf_app"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload
```

______________________________________________________________________

## 13. Performance Tuning

### Reduce API Calls

```bash
curl -X POST http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "price_ttl_s": 120,
    "price_ttl_open_s": 60,
    "news_ttl_s": 600
  }'
```

### Enable Auto-Save

```bash
export WOLF_AUTOSAVE_S=60  # Save every 60 seconds
```

### Optimize Provider Order

```bash
# If Yahoo is fastest
export PRICE_YAHOO_FIRST=1

# If AlphaVantage is fastest
export PRICE_YAHOO_FIRST=0
```

______________________________________________________________________

## 14. Logs & Monitoring

### View Recent Events

```bash
curl http://localhost:5000/diagnostics/summary | jq '.events'
```

### Prometheus Metrics

```bash
curl http://localhost:5000/metrics
```

**Key Metrics**:

- `ghost_price_calls_total` - Total price fetches
- `ghost_price_errors_total` - Price fetch errors
- `ghost_alerts_sent_total` - Alert delivery count
- `ghost_forecast_calls_total` - Forecast generation count

______________________________________________________________________

## 15. API Reference

### Core Endpoints

| Endpoint | Method | Description | |----------|--------|-------------| | `/api/cockpit`
| GET | Full cockpit snapshot | | `/api/forecast/overlay` | GET | Prediction vs actual
overlay | | `/diagnostics/summary` | GET | System diagnostics | | `/api/runtime/config`
| GET/POST | Runtime configuration | | `/api/positions/import` | POST | Import positions
| | `/api/positions/clear` | POST | Clear all positions | | `/health` | GET | System
health check |

### Debug Endpoints

| Endpoint | Method | Description | |----------|--------|-------------| | `/debug/yahoo`
| GET | Test Yahoo provider | | `/debug/alphavantage` | GET | Test AlphaVantage provider
| | `/debug/polygon` | GET | Test Polygon provider | | `/debug/breaker_reset` | POST |
Reset circuit breakers |

______________________________________________________________________

## Support & Documentation

- **Full Requirements**: See `GHOST_REQUIREMENTS_VERIFICATION.md`
- **Runtime Toggles**: See `docs/runtime_toggles.md`
- **Observability**: See `docs/observability.md`
- **README**: See `README.md`

______________________________________________________________________

## ✅ Verification Checklist

Before going live, verify:

- [ ] Server starts without errors
- [ ] Price fetches from live providers
- [ ] Positions persist across restart
- [ ] Cash balances persist across restart
- [ ] NAV calculation matches manual calculation
- [ ] PnL calculation matches broker
- [ ] Forecast overlay displays two lines
- [ ] Metrics (MAP/RMSE/Bias) update
- [ ] Diagnostics show clear provider status
- [ ] Runtime config changes apply immediately
- [ ] No random/placeholder values in UI
- [ ] All numbers traceable to sources

**Status**: 🟢 Production-Ready
