# Ghost Operations Guide

## Overview

This guide covers day-to-day operations, configuration tuning, troubleshooting, and
interpreting Ghost's forecast overlay and accuracy metrics.

______________________________________________________________________

## Starting Ghost

### Production Start

```bash
source .venv/bin/activate
uvicorn wolf_app:app --host 0.0.0.0 --port 5000
```

### Background Start (with logging)

```bash
source .venv/bin/activate
nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload > /tmp/ghost_server.log 2>&1 &
```

### Check Status

```bash
curl -s http://localhost:5000/health | jq
```

______________________________________________________________________

## Runtime Configuration

### View Current Config

```bash
curl -s http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" | jq
```

### Update Settings (No Restart Required)

```bash
curl -X POST http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "price_ttl_s": 60,
    "price_ttl_open_s": 15,
    "yahoo_first": 1,
    "reuters_feeds_on": 0,
    "diag_collapse_dupes": 1,
    "diag_ring_size": 200
  }' | jq
```

### Key Configuration Parameters

| Parameter | Default | Description | |-----------|---------|-------------| |
`price_ttl_s` | 60 | Cache duration for prices (market closed), seconds | |
`price_ttl_open_s` | 15 | Cache duration for prices (market open), seconds | |
`yahoo_first` | 1 | Try Yahoo Finance first (1=yes, 0=Polygon first) | |
`price_max_deviation_open` | 0.5 | Max allowed spread between providers (50%) | |
`reuters_feeds_on` | 1 | Enable Reuters news feeds (1=yes, 0=no) | |
`diag_collapse_dupes` | 1 | Deduplicate similar diagnostic events | | `diag_ring_size` |
200 | Number of diagnostic events to keep | | `forecast_pause_on_anomaly` | 1 | Pause
forecast during price anomalies |

______________________________________________________________________

## Position Management

### Import Positions

```bash
curl -X POST http://localhost:5000/api/positions/import \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "reset": true,
    "set_focus": true,
    "positions": [
      {
        "symbol": "WOLF",
        "qty": 8.41959051,
        "avg_cost": 359.40
      }
    ]
  }' | jq
```

### View Current Positions

```bash
curl -s http://localhost:5000/api/positions | jq
```

### Set Cash Balance

```bash
curl -X POST http://localhost:5000/api/bank/set_cash \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"stock": 1000.0}' | jq
```

______________________________________________________________________

## Price Overrides & Manual Mode

### Set Manual Price (for testing)

```bash
curl -X POST http://localhost:5000/debug/price_override \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "symbol": "WOLF",
    "price": 28.50,
    "ttl_s": 3600
  }' | jq
```

### Clear Manual Override

```bash
curl -X DELETE http://localhost:5000/debug/price_override \
  -H "Authorization: Bearer $GHOST_API_TOKEN" | jq
```

**Note:** When manual override is active:

- `prices.provider` will show `"manual"`
- `row.stale` will be `true`
- Forecast will pause (if `forecast_pause_on_anomaly` is enabled)

______________________________________________________________________

## Monitoring & Diagnostics

### Full Cockpit Snapshot

```bash
curl -s http://localhost:5000/api/cockpit | jq
```

### Price Diagnostics Only

```bash
curl -s http://localhost:5000/diagnostics/summary | jq '.price_diag'
```

### Live Events Stream (SSE)

```bash
curl -s --no-buffer http://localhost:5000/events | head -n 20
```

### Run Validation Script

```bash
./scripts/validate_ghost_simple.sh
```

______________________________________________________________________

## Understanding Forecast Overlay

### Forecast Structure

The `/api/cockpit` endpoint includes:

```json
{
  "forecast": {
    "ticker": "WOLF",
    "as_of": 1759429071,
    "horizon_h": 48,
    "step_h": 2,
    "points": [
      {
        "t": 1759436271,
        "price_mid": 359.4,
        "price_lo": 353.175,
        "price_hi": 365.625,
        "pnl_mid": 0.0,
        "pnl_lo": -52.41,
        "pnl_hi": 52.41
      }
      // ... 23 more points
    ],
    "summary": {
      "confidence": 60,
      "drift_daily_pct": 0.0,
      "pnl_48h_mid": 0.0
    }
  }
}
```

### Interpreting Forecast Fields

| Field | Meaning | |-------|---------| | `as_of` | Unix timestamp when forecast was
generated | | `horizon_h` | Forecast window (48 hours) | | `step_h` | Time between
forecast points (2 hours) | | `t` | Unix timestamp for this forecast point | |
`price_mid` | Mid-range price prediction | | `price_lo` | Lower confidence band
(pessimistic) | | `price_hi` | Upper confidence band (optimistic) | | `pnl_mid` |
Expected P&L at mid-range price | | `confidence` | Forecast confidence score (0-100) | |
`drift_daily_pct` | Expected daily price change (%) |

### Forecast Accuracy Metrics

Once Ghost has accumulated 24-48 hours of historical data, the `metrics` field will
populate:

```json
{
  "metrics": {
    "map": 8.5,     // Mean Absolute Percentage Error
    "rmse": 2.3,     // Root Mean Square Error
    "bias": -1.2,    // Prediction bias (negative = too pessimistic)
    "as_of": 1759500000
  }
}
```

**Interpreting Metrics:**

- **MAP < 10%**: Excellent forecast accuracy
- **MAP 10-20%**: Good accuracy
- **MAP > 20%**: Poor accuracy, model needs tuning
- **Bias < 0**: Ghost consistently under-predicts (too bearish)
- **Bias > 0**: Ghost consistently over-predicts (too bullish)
- **Bias ≈ 0**: Well-calibrated predictions

______________________________________________________________________

## Forecast Pause Conditions

Ghost automatically pauses forecasts when:

1. **Manual Price Override Active**

   - Reason: `"paused:manual_override"`
   - Action: Clear override to resume

2. **Price Anomaly Detected**

   - Reason: `"paused:price_anomaly"`
   - Triggers:
     - Price deviates >50% from previous close
     - Fresh Reuters news + extreme move
     - Provider spread exceeds threshold
   - Action: Wait for anomaly to clear, or disable pause via config

Check pause status:

```bash
curl -s http://localhost:5000/api/cockpit | jq '{
  forecast_enabled: .forecast_summary.enabled,
  forecast_note: .forecast.note
}'
```

______________________________________________________________________

## Troubleshooting

### Problem: No Live Price (Provider = "unavailable")

**Diagnosis:**

```bash
curl -s http://localhost:5000/diagnostics/summary | jq '.price_diag'
```

**Common Causes:**

1. **Rate Limiting**: Yahoo/AlphaVantage hit request limits

   - **Fix**: Wait 5-10 minutes, or toggle provider order

   ```bash
   curl -X POST http://localhost:5000/api/runtime/config \
     -H "Authorization: Bearer $GHOST_API_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"yahoo_first": 0}'
   ```

2. **Market Closed**: Outside trading hours

   - **Expected**: `prev_close` used instead
   - **Check**: `curl -s http://localhost:5000/api/cockpit | jq '.flags.market_open'`

3. **API Keys Missing**:

   - Check: `curl -s http://localhost:5000/api/secrets/health | jq`
   - **Fix**: Set environment variables:
     ```bash
     export POLYGON_API_KEY="your_key"
     export ALPHAVANTAGE_API_KEY="your_key"
     ```

### Problem: Forecast Metrics Still Null

**Reason:** Need 24-48h of historical forecast vs. actual data

**Check Progress:**

```bash
# Check how many forecasts have been recorded
curl -s http://localhost:5000/api/cockpit | jq '.forecast.as_of'
# If recent (within last hour), Ghost is generating forecasts
```

**Timeline:**

- **0-24h**: Metrics will be `null` (collecting data)
- **24-48h**: MAP/RMSE/Bias will begin populating
- **48h+**: Full accuracy tracking active

### Problem: Position Not Persisting Across Restart

**Diagnosis:**

```bash
# Before restart
curl -s http://localhost:5000/api/positions | jq '.positions[0].qty'

# Restart Ghost
pkill -9 -f uvicorn && sleep 2
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 &
sleep 5

# After restart
curl -s http://localhost:5000/api/positions | jq '.positions[0].qty'
```

**If quantity changed:**

1. Check persistence path:

   ```bash
   echo $WOLF_SQLITE_PATH  # Default: /data/wolf.db
   ls -lh /data/wolf.db
   ```

2. Check for write errors in logs:

   ```bash
   tail -n 100 /tmp/ghost_server.log | grep -i "persist"
   ```

3. Verify autosave worker started:

   ```bash
   curl -s http://localhost:5000/diagnostics/summary | jq '.events[] | select(.message | contains("autosave"))'
   ```

### Problem: UI Not Loading

**Check:**

```bash
curl -s http://localhost:5000/ | head -n 10
```

**If blank or error:**

1. Verify UI files exist:

   ```bash
   ls -lh /workspaces/GHOST/ui_dist/index.html
   ```

2. Check Static file mount:

   ```bash
   curl -s http://localhost:5000/static/ghost.css | head -n 5
   ```

3. Rebuild UI (if needed):

   ```bash
   # Copy from static/ to ui_dist/ if missing
   cp -r /workspaces/GHOST/static/* /workspaces/GHOST/ui_dist/
   ```

______________________________________________________________________

## Performance Tuning

### Reduce API Call Frequency

```bash
# Increase TTL for closed market hours
curl -X POST http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"price_ttl_s": 300}' | jq
```

### Speed Up Cockpit Response

```bash
# Disable Reuters (slow DNS lookups)
curl -X POST http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"reuters_feeds_on": 0}' | jq
```

### Reduce Diagnostic Noise

```bash
# Collapse duplicate events
curl -X POST http://localhost:5000/api/runtime/config \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"diag_collapse_dupes": 1, "diag_ring_size": 50}' | jq
```

______________________________________________________________________

## Backup & Restore

### Backup Positions and State

```bash
# Backup SQLite database
cp /data/wolf.db /data/wolf.db.backup_$(date +%Y%m%d_%H%M%S)

# Export positions as JSON
curl -s http://localhost:5000/api/positions > positions_backup.json
```

### Restore Positions

```bash
curl -X POST http://localhost:5000/api/positions/import \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d @positions_backup.json | jq
```

______________________________________________________________________

## Security

### Rotate API Token

```bash
# Generate new token
NEW_TOKEN=$(openssl rand -hex 12)

# Update environment
export GHOST_API_TOKEN="$NEW_TOKEN"

# Restart Ghost
pkill -f uvicorn && sleep 2
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 &
```

### Secure Deployment Checklist

- [ ] Set `GHOST_API_TOKEN` environment variable
- [ ] Restrict access to port 5000 (firewall/reverse proxy)
- [ ] Use HTTPS (TLS) via reverse proxy (Caddy, nginx)
- [ ] Enable rate limiting on API endpoints
- [ ] Rotate API keys for external providers (Polygon, AlphaVantage)
- [ ] Backup SQLite database daily

______________________________________________________________________

## Quick Reference

### Essential Endpoints

- **Health**: `GET /health`
- **Cockpit**: `GET /api/cockpit`
- **Positions**: `GET /api/positions`
- **Diagnostics**: `GET /diagnostics/summary`
- **Events Stream**: `GET /events` (SSE)
- **Runtime Config**: `GET/POST /api/runtime/config` (auth required)

### Environment Variables

```bash
GHOST_API_TOKEN=""           # Bearer token for auth endpoints
POLYGON_API_KEY=""           # Polygon.io API key
ALPHAVANTAGE_API_KEY=""      # AlphaVantage API key
WOLF_SQLITE_PATH="/data/wolf.db"  # Persistence database path
REDIS_URL=""                 # Optional Redis connection string
REUTERS_FEEDS_ON="0"         # Enable Reuters news (0=off, 1=on)
```

### Validation Commands

```bash
# Quick health check
curl -s http://localhost:5000/health | jq

# Full validation
./scripts/validate_ghost_simple.sh

# Check forecast status
curl -s http://localhost:5000/api/cockpit | jq '{
  forecast_enabled: .forecast_summary.enabled,
  points: (.forecast.points | length),
  confidence: .forecast_summary.confidence,
  metrics_available: (.metrics != null)
}'
```

______________________________________________________________________

## Support & Documentation

- **Production Status**: `GHOST_PRODUCTION_STATUS.md`
- **Quick Start**: `QUICK_START.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Requirements Verification**: `GHOST_REQUIREMENTS_VERIFICATION.md`

______________________________________________________________________

**Last Updated:** October 2, 2025\
**Ghost Version:** 0.3.0-production
