# 🎯 Ghost: Production-Ready Trading AI

**Version**: 1.0 (October 2, 2025)\
**Status**: ✅ Production-Ready\
**Compliance**: All 7 Requirements Met

______________________________________________________________________

## What Is Ghost

Ghost is a **live-data, stateful trading AI**for real-time portfolio oversight and
price forecasting. It provides:

-**Real-time price tracking**from multiple live providers (Yahoo, Polygon,

  AlphaVantage)

-**Multi-position portfolio management**with persistent state
-**48-hour price forecasting**with accuracy tracking (MAP, RMSE, Bias)
-**Live configuration control**without server restarts
-**Complete transparency**- every number is traceable to real sources**Zero simulation. Zero placeholders.
Zero randomness.**______________________________________________________________________

## Quick Start (3 Steps)

### 1. Start Server

```bash
source .venv/bin/activate
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload

```text

### 2. Import Portfolio

```bash

curl -X POST <<<<<http://localhost:5000/api/positions/import>>>>> \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [
      {"symbol": "WOLF", "qty": 100, "price_paid": 120.00}
    ]
  }'

```text

### 3. Open Cockpit

```text

<<<<<http://localhost:5000/cockpit>>>>>

```text**Done!**Ghost is now tracking your portfolio with live prices.

______________________________________________________________________

## Key Features

### ✅ Live Data Only

- Fetches from Yahoo, Polygon, AlphaVantage, yfinance
- Consensus quorum (≥2 providers must agree)
- Graceful fallback to prev_close if all fail
- Never generates fake/placeholder values


### ✅ Accurate Math

- `NAV = Cash + Σ(Market Values)`
- `PnL = (Current - Entry) × Quantity`
- Real-time updates every refresh


### ✅ Full Persistence

- Positions saved to Redis/SQLite/File
- Cash balances preserved across restarts
- Auto-save on changes (optional)
- Import once, remember forever


### ✅ Forecast Accuracy

- Predicted vs Actual price overlay
- MAP (Mean Absolute % Error)
- RMSE (Root Mean Squared Error)
- Bias (systematic over/under prediction)
- Updated continuously


### ✅ Live Configuration

- Change TTLs, provider preferences, thresholds
- No server restart required
- Takes effect immediately


### ✅ Complete Transparency

- Every price has explicit provider attribution
- Diagnostics show why fallbacks happen
- Event log tracks all operations
- Prometheus metrics for monitoring


______________________________________________________________________

## What Was Fixed (October 2, 2025)

### 1. Price Fallback Logic ✅**Before**: Prices stuck at stale/prev_close despite market open\

**After**: Attempts ALL providers before fallback, tracks diagnostics

### 2. Position Persistence ✅

**Before**: Portfolio reset to Qty 0 after restart\
**After**: Full positions array + cash balances persist

### 3. Forecast Auto-Resume ✅

**Before**: Forecast stayed paused after recovery\
**After**: Auto-resumes when price normalizes

### 4. Accuracy Tracking ✅

**Before**: No way to verify forecast quality\
**After**: MAP/RMSE/Bias metrics in UI

### 5. Diagnostics Panel ✅

**Before**: Unclear why degraded\
**After**: Shows provider, latency, fallback reason

______________________________________________________________________

## Documentation

| Document | Purpose | |----------|---------| | **QUICK_START.md**| Step-by-step setup
guide | |**IMPLEMENTATION_SUMMARY.md**| Technical details of all fixes | |**GHOST_REQUIREMENTS_VERIFICATION.md**| Full
requirement compliance proof | |**validate_ghost.sh**| Automated validation script | |**README.md**(original) | API
reference and features |

______________________________________________________________________

## Key Endpoints

### Main Dashboard

```bash

GET /api/cockpit

```text

Returns: Prices, portfolio, forecast, news, diagnostics

### Forecast Accuracy

```bash

GET /api/forecast/overlay?symbol=WOLF

```text

Returns: Predicted vs actual paths + metrics

### Diagnostics

```bash

GET /diagnostics/summary

```text

Returns: Health, events, provider status

### Runtime Config

```bash

GET /api/runtime/config
POST /api/runtime/config

```text

Get/update settings without restart

### Position Management

```bash

POST /api/positions/import
POST /api/positions/clear
GET /api/portfolio

```text

______________________________________________________________________

## Validation

Run automated validation:

```bash

./validate_ghost.sh

```text**Expected Output**: ✅ All checks passed!

**Validates**:

1. Live data fetching
2. Price & portfolio math
3. Persistence (state files exist)
4. Forecast overlay working
5. UI structure correct
6. Runtime config available
7. No randomness (provider tracking)


______________________________________________________________________

## Environment Setup

### Required

```bash

export GHOST_API_TOKEN="$(railway variables get GHOST_API_TOKEN)"

```text

### Optional (Enhanced Data)

```bash

export ALPHAVANTAGE_API_KEY="$(railway variables get ALPHAVANTAGE_API_KEY)"
export POLYGON_API_KEY="$(railway variables get POLYGON_API_KEY)"

```text

### Optional (Alerts)

```bash

export TELEGRAM_BOT_TOKEN="$(railway variables get TELEGRAM_BOT_TOKEN)"
export TELEGRAM_CHAT_ID="$(railway variables get TELEGRAM_CHAT_ID)"

```text

### Optional (Storage)

```bash

export REDIS_URL="redis://localhost:6379/0"
export WOLF_PERSIST_MODE="auto"  # auto|redis|sqlite|file

```text

______________________________________________________________________

## Architecture

### Price Provider Chain

```text

get_wolf_price()
  ↓
  ├─ Cache hit? → Return cached
  ↓
  ├─ Yahoo Finance
  ├─ AlphaVantage (if key)
  ├─ Polygon.io (if key)
  ├─ yfinance
  ↓
  ├─ Quorum (≥2 agree)? → Consensus
  ↓
  └─ All fail? → prev_close fallback

```text

### Persistence Flow

```text

_persist_save()
  ↓
  ├─ Redis (if REDIS_URL)
  ├─ SQLite (/data/wolf.db)
  └─ File (/data/wolf_state.json)

_persist_load() @ startup
  ↓
  └─ Restores positions + cash

```text

### Forecast Accuracy

```text

Background Tasks:
  ├─ _auto_record_forecast() → Save predictions
  ├─ _auto_record_actual_prices() → Poll live
  └─ _auto_score_forecasts() → Compute metrics

/api/forecast/overlay
  ↓
  └─ Returns: predicted, actual, metrics

```text

______________________________________________________________________

## Monitoring

### Prometheus Metrics

```bash

curl <<<<<http://localhost:5000/metrics>>>>>

```text

**Key Metrics**:

- `ghost_price_calls_total` - Price fetches
- `ghost_price_errors_total` - Fetch failures
- `ghost_forecast_calls_total` - Forecast generations
- `ghost_alerts_sent_total` - Alert deliveries


### Event Log

```bash

curl <<<<<http://localhost:5000/diagnostics/summary>>>>> | jq '.events'

```text

### Price Diagnostics

```bash

curl <<<<<http://localhost:5000/diagnostics/summary>>>>> | jq '.price_diag'

```text

______________________________________________________________________

## Common Issues

### Prices showing "prev-close"

**Cause**: All providers failed or market closed\
**Fix**: Check diagnostics for fallback_reason

### Positions reset to zero

**Cause**: Persistence not working\
**Fix**: Check WOLF_PERSIST_MODE and storage paths

### Forecast paused

**Cause**: Anomaly detected\
**Fix**: Wait for price normalization or manual override

### High MAP/RMSE

**Cause**: Market volatility\
**Fix**: Adjust band_widen_factor in runtime config

______________________________________________________________________

## Compliance Status

| Requirement | Status | Evidence | |-------------|--------|----------| | 1. Live data
only | ✅ | get_wolf_price() Lines 2360-2450 | | 2. Correct math | ✅ | NAV/PnL formulas
Lines 5264-5266 | | 3. Persistence | ✅ | \_persist_save/load Lines 2739-2963 | | 4.
Accuracy tracking | ✅ | /api/forecast/overlay Lines 3883-3974 | | 5. Proper UI | ✅ |
Cockpit panels w/ diagnostics | | 6. Runtime config | ✅ | /api/runtime/config Lines
4861-4928 | | 7. No randomness | ✅ | All data traceable, no random() |

______________________________________________________________________

## Production Checklist

Before going live:

- [ ] Server starts without errors
- [ ] API token configured (`GHOST_API_TOKEN`)
- [ ] Price providers working (check diagnostics)
- [ ] Portfolio imported successfully
- [ ] Positions persist across restart
- [ ] NAV matches broker/manual calculation
- [ ] Forecast overlay displays metrics
- [ ] Runtime config changes take effect
- [ ] Validation script passes (`./validate_ghost.sh`)


______________________________________________________________________

## Support

**Documentation**:

- QUICK_START.md - Setup guide
- IMPLEMENTATION_SUMMARY.md - Technical details
- GHOST_REQUIREMENTS_VERIFICATION.md - Full spec


**Diagnostics**:

- GET /diagnostics/summary - System health
- GET /api/cockpit - Full snapshot
- ./validate_ghost.sh - Automated checks


______________________________________________________________________

## Version History

**v1.0**(October 2, 2025)

- ✅ Fixed price provider fallback
- ✅ Fixed position persistence
- ✅ Verified forecast auto-resume
- ✅ Verified accuracy tracking
- ✅ Enhanced diagnostics panel
- ✅ All 7 requirements met


______________________________________________________________________

## Status: 🟢 Production-Ready**Ghost is a live, real-time, stateful trading AI.**- No placeholders

- No fake values
- No silent resets
- Every number ties back to real data
- Predictions testable against reality**Ready for live production oversight of real portfolios.** 🚀


______________________________________________________________________

## License

See LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.

## Author

Ghost WOLF-only v1.0 - Production trading AI for real-time oversight.
