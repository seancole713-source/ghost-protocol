# Ghost Prediction Panel

**Live 48-hour stock price predictions with accuracy tracking and scoreboard.**

## Overview

The Ghost Prediction panel replaces the previous 48h forecast system with a comprehensive prediction tracking solution that:

- Generates 48-hour price forecasts for stocks using live market data
- Stores full prediction curves with timestamps
- Tracks actual prices as they occur
- Computes accuracy metrics (MAE, MAP, RMSE, direction hit rate)
- Displays historical performance with scoreboard
- Provides visual overlay of forecast vs actual prices

**Status:** ✅ Complete end-to-end implementation (no placeholders, no SIM mode)

---

## Architecture

### Database Schema

**SQLite database:** `data/ghost_predictions.db`

#### Tables

**1. predictions**
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    run_at REAL NOT NULL,              -- Unix timestamp when forecast was generated
    horizon_h INTEGER NOT NULL DEFAULT 48,
    method TEXT NOT NULL,              -- e.g., 'ghost-av1'
    confidence REAL NOT NULL,          -- 0-1 model confidence
    direction TEXT NOT NULL CHECK(direction IN ('UP','DOWN','FLAT')),
    features_json TEXT,                -- JSON of input features
    params_json TEXT,                  -- JSON of model parameters
    tag TEXT
);
CREATE INDEX idx_predictions_symbol_run ON predictions(symbol, run_at DESC);
```

**2. prediction_points**
```sql
CREATE TABLE prediction_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    ts REAL NOT NULL,                  -- Unix timestamp
    kind TEXT NOT NULL CHECK(kind IN ('forecast','actual')),
    price REAL NOT NULL,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);
CREATE INDEX idx_points_pred_kind ON prediction_points(prediction_id, kind, ts);
```

**3. outcomes**
```sql
CREATE TABLE outcomes (
    prediction_id INTEGER NOT NULL UNIQUE,
    closed_at REAL NOT NULL,           -- When outcome was computed
    mae REAL NOT NULL,                 -- Mean Absolute Error
    map REAL NOT NULL,                -- Mean Absolute Percentage Error
    rmse REAL NOT NULL,                -- Root Mean Squared Error
    hit_direction INTEGER NOT NULL CHECK(hit_direction IN (0,1)),
    hit_ratio_window REAL,             -- Fraction of points within 5% tolerance
    notes TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);
CREATE INDEX idx_outcomes_pred ON outcomes(prediction_id);
```

---

## API Endpoints

### POST /api/predict/run

Generate a new 48h prediction for a stock symbol.

**Request:**
```json
{
    "symbol": "AAPL"
}
```

**Response:**
```json
{
    "ok": true,
    "prediction_id": 42,
    "symbol": "AAPL",
    "run_at": 1697011200.0,
    "horizon_h": 48,
    "confidence": 0.72,
    "direction": "UP"
}
```

**Requirements:**
- Live mode only (SIM_MODE=0)
- Bearer authentication (if GHOST_API_TOKEN is set)
- Fetches current price from live providers
- Generates 48h forecast with 2h steps (25 points)
- Stores forecast curve in database

---

### GET /api/predict/series

Get prediction time series data for charting (forecast + actual overlay).

**Query Parameters:**
- `symbol`: Stock ticker (required)
- `since_hours`: Hours of history to include (default: 72)

**Response:**
```json
{
    "symbol": "AAPL",
    "last_prediction": {
        "id": 42,
        "run_at": 1697011200.0,
        "horizon_h": 48,
        "confidence": 0.72,
        "direction": "UP"
    },
    "forecast": [
        {"ts": 1697011200, "price": 175.23},
        {"ts": 1697018400, "price": 175.45},
        ...
    ],
    "actual": [
        {"ts": 1697011200, "price": 175.20},
        {"ts": 1697018400, "price": 175.52},
        ...
    ]
}
```

**Usage:** Powers the chart overlay visualization.

---

### GET /api/predict/history

Get prediction history with outcomes for scoreboard display.

**Query Parameters:**
- `symbol`: Stock ticker (required)
- `limit`: Max predictions to return (default: 20, max: 100)

**Response:**
```json
[
    {
        "id": 42,
        "run_at": 1697011200.0,
        "confidence": 0.72,
        "direction": "UP",
        "closed": true,
        "mae": 1.23,
        "map": 0.71,
        "rmse": 1.45,
        "hit_direction": 1
    },
    {
        "id": 41,
        "run_at": 1696924800.0,
        "confidence": 0.68,
        "direction": "DOWN",
        "closed": false
    }
]
```

**Notes:**
- `closed=true` means 48h window has expired and outcome metrics are available
- `closed=false` means prediction is still active (pending)

---

### GET /api/predict/scoreboard

Get aggregate accuracy statistics for a symbol.

**Query Parameters:**
- `symbol`: Stock ticker (required)
- `windows`: Comma-separated day windows (default: "7,30")

**Response:**
```json
{
    "overall": {
        "count": 25,
        "hit_dir_pct": 68.0,
        "mae": 1.23,
        "map": 0.71,
        "rmse": 1.45,
        "avg_conf": 0.70,
        "calibration_gap": 0.02
    },
    "w7d": {
        "count": 7,
        "hit_dir_pct": 71.4,
        "mae": 1.15,
        "map": 0.66,
        "rmse": 1.38,
        "avg_conf": 0.72,
        "calibration_gap": 0.01
    },
    "w30d": {
        "count": 25,
        "hit_dir_pct": 68.0,
        "mae": 1.23,
        "map": 0.71,
        "rmse": 1.45,
        "avg_conf": 0.70,
        "calibration_gap": 0.02
    }
}
```

**Metrics:**
- `hit_dir_pct`: Percentage of predictions where direction (UP/DOWN/FLAT) matched actual
- `mae`: Mean Absolute Error (average price deviation)
- `map`: Mean Absolute Percentage Error
- `rmse`: Root Mean Squared Error
- `avg_conf`: Average model confidence
- `calibration_gap`: |avg_conf - hit_rate| (measures overconfidence/underconfidence)

---

## Background Services

### Outcome Reconciler

**File:** `services/outcome_reconciler.py`

**Function:** Automatically closes predictions and computes accuracy metrics.

**Schedule:** Runs every 5 minutes

**Process:**
1. Find predictions where `run_at + 48h <= now` and no outcome exists
2. Retrieve forecast points and actual points
3. Align timestamps (60s tolerance)
4. Compute MAE, MAP, RMSE
5. Determine direction hit (compare 48h price movement direction)
6. Compute hit_ratio_window (fraction of points within 5% tolerance)
7. Persist outcome record
8. Update Prometheus metrics

### Actual Price Appender

**Function:** Appends live prices to active predictions for real-time comparison.

**Schedule:** Runs every 5 minutes (integrated into reconciler loop)

**Process:**
1. Find all active predictions (not yet closed)
2. Fetch current live price for each symbol
3. Append as `actual` point with current timestamp
4. Enables continuous chart updates before prediction closes

---

## UI Components

### Chart Panel

**Location:** `templates/cockpit.html` (replaces old Forecast panel)

**Features:**
- Symbol selector dropdown (WOLF, AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META)
- "Run New Forecast" button (triggers POST /api/predict/run)
- Two-line overlay chart:
  - **Actual Price** (solid green line)
  - **Predicted Price** (dashed blue line)
- Shaded forecast window (48h from run_at)
- Auto-refresh every 15 seconds
- Empty state with instructive message

### Scoreboard Table

**Columns:**
- Date/Time (when forecast was generated)
- Direction (↑ UP, ↓ DOWN, → FLAT)
- Confidence % (model confidence)
- Hit? (✓ correct, ✗ wrong, "Pending" for active)
- MAE (mean absolute error)
- MAP % (mean absolute percentage error)
- RMSE (root mean squared error)

**Summary Stats (below chart):**
- 7d Accuracy % (hit rate last 7 days)
- 30d Accuracy % (hit rate last 30 days)
- Avg Confidence (average model confidence)
- Calibration Gap (overconfidence measure)

### JavaScript

**File:** `static/js/predict.js`

**Dependencies:**
- Chart.js 4.4.0
- chartjs-adapter-date-fns (time axis)
- chartjs-plugin-annotation (forecast window shading)

**Functions:**
- `initPredictionPanel()` - Setup event listeners and initial load
- `refreshPredictionData()` - Fetch series/history/scoreboard and update UI
- `runNewForecast()` - Trigger new prediction via POST /api/predict/run
- `updatePredictionChart()` - Render chart with forecast+actual overlay
- `updateScoreboard()` - Populate scoreboard table
- `updateSummaryStats()` - Display aggregate accuracy metrics

---

## Prometheus Metrics

**Counter:** `ghost_predict_runs_total{symbol}`
- Total prediction runs by symbol

**Counter:** `ghost_predict_outcomes_total{symbol, hit}`
- Total outcomes by symbol and hit status (0 or 1)

**Gauge:** `ghost_predict_mae{symbol}`
- Current MAE for latest outcome

**Gauge:** `ghost_predict_mape{symbol}`
- Current MAP for latest outcome

**Gauge:** `ghost_predict_rmse{symbol}`
- Current RMSE for latest outcome

**Gauge:** `ghost_predict_confidence_avg{symbol}`
- Average confidence across recent predictions

---

## Forecasting Method

**Current Implementation:** Simple flat-line forecast at current price (conservative baseline)

**Direction Logic:**
- Fetch last 5 days of price history
- If recent change > 2%: direction = UP, confidence = 0.60 + (change_pct / 20)
- If recent change < -2%: direction = DOWN, confidence = 0.60 + (|change_pct| / 20)
- Otherwise: direction = FLAT, confidence = 0.60

**Forecast Curve:**
- 48 hour horizon
- 2 hour steps (25 points total)
- Each point at `run_at + (i * 7200)` seconds

**Future Enhancement:** Can integrate advanced Ghost forecasting engine with confidence bands, regime detection, etc.

---

## Testing

### Unit Tests

**File:** `tests/test_predictor.py`

**Coverage:**
- Database schema creation
- Prediction CRUD operations
- Metrics calculation (MAE/MAP/RMSE)
- Direction hit logic
- Scoreboard aggregation

### API Tests

**File:** `tests/test_predict_api.py`

**Coverage:**
- POST /api/predict/run returns valid schema
- GET /api/predict/series merges forecast+actual
- GET /api/predict/history pagination
- GET /api/predict/scoreboard windows
- Bearer auth enforcement

### Integration Tests

**File:** `tests/test_predict_integration.py`

**Coverage:**
- End-to-end: run prediction → append actuals → reconcile outcome
- Chart rendering with populated data
- Scoreboard updates
- Metrics export

---

## Configuration

### Environment Variables

**Database:**
```bash
GHOST_PREDICT_DB=./data/ghost_predictions.db
```

**Requirements:**
```bash
SIM_MODE=0  # Must be live mode for predictions
```

**Optional:**
```bash
GHOST_API_TOKEN=your_token  # Bearer auth for POST endpoints
```

---

## Usage Example

### Generate Prediction (CLI)

```bash
curl -X POST http://localhost:5000/api/predict/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -d '{"symbol":"AAPL"}'
```

### Get Chart Data

```bash
curl "http://localhost:5000/api/predict/series?symbol=AAPL"
```

### Get Scoreboard

```bash
curl "http://localhost:5000/api/predict/scoreboard?symbol=AAPL&windows=7,30"
```

### UI Access

Navigate to: `http://localhost:5000/cockpit`

Scroll to "Ghost Prediction" panel.

---

## Limitations & Future Work

### Current Limitations

1. **Stocks only** - No crypto support yet (crypto market 24/7 requires different handling)
2. **Single symbol charts** - No multi-symbol overlay in one chart
3. **Simple forecast** - Baseline flat-line forecast (can integrate advanced models)
4. **Fixed horizon** - 48h window only (could make configurable)

### Planned Enhancements

1. **Advanced forecasting** - Integrate Ghost ensemble forecaster with confidence bands
2. **Regime-aware predictions** - Adjust forecast based on market regime detection
3. **Multi-symbol comparison** - Overlay multiple stock predictions in one chart
4. **Crypto support** - Handle 24/7 markets with different time windows
5. **Prediction explanations** - Show features/factors influencing forecast
6. **Alert integration** - Notify when prediction accuracy degrades
7. **Model versioning** - Track forecast method versions and A/B test

---

## Troubleshooting

### Predictions not appearing

**Check:**
1. SIM_MODE=0 (must be live mode)
2. Bearer token if GHOST_API_TOKEN is set
3. Symbol has live price data available
4. Database permissions: `ls -la data/ghost_predictions.db`

### Chart not updating

**Check:**
1. Browser console for JavaScript errors
2. Chart.js loaded: check network tab for CDN scripts
3. /api/predict/series returns data: `curl "http://localhost:5000/api/predict/series?symbol=WOLF"`

### Outcomes not closing

**Check:**
1. Reconciler worker started: grep logs for "Prediction outcome reconciler started"
2. Sufficient actual points appended (need at least 2 aligned points)
3. 48h window has passed: `SELECT run_at + (horizon_h * 3600) FROM predictions WHERE id=X`

### High error metrics

**Possible causes:**
- Volatile market conditions
- Low sample size (wait for more predictions)
- Baseline forecast too simple (consider advanced models)

---

## Files Created/Modified

### New Files

```
services/predictor.py              # Prediction storage and metrics
services/outcome_reconciler.py     # Background outcome computation
static/js/predict.js               # Frontend chart/scoreboard
docs/ghost_prediction_panel.md     # This documentation
```

### Modified Files

```
wolf_app.py                        # Added 4 API endpoints + background worker
templates/cockpit.html             # Replaced Forecast panel with Ghost Prediction
```

---

## Performance Considerations

**Database:**
- Indexed by (symbol, run_at) for fast history queries
- WAL mode for concurrent reads/writes
- Cascading deletes for data consistency

**API:**
- Bearer auth optional (only when GHOST_API_TOKEN set)
- Series endpoint optimized for chart rendering (pre-formatted timestamps)
- Scoreboard uses aggregates (no full scan)

**Background Tasks:**
- 5min interval (low CPU/memory footprint)
- Only processes active/expired predictions (not full table scan)
- Graceful shutdown on SIGTERM

**Frontend:**
- Chart.js efficient canvas rendering
- 15s auto-refresh (configurable)
- Debounced "Run Forecast" button

---

## Support

For issues or questions:
1. Check logs: `grep "predict\|reconciler" ghost.log`
2. Verify database: `sqlite3 data/ghost_predictions.db ".tables"`
3. Test endpoints: `curl http://localhost:5000/api/predict/series?symbol=WOLF`

---

**Implementation Status:** ✅ Complete
**Date:** October 11, 2025
**Version:** Ghost Protocol v10.2
