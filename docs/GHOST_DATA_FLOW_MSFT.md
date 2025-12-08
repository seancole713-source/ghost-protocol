# GHOST DATA FLOW – MSFT END-TO-END TRACE

**Purpose**: Map the complete journey of a prediction for MSFT from price fetch to Telegram alert.

---

## 1. PRICE & SIGNAL PROVIDERS

**Entry Point**: User triggers `/api/predict/run?symbol=MSFT` or auto-prediction loop runs

**Files**:

- `wolf_app.py:5849` - `run_prediction()` function
- `core/providers/polygon_provider.py` - Polygon API (paid, rate-limited)
- `core/providers/alpha_vantage_provider.py` - Alpha Vantage API (paid, backup)
- `yfinance` library - Free fallback


**Data Flow**:

```text
HTTP Request → run_prediction(symbol='MSFT')
  ↓
_get_provider_fetchers() → [Polygon, AlphaVantage, yfinance]
  ↓
get_price_for_symbol('MSFT') → {'price': 425.32, 'provider': 'polygon', 'fresh': True}

```text

**Errors Logged**:

- `wolf_app.py:8245` - Provider prioritization failures
- HTTP 403 (rate limit), 401 (bad key), timeout errors
- Falls back to next provider in chain


**Where Values Can Be None**:

- `price_data` → None if ALL providers fail
- `current_price` → 0.0 if no data available
- `prev_close` → None if market data unavailable


---

## 2. FEATURE EXTRACTION

**Files**:

- `wolf_app.py:5850-5950` - Feature extraction orchestrator
- `core/data_pillars/technical_engine.py` - RSI, MACD, Bollinger Bands
- `core/data_pillars/sentiment_engine.py` - News sentiment
- `core/data_pillars/volume_engine.py` - Volume analysis


**Data Flow**:

```text

run_prediction('MSFT')
  ↓
orchestrator.get_all_features('MSFT') → {
    'rsi': 68.5,
    'macd_histogram': 0.23,
    'bollinger_position': 0.85,
    'volume_ratio': 1.2,
    'sentiment_score': 0.15,
    'price_momentum': 0.035,
    ...
}

```text

**Errors Logged**:

- `wolf_app.py:5900` - "Extracted X/26 features in Yms"
- Missing features logged as warnings
- Feature status: `{'degraded_features': False, 'num_features': 5}`


**Where Values Can Be None**:

- Any individual feature (RSI, MACD, etc.) → None if calculation fails
- `features` dict → Partially populated, never completely empty
- Confidence calculation handles missing features gracefully


---

## 3. PREDICTION GENERATION & CONFIDENCE CALCULATION

**Files**:

- `wolf_app.py:5920-5990` - Direction logic (UP/DOWN/FLAT)
- `wolf_app.py:5850-5950` - Confidence scoring (40-85% range)
- `wolf_app.py:6000-6020` - Ghost V3 confidence bypass


**Data Flow**:

```text

features = orchestrator.get_all_features('MSFT')
  ↓
base_confidence = 45.0  # Conservative baseline
  ↓
Apply signal-based boosts:

  - RSI extreme (+8% if >70 or <30)
  - MACD momentum (+6%)
  - Bollinger position (+5%)
  - Volume spike (+5%)
  - News sentiment (+7%)
  - Price momentum (+6%)
  - Signal convergence (+5% if 4+ aligned)
  - Weak signal penalty (-5% if ≤1 signal)


  ↓
confidence = max(0.40, min(0.85, base_confidence + boosts))
  ↓
direction = determine_direction(features) → 'UP', 'DOWN', or 'FLAT'

```text

**Errors Logged**:

- `wolf_app.py:5990` - "[MSFT] Direction: {direction}, Confidence: {confidence}%, Signals: {count}"
- Legacy diagnostics bypassed (critical fix from commit 26e5483)


**Where Values Can Be None**:

- `direction` → Defaults to 'FLAT' if insufficient signals
- `confidence` → Never None, clamped to 40-85% range
- `signal_strength` → 0-10 integer, never None


---

## 4. DATABASE STORAGE

**Files**:

- `services/predictor.py:165` - `create_prediction()` function
- Database: `./data/ghost_predictions.db`


**Schema**:

```sql

-- predictions table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    run_at REAL NOT NULL,               -- Unix timestamp
    horizon_h INTEGER DEFAULT 48,        -- Forecast horizon (hours)
    method TEXT NOT NULL,                -- 'ghost-av1'
    confidence REAL NOT NULL,            -- 0.40-0.85
    direction TEXT NOT NULL,             -- 'UP', 'DOWN', 'FLAT'
    features_json TEXT,                  -- Serialized features dict
    params_json TEXT,                    -- Model parameters
    tag TEXT                             -- Optional label
);

-- prediction_points table
CREATE TABLE prediction_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    ts REAL NOT NULL,                    -- Timestamp of forecast point
    kind TEXT NOT NULL,                  -- 'forecast' or 'actual'
    price REAL NOT NULL,                 -- Predicted/actual price
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

-- outcomes table
CREATE TABLE outcomes (
    prediction_id INTEGER NOT NULL UNIQUE,
    closed_at REAL NOT NULL,             -- When outcome was evaluated
    mae REAL NOT NULL,                   -- Mean Absolute Error
    map REAL NOT NULL,                   -- Mean Absolute Percentage Error
    rmse REAL NOT NULL,                  -- Root Mean Squared Error
    hit_direction INTEGER NOT NULL,      -- 1 if direction correct, 0 if wrong
    hit_ratio_window REAL,               -- Accuracy within horizon window
    notes TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

```text

**Data Flow**:

```text

create_prediction(
    symbol='MSFT',
    forecast_points=[(ts1, price1), (ts2, price2), ...],  # 25 points
    method='ghost-av1',
    confidence=0.68,
    direction='UP',
    features={'rsi': 68.5, ...},
    params={...}
) → prediction_id (e.g., 87)

```text

**Errors Logged**:

- `services/predictor.py:180` - "Created prediction {id} for {symbol} with {N} forecast points"
- Database write failures logged as errors
- WAL mode + NORMAL synchronous for performance


**Where Values Can Be None**:

- `features_json` → Empty dict `{}` if no features available
- `params_json` → Empty dict `{}` if no model parameters
- `tag` → Empty string `''` if not specified


---

## 5. V3 API EXPOSURE

**Files**:

- `api/cockpit_v3_live_endpoints.py:833` - `/api/v3/predictions/latest` endpoint


**Data Flow**:

```text

GET /api/v3/predictions/latest?symbol=MSFT&limit=10
  ↓
sqlite3.connect(PREDICTIONS_DB)
  ↓
SELECT p.*, o.hit_direction, o.hit_ratio_window, o.map
FROM predictions p
LEFT JOIN outcomes o ON p.id = o.prediction_id
WHERE p.symbol = 'MSFT'
ORDER BY p.run_at DESC
LIMIT 10
  ↓
Format as JSON:
{
    "predictions": [
        {
            "id": 87,
            "symbol": "MSFT",
            "run_at": 1764000894,
            "direction": "UP",
            "confidence": 0.68,
            "horizon_h": 48,
            "outcome": "pending",        # or "correct"/"wrong"
            "accuracy_pct": null         # or 87.5 if outcome evaluated
        },
        ...
    ],
    "count": 10,
    "timestamp": 1764000907
}

```text

**Errors Logged**:

- `api/cockpit_v3_live_endpoints.py:870` - Database connection failures
- HTTP 500 if query fails
- Empty array `[]` if no predictions found


**Where Values Can Be None**:

- `predictions` → Empty array `[]` if no data
- `outcome` → "pending" if horizon window not yet closed
- `accuracy_pct` → null if outcome not yet evaluated


---

## 6. COCKPIT V3 UI RENDERING

**Files**:

- `static/cockpit_v3.js:395` - `loadWatchlist()` function
- `static/cockpit_v3.js:325` - `loadForecast()` function
- `templates/cockpit_v3.html` - UI panels


**Data Flow**:

```text

JavaScript: loadWatchlist()
  ↓
fetch('/api/v3/watchlist')  → {stocks: ['WOLF', 'AAPL', 'MSFT', ...], crypto: [...]}
  ↓
fetch('/api/v3/predictions/latest?limit=100')  → {predictions: [...]}
  ↓
Create lookup map: predMap['MSFT'] = {confidence: 68, direction: 'UP'}
  ↓
Enrich watchlist with prediction data:
watchlistData = symbols.map(symbol => ({
    symbol: symbol,
    change: 0,                           # Price change (not available yet)
    ghost_score: predMap[symbol]?.confidence || 0,
    direction: predMap[symbol]?.direction || 'FLAT',
    type: 'stock'
}))
  ↓
renderWatchlist(watchlistData) → Updates DOM:
<div class="watchlist-row">
    <div class="watchlist-ticker">MSFT</div>
    <div class="watchlist-score">↑ Ghost: 68%</div>
</div>

```text

**UI Panels Wired**:

1. **Watchlist**(`#panel-watchlist`) - Shows all symbols with Ghost scores


2.**Forecast**(`#panel-forecast`) - Shows direction/confidence for searched symbol
3.**Top Movers**(`#panel-movers`) - Shows hunter/feed data (separate endpoint)
4.**Health Score**(`#panel-health`) - Shows Ghost Score and goal progress**Errors Logged**:

- `cockpit_v3.js:410` - "Error loading watchlist" (console.error)
- Graceful degradation: Shows "--" if data unavailable
- Empty panels show placeholder text


**Where Values Can Be None**:

- `predMap[symbol]` → undefined if no prediction for that symbol
- `ghost_score` → 0 if no prediction
- `direction` → 'FLAT' if no prediction


---

## 7. TELEGRAM ALERT GENERATION

**Files**:

- `wolf_app.py:10240` - `_build_multi_symbol_prediction_message()` function
- `wolf_app.py:10577` - `enqueue_alert_text()` function
- `wolf_app.py:3860` - Alert worker thread


**Data Flow**:

```text

Auto-prediction loop completes batch
  ↓
_build_multi_symbol_prediction_message(predictions_data)
  ↓
Header (HARDCODED LIE DETECTED):
"🎯 GHOST AI TRADING SIGNALS
⏰ {now_str}
🤖 85%+ Accuracy | Smart Filter Active"
  ↓
Rank opportunities by confidence & gain potential
  ↓
Format message with symbol list:
"1. 📈 MSFT
   💰 $425.32 → $445.20 (+4.7%)
   ✅ Confidence: 68%"
  ↓
enqueue_alert_text(message) → Telegram queue
  ↓
Alert worker sends via Telegram Bot API

```text

**🚨 CRITICAL LIE DETECTED**:

- **Line 10261**: `"🤖 85%+ Accuracy | Smart Filter Active"`
- This text is HARDCODED and ALWAYS appears
- It claims 85%+ accuracy even when:
  - Zero predictions have been evaluated (total_predictions = 0)
  - No accuracy data exists in database
  - The system just started and has no historical data


**Errors Logged**:

- `wolf_app.py:10580` - "alert_queue_full" if queue is full
- Telegram API errors logged by alert worker
- No validation that accuracy claim matches reality


**Where Values Can Be None**:

- `predictions_data.get("predictions")` → Empty dict `{}` if no predictions
- Telegram message still sent with "No high-conviction plays" text
- **Accuracy claim is ALWAYS sent, regardless of actual accuracy**---


## 8. ACCURACY TRACKING (48h EVALUATION WINDOW)**Files**

- `core/accuracy_tracker.py` - Outcome reconciliation
- `wolf_app.py:10727` - Reconciler worker thread


**Data Flow**(runs every 5 minutes):

```text

_reconciler_loop()
  ↓
Find predictions where (now - run_at) >= horizon_h
  ↓
Fetch actual price for symbol at outcome timestamp
  ↓
Calculate metrics:

  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - RMSE (Root Mean Squared Error)
  - hit_direction (1 if direction correct, 0 if wrong)


  ↓
INSERT INTO outcomes (prediction_id, mae, map, rmse, hit_direction, ...)
  ↓
/api/v3/accuracy/summary queries outcomes table for win rate

```text**Accuracy API Response**:

```json

{
    "daily_accuracy_pct": 0.0,
    "weekly_accuracy_pct": 0.0,
    "monthly_accuracy_pct": 0.0,
    "correct": 0,
    "wrong": 0,
    "pending": 5,
    "total_predictions": 5
}

```text

**Errors Logged**:

- `core/accuracy_tracker.py` - Price fetch failures during reconciliation
- Prediction outcomes marked as "unresolved" if price unavailable
- Win rate = 0% if no outcomes evaluated yet


**Where Values Can Be None**:

- All accuracy metrics → 0 until first outcome is evaluated (48h+ after first prediction)
- `hit_direction` → null until outcome reconciliation runs
- Telegram message claims "85%+ Accuracy" regardless of these values


---

## SUMMARY: COMPLETE DATA LINEAGE FOR MSFT

```text

[1] HTTP /api/predict/run?symbol=MSFT
      ↓
[2] Polygon/AlphaVantage/yfinance price fetch → $425.32
      ↓
[3] Feature extraction → {rsi: 68.5, macd: 0.23, sentiment: 0.15, ...}
      ↓
[4] Confidence calculation → 68% (base 45% + signal boosts)
      ↓
[5] Direction determination → 'UP' (based on RSI, MACD, Bollinger convergence)
      ↓
[6] Database write → predictions table (id=87, symbol=MSFT, confidence=0.68, direction=UP)
      ↓
[7] V3 API exposure → GET /api/v3/predictions/latest?symbol=MSFT → JSON response
      ↓
[8] Cockpit UI fetch → JavaScript loads predictions, enriches watchlist, renders DOM
      ↓
[9] Telegram alert → HARDCODED "85%+ Accuracy" claim (ALWAYS SENT, NEVER VALIDATED)
      ↓
[10] Outcome reconciliation → Runs 48h later, calculates actual accuracy, stores in outcomes table

```text

---

## KEY FAILURE POINTS

| **Step**|**What Can Break**|**Symptom**|**Error Log Location**|
|----------|-------------------|-------------|----------------------|
| Price Fetch | Rate limit (403), bad key (401), timeout | prediction fails, no data stored | `wolf_app.py:8270` |
| Feature Extraction | yfinance data unavailable, API errors | Low feature count (5/26), degraded features |
`wolf_app.py:5900` |
| Database Write | Disk full, permission denied | Prediction not stored, DB error | `services/predictor.py:180` |
| V3 API | Database locked, query timeout | HTTP 500, empty predictions array | `api/cockpit_v3_live_endpoints.py:870` |
| UI Rendering | Network error, CORS, wrong base URL | Empty panels, "Failed to load" message | `cockpit_v3.js:410`
(console) |
| Telegram Alert | Queue full, bot token invalid | No alerts sent, queue full warning | `wolf_app.py:10580` |
| Accuracy Tracking | Price fetch fails during reconciliation | Outcomes not updated, win rate stuck at 0% |
`core/accuracy_tracker.py` |

---

## THE BIG LIE**Line 10261 of wolf_app.py**

```python

🤖 85%+ Accuracy | Smart Filter Active

```text

This text is **ALWAYS**sent in Telegram alerts, regardless of:

- Actual accuracy (which may be 0%, 30%, or any value)
- Total predictions evaluated (may be 0)
- Whether accuracy tracking is even running**FIX REQUIRED**: Replace hardcoded text with dynamic accuracy from `/api/v3/accuracy/summary` endpoint.


---

**Generated**: November 24, 2025
**Auditor**: Ghost Truth Squad
**Status**: ✅ Trace Complete | 🚨 Lie Detected
