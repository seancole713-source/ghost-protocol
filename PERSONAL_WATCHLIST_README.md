# 🎯 GHOST PROTOCOL PERSONAL WATCHLIST MODULE

**Version:** 1.0  
**Date:** December 2, 2025  
**Status:** ✅ COMPLETE - Ready for Integration

---

## 📋 EXECUTIVE SUMMARY

A **single-owner persistent personal watchlist** system for Ghost Protocol v3 that enables manual tracking of stocks and crypto with:

✅ **Postgres-backed persistence** (survives browser sessions)  
✅ **Manual add/remove** from Cockpit UI  
✅ **Continuous 48h predictions** (daily + intraday)  
✅ **Telegram alerts** (market open/close + big moves)  
✅ **Position tracking** (owns_position flag)  
✅ **NO trade execution** (signal generation only)  
✅ **Live data only** (SIM_MODE=0 preserved)

---

## 🏗️ ARCHITECTURE

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     COCKPIT UI (Browser)                         │
│  personal_watchlist_ui.js → Add/Remove/Update symbols           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    FastAPI Endpoints
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│           /api/v3/watchlist/* (7 endpoints)                     │
│  personal_watchlist_endpoints.py                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│              Core Personal Watchlist Manager                     │
│  core/personal_watchlist.py → CRUD + enrichment                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                    Postgres Database                             │
│  - ghost_watchlist_items (main table)                           │
│  - watchlist_prediction_tracking                                │
│  - watchlist_price_snapshots                                    │
│  - watchlist_alerts_log                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Parallel Systems

```
┌─────────────────────────────────────────────────────────────────┐
│         Watchlist Prediction Scheduler (Background)              │
│  core/watchlist_prediction_scheduler.py                         │
│  - Market open: 9 AM EST stocks                                 │
│  - Market close: 4 PM EST stocks                                │
│  - Big moves: Every 15 minutes (±5% threshold)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    services/predictor.py
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│         Watchlist Telegram Alerts (On-Demand)                   │
│  core/watchlist_telegram_alerts.py                              │
│  - Cooldown: 4h per symbol per alert type                       │
│  - Rate limit: 5 alerts/hour global                             │
│  - Format: 📌 WATCHLIST prefix                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 DATABASE SCHEMA

### 1. `ghost_watchlist_items` (Main Table)

| Column                | Type          | Description                           |
|-----------------------|---------------|---------------------------------------|
| `id`                  | BIGSERIAL PK  | Auto-increment ID                     |
| `symbol`              | TEXT NOT NULL | Ticker symbol (uppercase)             |
| `asset_type`          | TEXT NOT NULL | 'crypto' or 'stock'                   |
| `owns_position`       | BOOLEAN       | TRUE if user holds asset              |
| `notes`               | TEXT          | User notes/comments                   |
| `added_at`            | TIMESTAMPTZ   | When symbol was added                 |
| `updated_at`          | TIMESTAMPTZ   | Last modification time                |
| `active`              | BOOLEAN       | TRUE = active, FALSE = soft-deleted   |
| `price_at_add`        | REAL          | Price when first added                |
| `alert_threshold_pct` | REAL          | Alert if price moves ±this % (def 5%) |
| `priority`            | INTEGER       | 1=normal, 2=high, 3=critical          |

**Constraints:**
- UNIQUE (symbol, asset_type) WHERE active = TRUE
- CHECK (asset_type IN ('crypto', 'stock'))
- CHECK (LENGTH(symbol) > 0 AND LENGTH(symbol) <= 20)

**Indexes:**
- idx_watchlist_symbol ON (symbol) WHERE active = TRUE
- idx_watchlist_asset_type ON (asset_type) WHERE active = TRUE
- idx_watchlist_active ON (active, priority DESC, added_at DESC)
- idx_watchlist_owns_position ON (owns_position) WHERE owns_position = TRUE AND active = TRUE

### 2. `watchlist_prediction_tracking`

Tracks prediction generation events for watchlist symbols.

| Column                | Type          | Description                           |
|-----------------------|---------------|---------------------------------------|
| `id`                  | BIGSERIAL PK  | Auto-increment ID                     |
| `watchlist_item_id`   | BIGINT FK     | References ghost_watchlist_items(id)  |
| `symbol`              | TEXT NOT NULL | Ticker symbol                         |
| `prediction_id`       | BIGINT        | References ghost_predictions(id)      |
| `direction`           | TEXT NOT NULL | UP/DOWN/FLAT                          |
| `confidence`          | REAL NOT NULL | 0.0-1.0                               |
| `expected_move_pct`   | REAL NOT NULL | Expected % change                     |
| `horizon_h`           | INTEGER       | Prediction horizon (default 48h)      |
| `price_at_prediction` | REAL          | Price when prediction made            |
| `generated_at`        | TIMESTAMPTZ   | When prediction was generated         |
| `reason`              | TEXT          | 'market_open', 'market_close', 'big_move', 'manual' |
| `alert_sent`          | BOOLEAN       | TRUE if Telegram alert sent           |
| `alert_sent_at`       | TIMESTAMPTZ   | When alert was sent                   |

**Indexes:**
- idx_watchlist_pred_item ON (watchlist_item_id, generated_at DESC)
- idx_watchlist_pred_symbol ON (symbol, generated_at DESC)
- idx_watchlist_pred_alerts ON (alert_sent, generated_at DESC)

### 3. `watchlist_price_snapshots`

High-frequency price tracking for big-move detection.

| Column                | Type          | Description                           |
|-----------------------|---------------|---------------------------------------|
| `id`                  | BIGSERIAL PK  | Auto-increment ID                     |
| `watchlist_item_id`   | BIGINT FK     | References ghost_watchlist_items(id)  |
| `symbol`              | TEXT NOT NULL | Ticker symbol                         |
| `price`               | REAL NOT NULL | Snapshot price                        |
| `change_pct_24h`      | REAL          | 24h price change %                    |
| `volume_24h`          | REAL          | 24h volume                            |
| `snapshot_at`         | TIMESTAMPTZ   | Snapshot timestamp                    |

**Retention:** Keep last 7 days only (manual cleanup job).

**Indexes:**
- idx_watchlist_prices_item ON (watchlist_item_id, snapshot_at DESC)
- idx_watchlist_prices_symbol ON (symbol, snapshot_at DESC)

### 4. `watchlist_alerts_log`

Historical log of all Telegram alerts sent.

| Column                | Type          | Description                           |
|-----------------------|---------------|---------------------------------------|
| `id`                  | BIGSERIAL PK  | Auto-increment ID                     |
| `watchlist_item_id`   | BIGINT FK     | References ghost_watchlist_items(id)  |
| `symbol`              | TEXT NOT NULL | Ticker symbol                         |
| `alert_type`          | TEXT NOT NULL | 'open', 'close', 'big_move'           |
| `direction`           | TEXT          | Prediction direction                  |
| `confidence`          | REAL          | Prediction confidence                 |
| `expected_move_pct`   | REAL          | Expected % move                       |
| `current_price`       | REAL          | Price at alert time                   |
| `change_pct`          | REAL          | Actual % change (for big_move)        |
| `message`             | TEXT          | Full alert message text               |
| `telegram_sent`       | BOOLEAN       | TRUE if delivered                     |
| `telegram_sent_at`    | TIMESTAMPTZ   | Delivery timestamp                    |
| `telegram_chat_id`    | BIGINT        | Telegram chat ID                      |
| `created_at`          | TIMESTAMPTZ   | Alert creation time                   |

**Indexes:**
- idx_watchlist_alerts_symbol ON (symbol, created_at DESC)
- idx_watchlist_alerts_type ON (alert_type, created_at DESC)
- idx_watchlist_alerts_cooldown ON (symbol, alert_type, created_at DESC)

---

## 🔌 API ENDPOINTS

Base path: `/api/v3/watchlist`

### 1. `POST /api/v3/watchlist/add`

Add symbol to personal watchlist (or re-activate if soft-deleted).

**Request:**
```json
{
  "symbol": "AAPL",
  "asset_type": "stock",
  "owns_position": false,
  "notes": "Apple Inc. - watching for entry",
  "alert_threshold_pct": 5.0,
  "priority": 2
}
```

**Response:**
```json
{
  "ok": true,
  "action": "added",
  "id": 123,
  "symbol": "AAPL",
  "asset_type": "stock",
  "owns_position": false,
  "added_at": "2025-12-02T12:34:56Z"
}
```

### 2. `POST /api/v3/watchlist/remove`

Soft-delete symbol from watchlist (sets active=FALSE).

**Request:**
```json
{
  "symbol": "AAPL",
  "asset_type": "stock"
}
```

**Response:**
```json
{
  "ok": true,
  "symbol": "AAPL",
  "asset_type": "stock"
}
```

### 3. `GET /api/v3/watchlist/user`

Get enriched watchlist with live predictions and prices.

**Response:**
```json
{
  "items": [
    {
      "id": 123,
      "symbol": "AAPL",
      "asset_type": "stock",
      "owns_position": false,
      "notes": "Apple Inc.",
      "alert_threshold_pct": 5.0,
      "priority": 2,
      "added_at": "2025-12-02T12:34:56Z",
      "current_price": 283.10,
      "prediction": {
        "prediction_id": 366,
        "direction": "DOWN",
        "confidence": 0.58,
        "expected_move": -4.5,
        "horizon_h": 48,
        "run_at": 1764635853.456
      }
    }
  ],
  "count": 1,
  "timestamp": 1764642576.184
}
```

### 4. `POST /api/v3/watchlist/update-position`

Update the owns_position flag for a symbol.

**Request:**
```json
{
  "symbol": "AAPL",
  "asset_type": "stock",
  "owns_position": true
}
```

**Response:**
```json
{
  "ok": true,
  "symbol": "AAPL",
  "owns_position": true
}
```

### 5. `GET /api/v3/watchlist/history/{symbol}`

Get prediction history for a watchlist symbol.

**Query Params:**
- `limit` (optional, default 50): Max number of records

**Response:**
```json
{
  "symbol": "AAPL",
  "history": [
    {
      "id": 789,
      "prediction_id": 366,
      "direction": "DOWN",
      "confidence": 0.58,
      "expected_move_pct": -4.5,
      "horizon_h": 48,
      "price_at_prediction": 283.10,
      "generated_at": "2025-12-02T12:34:56Z",
      "reason": "market_close",
      "alert_sent": true
    }
  ],
  "count": 1
}
```

### 6. `POST /api/v3/watchlist/trigger-prediction`

Manually trigger a prediction for a watchlist symbol.

**Request:**
```json
{
  "symbol": "AAPL",
  "asset_type": "stock"
}
```

**Response:**
```json
{
  "ok": true,
  "symbol": "AAPL",
  "reason": "manual",
  "message": "Prediction queued"
}
```

### 7. `GET /api/v3/watchlist/stats`

Get watchlist statistics.

**Response:**
```json
{
  "total_symbols": 10,
  "stocks": 6,
  "crypto": 4,
  "owned_positions": 2,
  "alerts_sent_7d": {
    "total": 15,
    "by_type": {
      "open": 5,
      "close": 5,
      "big_move": 5
    }
  }
}
```

---

## 🚀 INTEGRATION GUIDE

### Step 1: Run Database Migration

```bash
# Connect to Postgres (Railway or local)
psql $DATABASE_URL

# Run migration
\i migrations/001_personal_watchlist.sql

# Verify tables created
\dt ghost_watchlist*
```

### Step 2: Mount API Endpoints in wolf_app.py

Add to `wolf_app.py`:

```python
# Import personal watchlist router
from api.personal_watchlist_endpoints import router as watchlist_router

# Mount router
APP.include_router(watchlist_router)
```

### Step 3: Start Watchlist Scheduler

Add to `wolf_app.py` startup:

```python
from core.watchlist_prediction_scheduler import start_watchlist_scheduler, stop_watchlist_scheduler
import atexit

# Start scheduler on app startup
@APP.on_event("startup")
async def startup_watchlist_scheduler():
    start_watchlist_scheduler()

# Stop scheduler on app shutdown
atexit.register(stop_watchlist_scheduler)
```

### Step 4: Add UI JavaScript to Cockpit HTML

Add to `templates/cockpit_v3.html` (before `</body>`):

```html
<!-- Personal Watchlist UI Module -->
<script src="/static/personal_watchlist_ui.js"></script>
```

### Step 5: Configure Environment Variables

Add to Railway/production env:

```bash
# Watchlist Scheduler
WATCHLIST_SCHEDULER_ENABLED=1
WATCHLIST_OPEN_HOUR=9          # 9 AM EST market open
WATCHLIST_CLOSE_HOUR=16        # 4 PM EST market close
WATCHLIST_BIG_MOVE_CHECK_MINUTES=15
WATCHLIST_BIG_MOVE_THRESHOLD_PCT=5.0

# Telegram Alerts
WATCHLIST_ALERTS_ENABLED=1
WATCHLIST_ALERTS_INCLUDE_OPEN_CLOSE=1
WATCHLIST_ALERTS_INCLUDE_BIG_MOVES=1
WATCHLIST_ALERT_COOLDOWN_HOURS=4
WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR=5
```

---

## 🧪 TESTING CHECKLIST

### Unit Tests

- [ ] `test_personal_watchlist_add_remove()` - CRUD operations
- [ ] `test_personal_watchlist_enrichment()` - Prediction integration
- [ ] `test_watchlist_prediction_scheduler()` - Scheduling logic
- [ ] `test_watchlist_telegram_alerts()` - Alert formatting

### Integration Tests

- [ ] Add 5 symbols (3 stocks, 2 crypto) via API
- [ ] Verify symbols appear in `/api/v3/watchlist/user`
- [ ] Trigger manual prediction for each symbol
- [ ] Verify predictions tracked in `watchlist_prediction_tracking`
- [ ] Remove 1 symbol, verify soft-delete (active=FALSE)
- [ ] Re-add removed symbol, verify re-activation

### End-to-End Tests

- [ ] Open Cockpit UI in browser
- [ ] Click "Add Symbol" button
- [ ] Add AAPL (stock, owned=false)
- [ ] Verify AAPL appears in watchlist with prediction
- [ ] Toggle "I own this" checkbox
- [ ] Verify OWN badge appears
- [ ] Click history icon, verify prediction history modal
- [ ] Click remove icon, verify symbol removed
- [ ] Reload browser, verify watchlist persists

### Production Validation

- [ ] Deploy to Railway with env vars set
- [ ] Verify scheduler starts on app startup (check logs)
- [ ] Add 3-5 symbols to watchlist
- [ ] Wait for market open (9 AM EST), verify predictions generated
- [ ] Check `watchlist_prediction_tracking` table for entries
- [ ] Verify Telegram alert sent (if WATCHLIST_ALERTS_ENABLED=1)
- [ ] Check alert cooldown enforcement (4h)
- [ ] Verify global rate limit (5 alerts/hour)

---

## 📱 TELEGRAM ALERT FORMAT

### Market Open/Close Alert

```
📌 **WATCHLIST** – MARKET OPEN

🎯 **AAPL** (STOCK)
🔴 **48h Prediction:** DOWN
📊 **Confidence:** 58%
📈 **Expected Move:** -4.5%
💰 **Current Price:** $283.10

⚠️ You DO NOT own this yet

⏰ Ghost AI – MARKET OPEN Signal
```

### Big Move Alert

```
📌 **WATCHLIST** – BIG MOVE DETECTED

🎯 **BTC** (CRYPTO)
🚀 **Price Move:** +6.2% (last 15-60 min)
💰 **Current Price:** $87,105.40

🟢 **48h Ghost Prediction:** UP
📊 **Confidence:** 46%
📈 **Expected Move:** +2.3%

✅ **You OWN this**

⚡ Ghost AI – Intraday Alert
```

---

## ⚙️ CONFIGURATION

### Scheduler Settings

| Variable                             | Default | Description                          |
|--------------------------------------|---------|--------------------------------------|
| WATCHLIST_SCHEDULER_ENABLED          | 1       | Enable/disable scheduler             |
| WATCHLIST_OPEN_HOUR                  | 9       | Market open hour (EST)               |
| WATCHLIST_CLOSE_HOUR                 | 16      | Market close hour (EST)              |
| WATCHLIST_BIG_MOVE_CHECK_MINUTES     | 15      | How often to check for big moves     |
| WATCHLIST_BIG_MOVE_THRESHOLD_PCT     | 5.0     | Price move % to trigger alert        |

### Alert Settings

| Variable                                  | Default | Description                          |
|-------------------------------------------|---------|--------------------------------------|
| WATCHLIST_ALERTS_ENABLED                  | 1       | Enable/disable all watchlist alerts  |
| WATCHLIST_ALERTS_INCLUDE_OPEN_CLOSE       | 1       | Send market open/close alerts        |
| WATCHLIST_ALERTS_INCLUDE_BIG_MOVES        | 1       | Send big move alerts                 |
| WATCHLIST_ALERT_COOLDOWN_HOURS            | 4       | Min hours between same alert type    |
| WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR     | 5       | Max alerts per hour (all symbols)    |

---

## 📂 FILE STRUCTURE

```
/workspaces/ghost-protocol/
├── migrations/
│   └── 001_personal_watchlist.sql          # Database schema
├── core/
│   ├── personal_watchlist.py               # Core manager + CRUD
│   ├── watchlist_prediction_scheduler.py   # Prediction scheduler
│   └── watchlist_telegram_alerts.py        # Telegram alert integration
├── api/
│   └── personal_watchlist_endpoints.py     # FastAPI REST endpoints
└── static/
    └── personal_watchlist_ui.js            # Cockpit UI module
```

---

## 🔒 SECURITY

- **Single-Owner:** No multi-tenant auth (same security as existing Ghost endpoints)
- **IP Allowlist:** Reuses existing Ghost IP protection
- **API Token:** Optional `X-API-Token` header verification
- **Soft Deletes:** Removed symbols can be recovered (active=FALSE, not dropped)
- **No Auto-Trading:** System generates signals only, NO trade execution

---

## 🚨 KNOWN LIMITATIONS

1. **Single Owner Only:** Not designed for multi-user/multi-tenant
2. **No Real-Time Websockets:** Uses polling (15s intervals in UI)
3. **No Symbol Validation:** API accepts any symbol string (validation in predictor)
4. **Telegram Dependency:** Alerts require existing telegram_hunter setup
5. **Price Snapshot Retention:** Manual cleanup needed (no auto-TTL in Postgres < 15)

---

## 🛠️ TROUBLESHOOTING

### Watchlist Not Loading in UI

**Symptom:** Empty watchlist despite symbols in database  
**Solution:**
1. Check browser console for errors
2. Verify `/api/v3/watchlist/user` returns data in browser network tab
3. Check `ghost_watchlist_items` table has `active=TRUE` rows
4. Restart scheduler if predictions not being generated

### Predictions Not Generating

**Symptom:** No entries in `watchlist_prediction_tracking`  
**Solution:**
1. Check scheduler is running: `grep "Watchlist scheduler" logs/wolf_app.log`
2. Verify `WATCHLIST_SCHEDULER_ENABLED=1` in env
3. Check market hours (open=9 AM, close=4 PM EST)
4. Manually trigger: `curl -X POST /api/v3/watchlist/trigger-prediction -d '{"symbol":"AAPL","asset_type":"stock"}'`

### Telegram Alerts Not Sending

**Symptom:** No alerts received despite predictions  
**Solution:**
1. Verify `WATCHLIST_ALERTS_ENABLED=1`
2. Check telegram_hunter configured (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
3. Check cooldown not blocking: `SELECT * FROM watchlist_alerts_log WHERE symbol='AAPL' ORDER BY created_at DESC LIMIT 5`
4. Check rate limit: max 5 alerts/hour global

---

## 📈 ROADMAP / FUTURE ENHANCEMENTS

- [ ] Multi-symbol bulk add (upload CSV)
- [ ] Watchlist groups/tags (e.g., "Tech Stocks", "DeFi Crypto")
- [ ] Custom alert rules (e.g., "Alert if confidence > 70%")
- [ ] Watchlist performance analytics (win rate per symbol)
- [ ] Mobile app integration (React Native / Flutter)
- [ ] Real-time websocket updates (replace polling)
- [ ] Watchlist import/export (JSON/CSV)
- [ ] Shareable watchlist URLs (read-only)

---

## 📝 CHANGELOG

### v1.0 (December 2, 2025)
- ✅ Initial release
- ✅ Postgres schema with 4 tables
- ✅ 7 REST API endpoints
- ✅ Prediction scheduler (daily + intraday)
- ✅ Telegram alert integration
- ✅ Full Cockpit UI CRUD interface
- ✅ Single-owner persistence

---

## 🤝 SUPPORT

**Issues:** Check troubleshooting section above  
**Questions:** Review API endpoint documentation  
**Bugs:** Check browser console + server logs  

---

**Status:** ✅ **COMPLETE - Ready for Production Integration**  
**Testing:** Pending database migration + wolf_app.py integration  
**Deployment:** Railway (production) + local dev
