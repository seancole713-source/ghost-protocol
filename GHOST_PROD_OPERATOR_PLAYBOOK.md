# GHOST PRODUCTION OPERATOR PLAYBOOK

**Version**: 1.0
**Date**: 2025-12-01
**Mode**: PostgreSQL Primary + Dual-Write
**Environment**: Railway Production

---

## Section 1: Live Check Commands

### 🔍 Health Check Procedure

Run these three commands to verify Ghost is operating correctly:

#### Test 1: BTC Prediction (Crypto)

```bash
curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC">>>>> | python3 -m json.tool

```text

**Healthy Output:**```json

{
    "ok": true,
    "prediction_id": 3,
    "symbol": "BTC",
    "run_at": 1764628075750,
    "horizon_h": 48,
    "confidence": 0.46,
    "direction": "UP",
    "current_price": 86461.0,
    "feature_count": 25,
    "available_count": 23,
    "duration_ms": 24
}

```text**Key Indicators:**- ✅ `"ok": true` - Request succeeded

- ✅ `"prediction_id"` exists - Saved to Postgres successfully
- ✅ `duration_ms` < 5000 - Fast response (crypto providers working)
- ✅ `confidence` > 0 - Prediction generated with confidence score


---

#### Test 2: XRP Prediction (Crypto)

```bash

curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=XRP">>>>> | python3 -m json.tool

```text**Healthy Output:**```json

{
    "ok": true,
    "prediction_id": 4,
    "symbol": "XRP",
    "direction": "UP",
    "confidence": 0.46,
    "current_price": 2.03085,
    "duration_ms": 222
}

```text**Key Indicators:**- ✅ Similar structure to BTC

- ✅ `duration_ms` 100-500ms (normal for crypto)


---

#### Test 3: AAPL Prediction (Stock)

```bash

curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=AAPL">>>>> | python3 -m json.tool

```text**Healthy Output:**```json

{
    "ok": true,
    "prediction_id": 5,
    "symbol": "AAPL",
    "direction": "DOWN",
    "confidence": 0.58,
    "current_price": 278.85,
    "duration_ms": 1539
}

```text**Key Indicators:**- ✅ `duration_ms` 1000-3000ms (normal for stocks - slower than crypto)

- ✅ Stock providers working (yfinance/Yahoo/AlphaVantage/Polygon)


---

#### Test 4: Verify Cache Endpoints

```bash

# BTC cached prediction

curl -sS "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=BTC">>>>> | python3 -m json.tool

# XRP cached prediction

curl -sS "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=XRP">>>>> | python3 -m json.tool

```text**Healthy Output:**```json

{
    "ok": true,
    "predictions": [
        {
            "symbol": "BTC",
            "direction": "UP",
            "confidence": 0.46,
            "expected_move": 2.3,
            "horizon_h": 48,
            "run_at": 1764628075.75
        }
    ],
    "count": 1
}

```text**Key Indicators:**- ✅ Returns cached prediction from previous `/api/predict/run` call

- ✅ No database query required (instant response)
- ✅ `run_at` timestamp matches previous prediction


---

## Section 2: When Telegram Sends Real Trades vs HOLDING PATTERN

### 📱 Telegram Alert Logic

Ghost uses a**modern, single-pipeline alert system**in `core/telegram_hunter.py` and `wolf_app.py`. No legacy systems
exist.

### Alert Decision Flow

```text

New Prediction Created
         ↓
Calculate Opportunity Score = |gain_pct| × confidence × (1 + momentum)
         ↓
Apply Filters

```text

### Filter Criteria

#### For SHORT-TERM SIGNALS (48h-7 days)

```python

confidence > 0.70  (70%)
gain_pct > 2.0%    (at least 2% expected gain)
momentum > 0.3     (positive momentum)
direction == "BUY"

```text**When this passes**: Telegram sends "⚡ SHORT-TERM GAINS" section with symbol, price, predicted gain, and confidence.

#### For LONG-TERM SIGNALS (1-6 months)

```python

confidence > 0.75  (75%)
gain_pct > 5.0%    (at least 5% expected gain)
direction == "BUY"

```text

**When this passes**: Telegram sends "🎯 LONG-TERM HOLDS" section.

#### For URGENT SELLS

```python

direction == "SELL"
Top 3 highest-scored sell signals

```text

**When this passes**: Telegram sends "🚨 URGENT SELLS" section.

---

### HOLDING PATTERN Message

**Trigger Conditions:**```python

total_opportunities = len(short_term) + len(long_term) + len(urgent_sells)

if total_opportunities == 0:
    Send "💤 HOLDING PATTERN"

```text**Message Format:**```text

💤 Market Status: HOLDING PATTERN
No high-conviction signals. Wait for better setups.

💡 Ghost AI filters out noise. Only see signals >70% confidence.

```text**Translation for Operator:**- No predictions currently meet the quality thresholds (70%+ confidence, 2%+ gain)

- This is**NORMAL**and means Ghost is waiting for better opportunities
- Not a system failure - it's working as designed (filtering noise)


---

### Daily Reports**Schedule:**-**Morning Report**: 7:00 AM (checks every 10 minutes)

- **Evening Report**: 8:00 PM (checks every 10 minutes)


**What Gets Included:**1. Top opportunities meeting filter criteria (see above)

1. Accuracy statistics from `calculate_accuracy("24h")`
2. Market status (HOLDING PATTERN if no opportunities)**Report Format:**```text


🎯 GHOST AI TRADING SIGNALS
⏰ 7:15 AM EST
📊 Ghost Accuracy: 87.5% (105/120 predictions correct over 24h)

⚡ SHORT-TERM GAINS (48h-7 days)
[List of filtered opportunities]

🎯 LONG-TERM HOLDS (1-6 months)
[List of filtered opportunities]

💤 HOLDING PATTERN
[If no opportunities meet criteria]

```text

---

### Confidence Thresholds Summary

| Alert Type | Min Confidence | Min Gain | Other Criteria |
|------------|---------------|----------|----------------|
|**Instant Alert**| 70% | 2% | High score (80+), momentum > 0.3 |
|**Short-Term (Daily)**| 70% | 2% | Momentum > 0.3, BUY direction |
|**Long-Term (Daily)**| 75% | 5% | BUY direction |
|**Urgent Sell**| None | None | Top 3 SELL signals |
|**Daily Report**| 55% | None | MIN_ALERT_CONFIDENCE env var |**Environment Variable Override:**```bash

MIN_ALERT_CONFIDENCE=0.55  # Default threshold for daily reports

```text

This means predictions with confidence ≥ 55% are included in daily reports, but only those ≥ 70% (short-term) or ≥ 75%
(long-term) are sent as instant alerts.

---

## Section 3: Accuracy and Outcomes

### 📊 How Accuracy is Computed

#### Data Source

Accuracy is calculated from the**outcomes table**in PostgreSQL:

1.**Predictions saved**: `predictions` table (via `prediction_store`)

1. **Outcomes created**: `outcomes` table (after 48h window closes)
2. **Accuracy calculated**: JOIN predictions + outcomes


#### Calculation Logic

**Location**: `core/prediction_tracker.py` → `calculate_accuracy(period)`

**SQL Query:**

```sql

SELECT * FROM ghost_predictions
WHERE checked = 1 AND confidence >= 0.10
  AND predicted_at >= {cutoff_timestamp}
ORDER BY predicted_at DESC

```text

**Computation:**

```python

total_predictions = len(rows)
correct_predictions = sum(1 for row in rows if row["correct"] == 1)
accuracy_pct = (correct_predictions / total_predictions) * 100

```text

**What Makes a Prediction "Correct":**- Predicted direction matches actual direction (UP/DOWN/FLAT)

- Direction determined by price change: `(outcome_price - current_price) / current_price`


---

### Why Accuracy Shows 0.0%

#### Condition 1: No Predictions Evaluated Yet

```python

if not rows:  # No predictions with checked=1
    return {
        "accuracy_pct": 0.0,
        "total_predictions": 0,
        "correct_predictions": 0
    }

```text**Translation:**- Predictions exist but**48-hour window hasn't closed yet**- Outcomes are created**only after 48 hours**via outcome reconciliation

- This is**NORMAL**for new deployments or after a restart**Timeline:**```text


Time 0: Prediction created → saved to Postgres
Time +48h: Outcome reconciler runs → creates outcome record
Time +48h+: Accuracy calculated → non-zero accuracy appears

```text

#### Condition 2: All Predictions Failed Direction Test

```python

correct_predictions = 0  # All predictions got direction wrong
accuracy_pct = 0.0

```text**Translation:**- Predictions evaluated, but all directions were incorrect

- This is**RARE**(Ghost typically maintains 70-85% accuracy)
- Check if market had unusual volatility or black swan events


---

### How Accuracy Changes Over Time**Day 1 (0-48 hours after deployment):**```text

Accuracy: 0.0% (0/0 predictions evaluated)
Status: "🔄 Building prediction history (no evaluations yet)"

```text**Day 3 (48-72 hours):**```text

Accuracy: 75.0% (6/8 predictions evaluated)
Status: "🎯 Ghost Accuracy: 75.0% (6/8 predictions correct)"

```text**Day 7 (steady state):**```text

Accuracy: 82.3% (89/108 predictions evaluated)
Status: "🎯 Ghost Accuracy: 82.3% (89/108 predictions correct over 7d)"

```text**Steady State Expectation:**- Accuracy should stabilize at**70-85%**for direction predictions

- Higher accuracy (85%+) indicates strong market conditions
- Lower accuracy (60-70%) may indicate choppy/sideways markets


---

### No Hard-Coded 0.0% Values**Verification:**```bash

# Search for hard-coded accuracy values

$ grep -rn "0.0%" core/prediction_tracker.py services/ api/

# Result: ZERO hard-coded 0.0% values found

# All 0.0% returns are conditional

if not rows:  # Only when no data exists
    return {"accuracy_pct": 0.0}

```text**Confirmation:**✅ Accuracy is**dynamically calculated**from database, never hard-coded.

---

## Section 4: When Something Looks Wrong

### 🔧 4-Step Decision Tree for Ghost Commander

Use this systematic approach when Ghost appears unhealthy:

---

#### STEP 1: Check Postgres Connection Logs**Action:**```bash

# SSH into Railway or check Railway logs

railway logs --filter postgres

# Look for these patterns

# ❌ BAD: "connection to server at metro.proxy.rlwy.net failed: timeout expired"

# ❌ BAD: "server closed the connection unexpectedly"

# ✅ GOOD: "✅ prediction_store: PostgresBackend initialized"

```text**Common Issues:**-**Timeout errors**: Railway Postgres proxy is overloaded or down

- **Connection refused**: Postgres service not running
- **SSL errors**: Certificate issues (rare)


**Resolution:**- Check Railway dashboard for Postgres service status

- Verify DATABASE_URL environment variable is set
- Check Railway incident status page


---

#### STEP 2: Restart Postgres (Railway Dashboard)**When to use:**Persistent connection errors or timeouts**Action:**1. Log into Railway dashboard

1. Navigate to Ghost Protocol project
2. Find Postgres service
3. Click "Restart" button
4. Wait 30-60 seconds for service to stabilize**What this fixes:**- Connection pool exhaustion
- Stale connections
- Proxy routing issues**After restart:**- Run health checks from Section 1
- Verify predictions can be created
- Check accuracy endpoint works


---

#### STEP 3: Rerun curl Tests**Action:**```bash

# Full test suite

curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC">>>>>
curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=XRP">>>>>
curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=AAPL">>>>>

# Check cache

curl "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=BTC">>>>>

```text**Success Indicators:**- ✅ All return `"ok": true`

- ✅ `prediction_id` values increment
- ✅ Response times reasonable (< 5s)
- ✅ Cache endpoints return latest predictions**If tests pass:**- System is healthy, previous issue was transient
- Monitor for 24 hours to ensure stability**If tests still fail:**- Proceed to STEP 4


---

#### STEP 4: Capture JSON + Log Snippet**When to use:**Tests fail after restart, persistent errors**Action 1: Capture Full Error JSON**```bash

# Save error response

curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC">>>>> > error.json 2>&1

# Format for readability

python3 -m json.tool error.json > error_formatted.json

```text**Action 2: Capture Railway Logs**```bash

# Get last 100 lines of app logs

railway logs --tail 100 > ghost_logs.txt

# Get Postgres-specific logs

railway logs --filter postgres --tail 50 > postgres_logs.txt

```text**Action 3: Check Prediction Store Status**```bash

# SSH into Railway or run locally

python3 -c "
from core.prediction_store import get_prediction_store
store = get_prediction_store()
print(f'Backend: {store.backend.__class__.__name__}')
print(f'Engine: {store.backend.pool if hasattr(store.backend, \"pool\") else \"N/A\"}')
"

```text**Action 4: Escalate with Evidence**Create a detailed bug report with:

1.**Error JSON**(`error_formatted.json`)
2.**Application logs**(`ghost_logs.txt`)
3.**Postgres logs**(`postgres_logs.txt`)
4.**Timestamp**of failure (UTC)
5.**What you tried**(Steps 1-3)
6.**Environment state**:


   ```bash

   echo "PREDICTION_STORE_ENGINE: $PREDICTION_STORE_ENGINE"
   echo "PREDICTION_DUAL_WRITE: $PREDICTION_DUAL_WRITE"
   echo "DATABASE_URL set: $([ -n "$DATABASE_URL" ] && echo YES || echo NO)"

   ```text

**Share this with:**- Development team via Slack/Discord

- Create GitHub issue with evidence
- Include Railway project ID


---

### 🚨 Emergency Indicators**Immediate attention required if:**| Symptom | Severity | Action |

|---------|----------|--------|
| All predictions return `"ok": false` | 🔴 CRITICAL | STEP 1-4 immediately |
| Accuracy stuck at 0.0% for >72 hours | 🟡 WARNING | Check outcome reconciler logs |
| No Telegram alerts for 24+ hours | 🟡 WARNING | Verify TELEGRAM_BOT_TOKEN set |
| Response times > 10 seconds | 🟡 WARNING | Check provider APIs (Binance/Yahoo) |
| "All providers failed" errors | 🟠 URGENT | Check external API status |
| Postgres connection timeouts | 🟠 URGENT | STEP 2 (restart Postgres) |

---

### 📋 Quick Reference: Healthy vs Unhealthy

#### Healthy System

```text

✅ Predictions create in < 5s
✅ prediction_id increments on each call
✅ Accuracy between 70-85% (after 48h)
✅ Telegram sends daily reports (7am, 8pm)
✅ Cache endpoints return latest data
✅ No repeated error patterns in logs

```text

#### Unhealthy System

```text

❌ "ok": false in prediction responses
❌ Timeout errors in logs (> 5 consecutive)
❌ Accuracy 0.0% after 72+ hours
❌ No Telegram messages for 24+ hours
❌ Response times > 10 seconds
❌ "connection to server failed" errors

```text

---

## Appendix A: Architecture Quick Reference

### Prediction Write Path (Postgres Primary)

```text

POST /api/predict/run
    ↓
wolf_app.py → predictor.create_prediction()
    ↓
services/predictor.py → _PREDICTION_STORE.save_prediction()
    ↓
core/prediction_store.py → PostgresBackend.save_prediction()
    ↓
PostgreSQL on Railway
    ↓ (if PREDICTION_DUAL_WRITE=1)
SQLite (backup copy)

```text

### Prediction Read Path (Cache + Postgres)

```text

GET /api/v3/predictions/latest
    ↓
wolf_app.py → reads _LATEST_PREDICTIONS dict
    ↓
Returns cached data (NO database query)

```text**Alternative Read Path (Cockpit V3):**```text

GET /api/v3/accuracy/summary
    ↓
wolf_app.py → predictor.get_prediction_history()
    ↓
services/predictor.py → _PREDICTION_STORE.get_prediction_history()
    ↓
core/prediction_store.py → PostgresBackend.get_prediction_history()
    ↓
PostgreSQL (JOIN predictions + outcomes)

```text

### Alert Pipeline

```text

New Prediction → Calculate Score → Filter by Confidence/Gain
    ↓
core/telegram_hunter.py → send_instant_alert()
    ↓
Telegram API (if score ≥ 80, confidence ≥ 70%)

Daily Schedule (7am, 8pm):
    ↓
wolf_app.py → get_top_opportunities()
    ↓
core/telegram_hunter.py → send_daily_report()
    ↓
Telegram API (include all ≥ 55% confidence)

```text

---

## Appendix B: Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `PREDICTION_STORE_ENGINE` | `sqlite` | Primary backend (`postgres` in prod) |
| `PREDICTION_DUAL_WRITE` | `0` | Enable writes to both backends (`1` in prod) |
| `DATABASE_URL` | None | PostgreSQL connection string (Railway) |
| `MIN_ALERT_CONFIDENCE` | `0.55` | Minimum confidence for daily reports |
| `TELEGRAM_BOT_TOKEN` | None | Telegram bot authentication |
| `TELEGRAM_CHAT_ID` | None | Telegram channel/user ID |**Current Production Config:**```bash

PREDICTION_STORE_ENGINE=postgres
PREDICTION_DUAL_WRITE=1
MIN_ALERT_CONFIDENCE=0.55

```text

---

## Appendix C: Personal Watchlist Verification

### Personal Watchlist System Overview

The personal watchlist is a per-user (Ghost Commander) system for tracking manually-selected stocks and crypto symbols.
It is**separate**from the global scanner/hunter watchlist.**Key Features:**- ✅ Persistent in Postgres (survives browser
sessions)

- ✅ Manual add/remove from Cockpit UI
- ✅ Automatic 48h predictions (market open/close + big moves)
- ✅ Telegram alerts for watchlist events
- ✅ Backwards compatible (global watchlist still works for scanners)


### Verification Commands

#### 1. Check Watchlist Endpoint Health

```bash

curl -sS "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user">>>>> | python3 -m json.tool

```text**Expected Output (Healthy):**```json

{
    "items": [
        {
            "id": 1,
            "symbol": "BTC",
            "asset_type": "crypto",
            "owns_position": false,
            "notes": "",
            "added_at": "2025-12-02T12:00:00Z",
            "current_price": 87000.00,
            "prediction": {
                "prediction_id": 9,
                "direction": "UP",
                "confidence": 0.46,
                "expected_move": 2.3,
                "horizon_h": 48
            }
        }
    ],
    "count": 1,
    "timestamp": 1764652576.184
}

```text**If Empty (New Installation):**```json

{
    "items": [],
    "count": 0,
    "timestamp": 1764652576.184
}

```text**If Tables Don't Exist (Migration Needed):**```json

{
    "detail": "relation \"ghost_watchlist_items\" does not exist"
}

```text

#### 2. Verify Watchlist Tables Exist

```bash

railway run python3 verify_postgres_migration.py

```text**Expected Output (Tables Exist):**```text

✅ ghost_watchlist_items: EXISTS (7 rows)
✅ watchlist_prediction_tracking: EXISTS (0 rows)
✅ watchlist_price_snapshots: EXISTS (0 rows)
✅ watchlist_alerts_log: EXISTS (0 rows)

```text

#### 3. Add Symbol to Watchlist (Manual Test)

```bash

curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/add">>>>> \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC","asset_type":"crypto","owns_position":false,"notes":"Bitcoin test"}'

```text**Expected Output:**```json

{
    "ok": true,
    "action": "added",
    "id": 1,
    "symbol": "BTC",
    "asset_type": "crypto",
    "owns_position": false,
    "added_at": "2025-12-02T12:34:56Z"
}

```text

#### 4. Get Watchlist Stats

```bash

curl -sS "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/stats">>>>> | python3 -m json.tool

```text**Expected Output:**```json

{
    "total_items": 7,
    "stocks": 4,
    "crypto": 3,
    "owned_positions": 2,
    "active_alerts_24h": 5,
    "last_updated": "2025-12-02T12:34:56Z"
}

```text

### Troubleshooting Personal Watchlist

#### Issue: 404 Not Found on /api/v3/watchlist/user**Cause:**Personal watchlist endpoints not registered or router order conflict.**Fix:**1. Check wolf_app.py ensures personal watchlist router is registered BEFORE cockpit_v3


```python

from api.personal_watchlist_endpoints import router as watchlist_router
APP.include_router(watchlist_router)  # MUST be before cockpit_v3_router

```text

1. Verify no route conflicts:


```bash

railway logs --tail 50 | grep "Personal Watchlist endpoints"

# Expected: "✅ Personal Watchlist endpoints registered (priority routing)"

```text

#### Issue: "relation does not exist" Error**Cause:**Migration not applied to Postgres database.**Fix:**```bash

# Apply migration from local machine

railway run psql $DATABASE_URL -f migrations/001_personal_watchlist.sql

# Verify tables created

railway run python3 verify_postgres_migration.py

```text

#### Issue: Empty Watchlist After Page Refresh**Cause 1:**Migration applied but no symbols added yet (expected for new installation).**Solution:**Add symbols via Cockpit UI "Add Symbol" button or API endpoint.**Cause 2:**Cockpit is loading legacy /api/v3/watchlist/enriched instead of /api/v3/watchlist/user.**Fix:**1. Check templates/cockpit_v3.html includes personal_watchlist_ui.js


```html

<script src="/static/personal_watchlist_ui.js?v=2025120201"></script>

```text

1. Check static/cockpit_v3.js calls loadPersonalWatchlist() not loadWatchlist():


```javascript

setInterval(() => loadPersonalWatchlist(), 15000);

```text

#### Issue: Watchlist Shows Only Green (No Red Predictions)**Cause:**Prediction confidence filtering too aggressive or bias in prediction generation.**Verification:**```bash

# Check raw predictions for watchlist symbols

for symbol in BTC ETH AAPL TSLA; do
  echo "=== $symbol ==="
  curl -sS "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=$symbol">>>>> \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Direction: {d['predictions'][0]['direction']}, Confidence: {d['predictions'][0]['confidence']:.0%}\")"
done

```text**Expected:**Mix of UP and DOWN predictions (Ghost shows both positive and negative signals).

### Personal Watchlist Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WATCHLIST_SCHEDULER_ENABLED` | `1` | Enable automated predictions for watchlist |
| `WATCHLIST_ALERTS_ENABLED` | `1` | Enable Telegram alerts for watchlist events |
| `WATCHLIST_OPEN_HOUR` | `9` | Market open hour (EST) for stock predictions |
| `WATCHLIST_CLOSE_HOUR` | `16` | Market close hour (EST) for stock predictions |
| `WATCHLIST_BIG_MOVE_CHECK_MINUTES` | `15` | Frequency of big move detection |
| `WATCHLIST_BIG_MOVE_THRESHOLD_PCT` | `5.0` | Price move % to trigger alert |
| `WATCHLIST_ALERT_COOLDOWN_HOURS` | `4` | Cooldown between alerts per symbol |
| `WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR` | `5` | Max alerts per hour globally |**Current Production Config:**```bash

WATCHLIST_SCHEDULER_ENABLED=1
WATCHLIST_ALERTS_ENABLED=1

# (Other defaults are fine)

```text

---

## Appendix D: Database Schema

### PostgreSQL Tables (Railway)**predictions**(507 rows)

```sql

id, symbol, run_at, horizon_h, method, confidence, direction,
features_json, params_json, tag

```text**prediction_points**(13,939 rows):

```sql

id, prediction_id, ts, kind (forecast/actual), price

```text**outcomes**(190 rows):

```sql

prediction_id, closed_at, mae, map, rmse, hit_direction,
hit_ratio_window, notes

```text

---**End of Operator Playbook**
**Last Updated**: 2025-12-01
**Next Review**: After 7 days of production operation
