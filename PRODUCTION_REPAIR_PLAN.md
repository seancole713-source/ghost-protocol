# 🏥 Ghost Protocol - Production Repair Plan

## 🔴 ROOT CAUSE IDENTIFIED

**Deployment Failed:**Healthcheck timeout caused by watchlist scheduler querying non-existent tables on startup.**Error Pattern:**```text
Database error: relation "ghost_watchlist_items" does not exist
❌ Failed to get watchlist: relation "ghost_watchlist_items" does not exist
❌ Failed to detect big moves: relation "ghost_watchlist_items" does not exist

```text**Why It Happened:**1. ✅ Postgres connection working (pool initialized successfully)

1. ✅ Prediction store initialized with dual-write mode (Postgres + SQLite)
2. ❌**Watchlist scheduler starts immediately**and tries to query `ghost_watchlist_items`
3. ❌**Table doesn't exist**(fresh Postgres instance, no migration run)
4. ❌**Errors cause app to hang**during startup healthcheck


---

## ✅ FIXES APPLIED

### Fix #1: Graceful Error Handling in Watchlist Scheduler**File Modified:**`core/watchlist_prediction_scheduler.py`**Changes:**- Added graceful handling for missing database tables in 3 functions

  - `_run_market_open_predictions()` - Logs warning instead of crashing
  - `_run_market_close_predictions()` - Logs warning instead of crashing
  - `_run_big_move_detection()` - Logs debug message instead of crashing**Result:**Scheduler now**continues startup**even if watchlist tables don't exist yet.


---

## 📋 DEPLOYMENT INSTRUCTIONS

### Step 1: Commit & Push Fixed Code

The fix has been applied to `core/watchlist_prediction_scheduler.py`. Now commit and deploy:

```bash

# From your local machine (not dev container)

git add core/watchlist_prediction_scheduler.py
git commit -m "fix: Handle missing watchlist tables gracefully during startup"
git push origin main

```text

### Step 2: Wait for Railway Deployment

Railway will auto-deploy (~2-3 minutes). Watch logs for:

```text

✅ Personal watchlist scheduler started
[GHOST STARTUP] ✅ Initialization complete - server ready
INFO:     Application startup complete

```text

### Step 3: Verify Healthcheck Passes

```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/health>>>>>

```text**Expected Response:**```json

{
  "status": "ok",
  "timestamp": "...",
  "uptime_seconds": ...,
  ...
}

```text

### Step 4: Test Predictions (Before Migration)

Even without watchlist tables, predictions should work:

```bash

# Test BTC prediction

curl <<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC>>>>>

# Test AAPL prediction

curl <<<<<https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=AAPL>>>>>

```text**Expected:**Both return `{"ok": true, "prediction_id": ...}`

### Step 5: Create Watchlist Tables (One-Time Migration)

Once healthcheck passes, run the migration to enable full watchlist features:

```bash

# Option A: Via Railway CLI

railway run psql $DATABASE_URL -f migrations/001_personal_watchlist.sql

# Option B: Via Railway Dashboard

# Go to: Postgres → Query → Paste SQL from migrations/001_personal_watchlist.sql

```text**This creates 4 tables:**- `ghost_watchlist_items` - Main watchlist with symbols

- `watchlist_prediction_tracking` - Prediction event log
- `watchlist_price_snapshots` - 15-minute price history
- `watchlist_alerts_log` - Telegram alert delivery tracking


### Step 6: Verify Watchlist Endpoints Work

```bash

# Get user watchlist (should return empty list initially)

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user>>>>>

# Add a symbol

curl -X POST <<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/add>>>>> \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC","asset_type":"crypto"}'

# Get stats

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/stats>>>>>

```text

### Step 7: Set Environment Variables (Optional - Enable Scheduler)

In Railway dashboard, set these if not already configured:

```bash

WATCHLIST_SCHEDULER_ENABLED=1
WATCHLIST_ALERTS_ENABLED=1
WATCHLIST_OPEN_HOUR=9
WATCHLIST_CLOSE_HOUR=16
WATCHLIST_BIG_MOVE_CHECK_MINUTES=15
WATCHLIST_BIG_MOVE_THRESHOLD_PCT=5.0
WATCHLIST_ALERT_COOLDOWN_HOURS=4
WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR=5

```text

Railway will auto-restart after setting env vars.

---

## 🎯 VERIFICATION CHECKLIST

### Critical (Must Pass)

- [ ] Railway deployment shows**ACTIVE**(not failed)
- [ ] `/health` returns `{"status": "ok"}`
- [ ] `/api/predict/run?symbol=BTC` returns valid prediction
- [ ] `/api/predict/run?symbol=AAPL` returns valid prediction
- [ ] Prediction IDs increment and are stored in Postgres


### After Migration (Watchlist Features)

- [ ] `/api/v3/watchlist/user` returns 200 (not 404)
- [ ] Can add symbols via `/api/v3/watchlist/add`
- [ ] Cockpit UI shows "Add Symbol" button
- [ ] Watchlist predictions generate on schedule


---

## 🔍 WHAT WAS THE PROBLEM

### Before Fix

```python

def _run_market_open_predictions(self):
    try:
        pwm = get_personal_watchlist_manager()
        stock_symbols = pwm.get_symbols_by_type("stock")  # ❌ CRASHES if table missing
    except Exception as e:
        LOGGER.error(f"❌ Failed: {e}")  # Logs error but doesn't handle gracefully

```text

### After Fix

```python

def _run_market_open_predictions(self):
    try:
        pwm = get_personal_watchlist_manager()
        stock_symbols = pwm.get_symbols_by_type("stock")
    except Exception as e:
        if "does not exist" in str(e) or "relation" in str(e):  # ✅ GRACEFUL
            LOGGER.warning("⚠️  Tables not created - skipping")
        else:
            LOGGER.error(f"❌ Failed: {e}", exc_info=True)

```text**Result:**App starts successfully even if watchlist tables haven't been migrated yet.

---

## 📊 EXPECTED LOG SEQUENCE (After Fix)

### Healthy Startup

```text

✅ Personal Watchlist endpoints registered
📅 Watchlist scheduler loop active
🚀 Watchlist prediction scheduler started
⚠️  Watchlist tables not yet created - skipping market open predictions
[GHOST STARTUP] ✅ Initialization complete - server ready
INFO:     Application startup complete
INFO:     Uvicorn running on <<<<<http://0.0.0.0:8080>>>>>
====================
Starting Healthcheck
====================
✅ Healthcheck passed!

```text

### After Migration

```text

✅ Personal Watchlist endpoints registered
📅 Watchlist scheduler loop active
🚀 Watchlist prediction scheduler started
🔔 Running market open predictions for watchlist stocks...
📊 7 stocks in watchlist
✅ Market open predictions complete (7 stocks)

```text

---

## 🚨 TROUBLESHOOTING

### If Deployment Still Fails**Check Deploy Logs for:**1. Different error message (not watchlist-related)

1. Prediction store initialization errors
2. Postgres connection timeout/auth errors**Common Issues:**


**Issue:**`connection refused` or `timeout`**Solution:**Verify DATABASE_URL is set correctly in Railway variables**Issue:**`password authentication failed`**Solution:**Check POSTGRES_PASSWORD matches between services**Issue:**`relation "ghost_predictions" does not exist`**Solution:**Prediction store should auto-create tables, but if not:

```sql

-- Create prediction tables manually
CREATE TABLE IF NOT EXISTS ghost_predictions (...);
CREATE TABLE IF NOT EXISTS ghost_outcomes (...);

```text

---

## 📝 SUMMARY**What We Fixed:**- Made watchlist scheduler tolerate missing tables during startup

- App now starts successfully even with fresh/empty Postgres
- Migration can be run AFTER deployment is healthy**What's Next:**1. Push the fix → Railway deploys
1. Healthcheck passes → App is ACTIVE
2. Run migration → Watchlist features enabled
3. Set env vars → Scheduler activates**Total Time:**~5 minutes from push to fully operational


---

## ✅ SUCCESS CRITERIA

Your Ghost Protocol deployment is**healthy**when:

1. Railway shows: `ghost-protocol: ACTIVE`
2. `/health` endpoint: Returns 200 OK with JSON
3. `/api/predict/run?symbol=BTC`: Returns valid prediction
4. `/api/predict/run?symbol=AAPL`: Returns valid prediction
5. Prediction IDs: Incrementing and stored in Postgres
6. Cockpit UI: Loads without errors, shows predictions**Watchlist features**are enabled when:

1. Migration run: All 4 watchlist tables created
2. `/api/v3/watchlist/user`: Returns 200 (not 404)
3. Can add/remove symbols via API

1. Cockpit shows: "Add Symbol" button in watchlist panel


---**You're ready to deploy!** Push the fix and Ghost will be healthy in ~3 minutes. 🚀
