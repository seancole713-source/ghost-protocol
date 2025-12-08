# 🔧 CRITICAL RAILWAY ENVIRONMENT FIXES

**Date**: 2025-11-16
**Issue**: Web service deployment failing, cockpit completely blank
**Root Cause**: Price provider timeout settings too aggressive + safe price mode enabled

---

## ⚠️ IMMEDIATE FIXES REQUIRED IN RAILWAY

These THREE variable changes will fix the blank cockpit:

### Fix 1: Increase Provider Timeout (CRITICAL)

**Current Problem**: 1.0s timeout is TOO SHORT for Polygon API calls
**Impact**: All price fetches fail → delisted mode → safe overlay → cockpit dead

```bash

# Change these in Railway dashboard

PRICE_PROVIDER_TIMEOUT_S=2.5
REQUESTS_DEFAULT_TIMEOUT_S=3.0

```text

### Fix 2: Increase TTL to Reduce Load (CRITICAL)

**Current Problem**: 30s/60s TTL causes constant re-fetching
**Impact**: Rate limit exceeded → provider unauthorized → all data empty

```bash

# Change these in Railway dashboard

PRICE_TTL_S=120
PRICE_TTL_OPEN_S=300

```text

### Fix 3: Disable Safe/Seeded Price Mode (CRITICAL)

**Current Problem**: Fallback modes force degraded state
**Impact**: System refuses to operate when live prices unavailable

```bash

# Change these in Railway dashboard

ALLOW_SEEDED_PRICE=0
PRICE_FALLBACK_PREVCLOSE=0

```text

---

## ✅ CONFIRMED CORRECT (DO NOT CHANGE)

These are already set correctly in your Railway environment:

```bash

# API Keys (CORRECT)

POLYGON_API_KEY=(your key - confirmed present)
ALPHAVANTAGE_API_KEY=(your key - confirmed present)

# Core Settings (CORRECT)

SIM_MODE=0
PRICE_STRICT_LIVE=1
STOCK_PRICE_SOURCE=polygon
CRYPTO_PRICE_SOURCE=coingecko
PRICE_MIN_PROVIDERS=1

# Agent/AI (CORRECT)

OPENAI_API_KEY=(present)
AI_PROVIDER=openai
AGENT_MODEL=gpt-4

# Database (CORRECT)

REDIS_URL=(configured)
WOLF_SQLITE_PATH=data/wolf.db
PREDICTION_DB_PATH=data/predictions.db

# Ghost Config (CORRECT)

WOLF=WOLF
DEFAULT_QTY=1000
DEFAULT_AVG=2.45
FOCUS_WOLF_ONLY=0  # Good - allows flexible symbol selection

```text

---

## 📋 STEP-BY-STEP FIX PROCEDURE

### Step 1: Update Railway Variables

1. Log into Railway dashboard: <<<<<https://railway.app>>>>>
2. Navigate to your project: `ghost-protocol` (or your project name)
3. Click on the **web**service (the one that's failing)
4. Go to**Variables**tab
5. Find and update these 5 variables:


| Variable | OLD Value | NEW Value |
|----------|-----------|-----------|
| PRICE_PROVIDER_TIMEOUT_S | 1.0 | 2.5 |
| REQUESTS_DEFAULT_TIMEOUT_S | 1.5 | 3.0 |
| PRICE_TTL_S | 30 | 120 |
| PRICE_TTL_OPEN_S | 60 | 300 |
| ALLOW_SEEDED_PRICE | 1 | 0 |
| PRICE_FALLBACK_PREVCLOSE | 1 | 0 |

1. Click**Save**or**Deploy**after each change


### Step 2: Trigger Redeploy

Railway should auto-deploy after variable changes. If not:

1. Go to**Deployments**tab
2. Click**Deploy**button
3. Wait 2-3 minutes for build + startup


### Step 3: Verify Deployment Success

Check deployment logs for:

```text

✅ [GHOST BOOT] Environment flags: {...}
✅ prometheus_metrics_registered
✅ stage1_initialized
✅ forecast_grid_ready
✅ env_validation_passed  ← KEY: Must see this, not "env_validation_failed"
✅ Scheduled predictions enabled
✅ background_price_updater_started
✅ Application startup complete

```text**If you see**:

```text

❌ env_validation_failed
❌ violations: ["ALLOW_SAFE_PRICE must be 0", "PRICE_FALLBACK_PREVCLOSE must be 0"]

```text

Then the safe price variables are still set to 1.

### Step 4: Test Cockpit

1. Visit: `https://your-railway-url.up.railway.app/cockpit`
2. Check top banner:
   - ✅ Should show: `mode: live` with running clock
   - ✅ Should NOT show: `DELISTED MODE PROVIDER UNAUTHORIZED`
   - ✅ `tick:` should show latency (not `n/a`)

1. Check panels populate with data:
   - ✅ Ghost-AI Monitor: Confidence, decisions showing numbers
   - ✅ World Context: SPY, VIX showing prices
   - ✅ Market Regime: Fields populating
   - ✅ Ghost Prediction: Table has rows


### Step 5: Verify API Endpoints

```bash

# Test multi-run (should return non-zero counts)

curl <<<<<https://your-railway-url.up.railway.app/api/predictions/multi/run>>>>> | jq '.counts'

# Expected: {"stocks": X, "crypto": Y, "vip": Z} where X+Y+Z > 0

# Test health

curl <<<<<https://your-railway-url.up.railway.app/api/health/predictions>>>>> | jq '.ghost_score_v2'

# Expected: {"score": XX, "grade": "B", ...}

# Test cockpit

curl <<<<<https://your-railway-url.up.railway.app/api/cockpit>>>>> | jq '.ghost_2x.score'

# Expected: numeric value (not NO_DATA)

```text

---

## 🐛 WHAT WAS WRONG

### Problem 1: Timeout Too Short

```text

PRICE_PROVIDER_TIMEOUT_S=1.0  ← Polygon needs 2-4 seconds

```text

**Effect**:

- Every price fetch timed out after 1 second
- Polygon returned partial data or error
- System flagged as "provider unauthorized"
- Cascade failure → all panels empty


### Problem 2: TTL Too Short

```text

PRICE_TTL_S=30  ← Re-fetch every 30 seconds
PRICE_TTL_OPEN_S=60  ← Re-fetch every 60 seconds during market hours

```text

**Effect**:

- Constant hammering of Polygon API
- Rate limit exceeded
- Provider blocks requests
- System enters degraded mode


### Problem 3: Safe Price Mode Enabled

```text

ALLOW_SEEDED_PRICE=1  ← Allows fallback to previous close
PRICE_FALLBACK_PREVCLOSE=1  ← Uses previous close when live fails

```text

**Effect**:

- When real providers fail, system uses "safe" previous close
- Triggers "Corporate action/price anomaly — using safe price overlay"
- System flags itself as degraded
- Blocks all other functionality
- Cockpit shows "DELISTED MODE PROVIDER UNAUTHORIZED"


---

## 🎯 WHY THESE FIXES WORK

### Fix 1: Longer Timeout

- Allows Polygon API calls to complete normally
- Reduces false "provider unauthorized" errors
- Enables price fetches to succeed


### Fix 2: Longer TTL

- Reduces request frequency
- Stays under Polygon rate limits
- Prevents provider blocking
- Price data still fresh enough for trading


### Fix 3: Disable Safe Mode

- Forces system to use LIVE data only
- No fallback to previous close
- No "safe price overlay" triggering
- System either gets real data or returns explicit error (not silent degraded mode)


---

## 🔍 VERIFICATION CHECKLIST

After applying fixes, verify:

- [ ] Railway deployment shows "Active" (not "Failed")
- [ ] Logs show `env_validation_passed`
- [ ] Logs show `background_price_updater_started`
- [ ] Cockpit loads without "DELISTED MODE PROVIDER UNAUTHORIZED"
- [ ] Cockpit shows live clock (not stopped)
- [ ] Cockpit shows `tick: XXXms` (not `n/a`)
- [ ] Ghost-AI panels populate with data (not `—`)
- [ ] World Context shows SPY/VIX prices (not `—`)
- [ ] `/api/predictions/multi/run` returns counts > 0
- [ ] Ghost 2.x Health shows numeric score (not NO_DATA)


---

## 🚨 IF STILL FAILING AFTER FIXES

If cockpit is still blank after applying all 6 variable changes:

### Check 1: Deployment Logs

```bash

railway logs --service web

```text

Look for:

- Import errors
- Module not found
- Database connection failures
- API key invalid (not just missing)


### Check 2: API Key Validity

Test Polygon key directly:

```bash

curl "<<<<<https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey=$(railway>>>>> variables get POLYGON_API_KEY)"

```text

Should return data, not `{"error":"unauthorized"}`

### Check 3: Network/Firewall

Railway might block outbound calls to Polygon. Check if curl works from Railway:

```bash

railway run curl <<<<<https://api.polygon.io/v1/meta/symbols/AAPL/company>>>>>

```text

### Check 4: PORT Binding

Verify web service binds to correct port:

```bash

railway logs --service web | grep "Uvicorn running"

```text

Should see: `Uvicorn running on <<<<<http://0.0.0.0:8444`>>>>>

---

## 📊 EXPECTED OUTCOME

**Before Fixes**:

- ❌ Cockpit shows "DELISTED MODE PROVIDER UNAUTHORIZED"
- ❌ All panels show `—` or empty
- ❌ SSE stream dead (no updates)
- ❌ Multi-run returns 0 predictions
- ❌ Deployment fails or restarts constantly


**After Fixes**:

- ✅ Cockpit shows "mode: live" with running clock
- ✅ Panels populate with real data
- ✅ SSE stream updates every 5-10 seconds
- ✅ Multi-run returns 10-20 predictions
- ✅ Deployment stable (no restarts)
- ✅ Ghost 2.x Health shows green/yellow (not critical)


---

**Apply these fixes now and report back with deployment status.**
