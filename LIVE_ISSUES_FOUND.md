# Live Issues Diagnosed (Oct 6, 2025 - 4:56 PM EDT)

## 🚨 CRITICAL ISSUES FOUND

### 1. **ALL PRICE PROVIDERS ARE FAILING** ❌

**Root Causes:**

- **AlphaVantage**: Rate limited (25 requests/day FREE tier exceeded)

  - Error: "We have detected your API key...standard API rate limit is 25 requests per
    day"
  - Status: Circuit breaker OPEN (blocked)

- **Yahoo Finance HTTP**: Rate limited by Edge/Cloudflare

  - Error: "Edge: Too Many Requests"
  - Status: Circuit breaker OPEN (blocked)

- **YFinance (library)**: Circuit breaker HALF-OPEN

  - 2 failures, backoff_factor: 1

- **Polygon**: Only working provider but BLOCKLISTED for WOLF

  - Returns correct price ($24.37) but excluded by
    `PROVIDER_BLOCKLIST["WOLF"] = {"polygon"}`
  - Status: Circuit breaker CLOSED (working) but ignored

**Result**: Falling back to stale prev-close price ($24.37) indefinitely

______________________________________________________________________

### 2. **BACKGROUND PRICE UPDATER NOT RUNNING** ❌

**Evidence:**

- No logs containing "price_updater", "auto_refresh", or "price_updater_heartbeat"
- Searched ghost_server.out: "No updater logs found"
- Code was added to `wolf_app.py` but coroutine may not be scheduled properly

**Possible causes:**

- Server started before code changes were deployed
- Event loop not capturing task (silent failure)
- Logger not writing to expected output file

______________________________________________________________________

### 3. **PORTFOLIO STATE COMPLETELY NULL** ❌

**Evidence:**

```json
{
  "nav": null,
  "pnl": null,
  "holdings": null,
  "cash": null,
  "positions": null
}
```

**Root Cause:**

- `ghost_state.json` exists but contains only nulls
- Last modified: Sep 25 (11 days old!)
- No positions or cash data persisted
- User reports NAV $205.19 and -93% loss but API returns all nulls

**Impact**: Portfolio display broken, can't validate user's -93% loss claim

______________________________________________________________________

### 4. **POLYGON WORKING BUT ARTIFICIALLY BLOCKED** ⚠️

**Issue:**

```python
PROVIDER_BLOCKLIST: dict[str, set[str]] = {
    "WOLF": {"polygon"},
}
```

**Evidence:**

- Direct Polygon API call successful: `{"status": "OK", "results": 24.37}`
- But Ghost code excludes it from quorum for WOLF
- Comment says "Acceptance: never surface polygon as provider for WOLF if it disagrees"

**Result**: Only working provider is intentionally blocked!

______________________________________________________________________

### 5. **CIRCUIT BREAKERS STUCK IN EXPONENTIAL BACKOFF** ⚠️

**Current State (4:56 PM EDT):**

- AlphaVantage: OPEN (blocked until ~timestamps in future)
- Yahoo: OPEN (blocked)
- YFinance: HALF-OPEN (2 failures)
- All have `backoff_factor` >= 1

**Issue**: No automatic recovery mechanism; breakers stay tripped even if providers
recover

______________________________________________________________________

## 📊 PROVIDER TEST RESULTS (Live)

| Provider | Status | Result | Issue | |----------|--------|--------|-------| |
AlphaVantage | ❌ FAIL | Rate limit (25/day) | FREE tier exhausted | | Polygon | ✅ WORKS
| $24.37 | Artificially blocklisted | | Yahoo HTTP | ❌ FAIL | Edge rate limit | Too many
requests | | YFinance | ❌ FAIL | Half-open breaker | 2 prior failures |

**Conclusion**: Only Polygon works but is intentionally blocked for WOLF

______________________________________________________________________

## 🔧 REQUIRED FIXES

### Priority 1: IMMEDIATE (Enable Price Fetching)

1. **Remove Polygon from WOLF blocklist** (or upgrade AlphaVantage to paid tier)

   ```python
   # Line ~551 in wolf_app.py
   PROVIDER_BLOCKLIST: dict[str, set[str]] = {
       "WOLF": set(),  # Remove {"polygon"}
   }
   ```

2. **Add circuit breaker reset endpoint**

   ```python
   @APP.post("/debug/reset_breakers")
   async def debug_reset_breakers():
       for k in _PROVIDER_BREAKERS:
           _PROVIDER_BREAKERS[k] = {"state": "closed", "failures": 0, ...}
   ```

3. **Restart server to clear breaker backoff**

### Priority 2: FIX PORTFOLIO STATE

1. **Diagnose why ghost_state.json is all nulls**

   - Check state persistence logic (`_persist_save()`)
   - Verify STATE dict is actually populated
   - Last update was Sep 25 (11 days ago!)

2. **Initialize portfolio if missing**

   - Set cash balance
   - Add WOLF position with quantity
   - Trigger `_persist_save()`

### Priority 3: VERIFY BACKGROUND UPDATER

1. **Check server startup logs** for task scheduling
2. **Add explicit logging** to confirm coroutine is running
3. **Consider restart** to pickup new code if server predates changes

### Priority 4: UPGRADE OR DIVERSIFY PROVIDERS

1. **AlphaVantage**: Upgrade to paid tier (>25 req/day)
2. **Yahoo**: Implement proper backoff/retry with delays
3. **Polygon**: Remove from blocklist (it's the only working one!)
4. **Add fallback**: Consider IEX Cloud, Finnhub, or other providers

______________________________________________________________________

### Priority 5: FIX UI CLOCK SKIPPING ✅ FIXED

**Issue**: Clock seconds jump (0→15→30) because frontend only refreshes every 15
seconds\
**Fix Applied**: Added smooth client-side clock that updates every second

```javascript
// cockpit.html - Added smooth 1-second clock updater
setInterval(updateClock, 1000); // Smooth seconds display
```

______________________________________________________________________

## 🕐 TIMELINE CONTEXT

- **Current Time**: 4:56 PM EDT (Oct 6, 2025)
- **Market Status**: CLOSED (closed at 4:00 PM)
- **Last State Update**: Sep 25, 2025 (11 days old)
- **Diagnosis Window**: 4 minutes before end of useful data

______________________________________________________________________

## ✅ WHAT'S ACTUALLY WORKING

1. Server is healthy and responding
2. Polygon API returns correct data
3. Fusion/diagnostics endpoints operational
4. Trade card numeric values present

## ❌ WHAT'S BROKEN

1. All price providers except Polygon failing
2. Polygon blocked by intentional blocklist
3. Background price updater not logging (possibly not running)
4. Portfolio state completely null
5. Circuit breakers stuck in backoff with no reset mechanism
6. **UI Clock skipping seconds** ⚠️ - Frontend only updates every 15s (API refresh
   interval)

______________________________________________________________________

## 🚀 IMMEDIATE ACTION PLAN

```bash
# 1. Allow Polygon for WOLF (quickest fix)
# Edit wolf_app.py line ~551:
PROVIDER_BLOCKLIST = {"WOLF": set()}  # Remove polygon

# 2. Add breaker reset endpoint (add after line ~8274)
@APP.post("/debug/reset_breakers")
async def debug_reset_breakers():
    for k in _PROVIDER_BREAKERS:
        _PROVIDER_BREAKERS[k]["state"] = "closed"
        _PROVIDER_BREAKERS[k]["failures"] = 0
        _PROVIDER_BREAKERS[k]["backoff_factor"] = 0
    return {"ok": True, "breakers": _PROVIDER_BREAKERS}

# 3. Restart server
# pkill -f uvicorn
# uvicorn wolf_app:app --reload

# 4. Reset breakers
curl -X POST http://localhost:5000/debug/reset_breakers

# 5. Force refresh
curl -X POST http://localhost:5000/api/price/refresh

# 6. Verify
curl http://localhost:5000/api/price/diagnostics | jq .provider
# Should return "polygon" instead of "prev-close"
```
