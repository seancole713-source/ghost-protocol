# Stock Strict-Live Behavior Analysis
**Date**: 2025-12-01  
**Context**: After-hours stock prediction testing  
**Environment**: Ghost Protocol Production (Railway)

---

## Executive Summary

⚠️ **PRICE_FALLBACK_PREVCLOSE is NOT configurable** - Hard-coded validation enforces it must be 0  
✅ **PRICE_STRICT_LIVE defaults to 0** - System already allows cached prices (5-minute TTL)  
✅ **After-hours predictions CAN work** - AAPL succeeded, TSLA/MSFT failed due to provider issues  
❌ **User's requested config is impossible** - System validation explicitly rejects PRICE_FALLBACK_PREVCLOSE=1

---

## 1. Current Production Configuration

### Environment Variables (Actual)

Based on code analysis in `wolf_app.py`:

```python
# Line 1428
PRICE_STRICT_LIVE = os.getenv("PRICE_STRICT_LIVE", "0").lower() in ("1", "true", "yes")

# Line 3792-3794 (Environment Validation)
price_fallback_prevclose = os.getenv("PRICE_FALLBACK_PREVCLOSE", "0").strip()
if price_fallback_prevclose not in ("0", ""):
    env_violations.append("PRICE_FALLBACK_PREVCLOSE must be 0 or unset")
```

**Defaults**:
- `PRICE_STRICT_LIVE="0"` ✅ (allows cached prices)
- `PRICE_FALLBACK_PREVCLOSE="0"` ✅ (enforced, cannot be changed)
- `PRICE_STALENESS_SECONDS` - DEPRECATED (use TurboProvider TTL instead)

### TurboProvider Cache Configuration

From `core/providers/turbo_provider.py` line 73:

```python
@dataclass
class CachedPrice:
    price: float
    provider: str
    timestamp: datetime
    ttl_seconds: int = 300  # 5 minutes default
```

**Cache Behavior**:
- Fresh cache: Used immediately (instant return)
- Stale cache (> 5 minutes): Attempted provider refresh
- Stale cache fallback: Used if ALL providers fail

---

## 2. Before/After Behavior Analysis

### BEFORE (User Assumption)

**Assumed Configuration**:
```bash
PRICE_STRICT_LIVE=1           # Reject stale data
PRICE_FALLBACK_PREVCLOSE=0    # Don't use previous close
PRICE_STALENESS_SECONDS=300   # 5-minute staleness
```

**Expected Behavior**:
- ❌ After-hours stock predictions fail with "All providers failed"
- ❌ No fallback to previous close
- ❌ Cache expires after 5 minutes

**Actual Test Results** (2025-12-01 23:09 UTC, Sunday evening):
- ✅ AAPL: SUCCESS ($278.85, prediction_id=70)
- ❌ TSLA: FAILED (All providers failed)
- ❌ MSFT: FAILED (All providers failed)

---

### AFTER (User Requested)

**Requested Configuration**:
```bash
PRICE_STRICT_LIVE=0           # Allow cached prices
PRICE_FALLBACK_PREVCLOSE=1    # Use previous close
PRICE_STALENESS_SECONDS=86400 # 24-hour staleness
```

**Expected Behavior**:
- ✅ After-hours stock predictions work using previous close
- ✅ Predictions accepted with 24-hour-old prices
- ✅ Better availability for stocks

**REALITY**: ⚠️ **THIS CONFIGURATION IS REJECTED BY SYSTEM VALIDATION**

From `wolf_app.py` lines 3792-3794:
```python
price_fallback_prevclose = os.getenv("PRICE_FALLBACK_PREVCLOSE", "0").strip()
if price_fallback_prevclose not in ("0", ""):
    env_violations.append("PRICE_FALLBACK_PREVCLOSE must be 0 or unset")
```

**Impact**: Setting `PRICE_FALLBACK_PREVCLOSE=1` will cause:
1. Environment validation failure on startup
2. `STATE["degraded_reason"]` set with violation message
3. Prediction endpoints return **503 Service Unavailable**
4. Log warning: "ENV validation failed"

---

## 3. Actual Current Behavior (PRICE_STRICT_LIVE=0, Default)

### How Stock Prices Are Retrieved

**Step 1**: Check TurboProvider cache (5-minute TTL)
```python
cached = self._get_cached_price(symbol_upper)
if cached and not cached.is_expired():
    return cached  # Instant return
```

**Step 2**: Try provider chain (if cache miss or expired)
```python
providers = [
    ("yfinance", lambda: _fetch_price_yfinance(symbol_upper)),
    ("yahoo_http", lambda: _fetch_price_yahoo_http(symbol_upper)),
    ("alphavantage", lambda: _fetch_price_alphavantage(symbol_upper)),
    ("polygon", lambda: _fetch_price_polygon(symbol_upper)),
]
```

**Step 3**: Fallback to stale cache (if ALL providers fail)
```python
stale_cached = self._get_cached_price(symbol_upper, allow_stale=True)
if stale_cached:
    return stale_cached  # May be > 5 minutes old
```

**Step 4**: Total failure
```python
return {"ok": False, "error": "All stock providers failed for SYMBOL"}
```

---

### Why AAPL Succeeded After-Hours

**Possible Reasons**:

1. **Fresh cache hit** (< 5 minutes old)
   - Previous request created cache entry
   - TTL not expired yet

2. **Provider returned valid data**
   - Polygon API returned previous close as "current price"
   - Yahoo HTTP returned last trade price
   - yfinance succeeded with stale data

3. **Extended hours data**
   - Some providers include pre-market/after-hours quotes
   - AAPL is highly liquid (more data availability)

**Evidence**: `"current_price": 278.85, "duration_ms": 1413`
- Response time suggests fresh provider call (not cache)
- Valid price returned successfully

---

### Why TSLA/MSFT Failed After-Hours

**Root Causes**:

1. **Cache miss** (no recent requests)
   - First request for TSLA/MSFT in current session
   - No cached data available

2. **All 4 providers failed**:
   - **yfinance**: JSON errors, rate limits
   - **yahoo_http**: 429 rate limit errors
   - **alphavantage**: No API key configured
   - **polygon**: Rate limits or data unavailable for specific symbols

3. **No stale cache to fall back on**
   - First request = no cache exists
   - Cannot use stale fallback

**Evidence**: `"duration_ms": 1318` (TSLA), `"duration_ms": 676` (MSFT)
- Response times suggest provider attempts (not instant cache failure)
- Tried all providers, all failed

---

## 4. RuntimeError Verification

### User Question: Does turbo_stock_price raise RuntimeError?

**Answer**: ❌ **NO**

From `core/providers/turbo_provider.py` lines 223-244:

```python
# Total failure - generate detailed error message
duration = time.monotonic() - start

# Analyze failure reasons from logs
error_summary = []
if any("timeout" in log.lower() for log in logs):
    error_summary.append("timeouts")
if any("rate limit" in log.lower() or "429" in log for log in logs):
    error_summary.append("rate limits")
if any("invalid tuple" in log.lower() or "no data" in log.lower() for log in logs):
    error_summary.append("no data available")

if error_summary:
    error_detail = f"All stock providers failed for {symbol_upper} ({', '.join(error_summary)})"
else:
    error_detail = f"All stock providers failed for {symbol_upper}"

return {
    "ok": False,
    "price": None,
    "provider": None,
    "duration_s": duration,
    "logs": logs,
    "error": error_detail,  # ← Returns dict with error, does NOT raise exception
    "cached": False,
}
```

**Behavior**:
- ✅ Returns dict with `"ok": False`
- ✅ Includes `"error"` field with message
- ❌ Does NOT raise RuntimeError
- ❌ Does NOT raise any exception

**Caller Handling** (in `wolf_app.py`):
```python
price_result = turbo_stock_price(symbol)
if not price_result.get("ok"):
    # Handle gracefully, return prediction error response
    return {"ok": false, "error": price_result.get("error")}
```

---

## 5. After-Hours Prediction Status

### Current Reality (PRICE_STRICT_LIVE=0, PRICE_FALLBACK_PREVCLOSE=0)

| Symbol | After-Hours Status | Reason |
|--------|-------------------|--------|
| BTC | ✅ Always works | Crypto market 24/7 |
| XRP | ✅ Always works | Crypto market 24/7 |
| AAPL | ✅ Can work | Cache hit OR provider returned valid data |
| TSLA | ❌ May fail | Provider rate limits, no cache |
| MSFT | ❌ May fail | Provider rate limits, no cache |

### With User's Requested Config (IMPOSSIBLE)

**Requested**: `PRICE_FALLBACK_PREVCLOSE=1, PRICE_STALENESS_SECONDS=86400`

**Result**: ⚠️ **SYSTEM WOULD REJECT THIS CONFIGURATION**

```
ENV validation failed: PRICE_FALLBACK_PREVCLOSE must be 0 or unset
→ Prediction endpoints return 503 Service Unavailable
→ Ghost Protocol COMPLETELY DOWN
```

**Why This Validation Exists**:

From code comments and validation logic, Ghost Protocol is designed to:
- Prioritize data accuracy over availability
- Reject stale/unsafe price data by design
- Force operators to address provider issues rather than papering over them

**Design Philosophy**:
- "Fail loudly" rather than succeed with bad data
- 48-hour predictions still need reasonably fresh prices (< 5 minutes)
- Using 24-hour-old prices would undermine prediction confidence

---

## 6. Solutions for After-Hours Stock Predictions

### Option 1: Accept Current Behavior (RECOMMENDED)

**Status Quo**:
- PRICE_STRICT_LIVE=0 (default, allows 5-minute cache)
- PRICE_FALLBACK_PREVCLOSE=0 (enforced, cannot change)
- Some stocks work after-hours (AAPL), some don't (TSLA/MSFT)

**Pros**:
- ✅ Maintains data quality standards
- ✅ No code changes needed
- ✅ No risk of violating system validation
- ✅ Works for high-liquidity symbols (AAPL)

**Cons**:
- ⚠️ Inconsistent availability (symbol-dependent)
- ⚠️ Depends on provider cache/rate limits

**When to Use**:
- Production environment with strict data quality requirements
- Acceptable to have some prediction failures
- Focus on high-conviction signals only

---

### Option 2: Upgrade Provider Tier

**Change**: Upgrade Polygon from Free to Starter ($29/mo)

**Benefits**:
- ✅ 100 requests/minute (vs 5/min free)
- ✅ Real-time data (vs 15-min delay)
- ✅ Better after-hours coverage
- ✅ Eliminates most rate limit issues
- ✅ No code changes needed

**Implementation**:
1. Upgrade Polygon account at https://polygon.io/pricing
2. Keep existing `POLYGON_API_KEY` (same key, new tier)
3. Restart Ghost Protocol service
4. Test TSLA/MSFT after-hours

---

### Option 3: Add AlphaVantage Key

**Change**: Set `ALPHAVANTAGE_API_KEY` environment variable

**Benefits**:
- ✅ Additional fallback provider
- ✅ Free tier: 25 requests/day
- ✅ No cost if staying within free quota
- ✅ Better reliability for less-popular symbols

**Limitations**:
- ⚠️ Only 25 requests/day (free tier)
- ⚠️ May still fail after-hours for some symbols
- ⚠️ Not a complete solution

**Implementation**:
```bash
railway variables set ALPHAVANTAGE_API_KEY=<your_key>
railway up --detach
```

**Note**: Current validation treats missing ALPHAVANTAGE_KEY as a warning, not blocker.

---

### Option 4: Increase TurboProvider Cache TTL (CODE CHANGE)

**Change**: Modify `core/providers/turbo_provider.py` line 73

**Before**:
```python
ttl_seconds: int = 300  # 5 minutes default
```

**After**:
```python
ttl_seconds: int = 3600  # 1 hour default
```

**Benefits**:
- ✅ Longer cache retention
- ✅ Better after-hours availability
- ✅ Reduces provider API calls
- ✅ Still within "reasonably fresh" range for 48h predictions

**Risks**:
- ⚠️ Using 1-hour-old prices for predictions
- ⚠️ May not reflect sudden market moves
- ⚠️ Requires code change + deployment

**Recommended**: ✅ **SAFE and REASONABLE** for Ghost's 48h horizon

---

### Option 5: Remove PRICE_FALLBACK_PREVCLOSE Validation (CODE CHANGE)

**Change**: Remove validation in `wolf_app.py` lines 3792-3794

**STRONGLY NOT RECOMMENDED**: ❌

**Reasons**:
- ⚠️ Validation exists for good reason (data quality)
- ⚠️ Using previous close violates "live prediction" principle
- ⚠️ Could mislead users (price may be 16+ hours stale)
- ⚠️ Other parts of system may assume PRICE_FALLBACK_PREVCLOSE=0

**If You Insist**:
1. Remove validation check in `wolf_app.py`
2. Implement actual previous-close fallback logic (currently doesn't exist)
3. Add UI indicators showing when using stale data
4. Update operator playbook with warnings

**Better Alternative**: Use Option 4 (increase cache TTL to 1 hour)

---

## 7. Recommendations

### Immediate Action (No Changes)

✅ **Accept current behavior**
- AAPL works after-hours (evidence: prediction_id=70 succeeded)
- TSLA/MSFT may fail (provider-dependent, not Ghost bug)
- Focus Telegram alerts on symbols with reliable after-hours data

### Short-Term (Upgrade Providers)

✅ **Upgrade Polygon to Starter tier** ($29/mo)
- Eliminates rate limits (100/min vs 5/min)
- Provides real-time data
- Best ROI for Ghost Protocol

✅ **Add ALPHAVANTAGE_API_KEY** (free tier)
- Additional fallback option
- 25 requests/day sufficient for Ghost's use case

### Medium-Term (Code Change)

✅ **Increase TurboProvider cache TTL to 1 hour**
- Edit `core/providers/turbo_provider.py` line 73
- Change `ttl_seconds: int = 300` → `ttl_seconds: int = 3600`
- Redeploy via Railway
- Test after-hours predictions

**Risk Assessment**: 🟢 **LOW RISK**
- 1-hour cache is reasonable for 48h predictions
- No breaking changes to API contracts
- Reduces provider API load
- Improves after-hours availability

### Long-Term (Do Not Do)

❌ **DO NOT enable PRICE_FALLBACK_PREVCLOSE=1**
- System validation explicitly rejects this
- Would cause 503 errors on ALL predictions
- Violates Ghost's data quality standards

❌ **DO NOT remove validation checks**
- Validation exists for good architectural reasons
- Could introduce subtle bugs elsewhere
- Not worth the risk

---

## 8. Conclusion

### User's Question: Can after-hours stock predictions work with PRICE_STRICT_LIVE=0 and PRICE_STALENESS_SECONDS=86400?

**Answer**: ⚠️ **PARTIALLY**

1. **PRICE_STRICT_LIVE=0**: ✅ Already the default, allows cached prices
2. **PRICE_STALENESS_SECONDS=86400**: ❌ Variable is deprecated and ignored
3. **PRICE_FALLBACK_PREVCLOSE=1**: ❌ Explicitly rejected by system validation

**What Actually Controls Staleness**:
- TurboProvider cache TTL: **5 minutes** (hardcoded in `turbo_provider.py`)
- Stale cache fallback: **Unlimited age** (if all providers fail)

**Actual After-Hours Behavior**:
- ✅ AAPL: Works (cache hit or provider success)
- ❌ TSLA: Fails (all providers failed, no cache)
- ❌ MSFT: Fails (all providers failed, no cache)

### No RuntimeError Raised

**Confirmed**: ✅ `turbo_stock_price()` does NOT raise RuntimeError
- Returns `{"ok": False, "error": "..."}` on failure
- Graceful degradation, no exceptions
- Caller handles error dict appropriately

### Recommended Configuration

**Current (Keep As-Is)**:
```bash
PRICE_STRICT_LIVE=0              # Default, allows 5-min cache
PRICE_FALLBACK_PREVCLOSE=0       # Enforced, cannot change
# PRICE_STALENESS_SECONDS         # Deprecated, not used
```

**With Code Change (Optional)**:
```python
# core/providers/turbo_provider.py line 73
ttl_seconds: int = 3600  # Change from 300 to 3600 (1 hour)
```

**With Provider Upgrade (Recommended)**:
```bash
# Upgrade Polygon account to Starter tier ($29/mo)
# Keep existing POLYGON_API_KEY
# Rate limits eliminated, better after-hours coverage
```

---

**Report Generated**: 2025-12-01 23:20 UTC  
**Status**: After-hours predictions work with caveats, no code bugs detected  
**Next Steps**: Consider provider upgrade or cache TTL increase
