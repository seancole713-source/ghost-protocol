# Railway Performance Fix - SQLite Lock Issue

## Problem
Railway deployment experiencing SQLite "database is locked" errors causing:
- Health endpoint timeouts (7-14 seconds)
- Prediction write failures
- Overall system slowdown

## Root Cause
Dual-write mode is enabled (`PREDICTION_DUAL_WRITE=1`), causing Ghost to write to both PostgreSQL and SQLite simultaneously. Under heavy load, SQLite locks cause contention.

Railway logs show:
```
[BNB] Failed to write to ghost_predictions table: database is locked
[WOLF] Failed to write to ghost_predictions table: database is locked
[TSLA] Failed to write to ghost_predictions table: database is locked
[AAPL] Failed to write to ghost_predictions table: database is locked
```

## Solution (Zero Code Changes - Guardian Mode Compliant)

### Option 1: Disable Dual-Write (Recommended)
**Action**: Set Railway environment variable:
```
PREDICTION_DUAL_WRITE=0
```

**Impact**:
- ✅ Eliminates SQLite lock contention
- ✅ PostgreSQL remains primary (production-grade)
- ✅ No code changes required
- ✅ Baseline unchanged

**Trade-off**:
- ❌ Lose SQLite backup (not critical - PostgreSQL is reliable)

### Option 2: Disable SQLite Writes Entirely
**Action**: Set Railway environment variable:
```
PREDICTION_STORE_ENGINE=postgres
PREDICTION_DUAL_WRITE=0
```

**Impact**:
- ✅ PostgreSQL only (cleanest solution)
- ✅ Zero SQLite operations
- ✅ Maximum performance

### Option 3: Increase SQLite Timeout (Temporary)
**Code Change Required** (breaks Guardian Mode - not recommended):
```python
# In SQLiteBackend._get_connection()
conn = sqlite3.connect(db_path, timeout=30.0)  # Default is 5.0
```

**Why Not Recommended**:
- ⚠️ Doesn't fix root cause
- ⚠️ Requires baseline code modification
- ⚠️ Just delays the lock issue

## Recommended Action

**Immediately apply Option 1** on Railway:

1. Go to Railway Dashboard → ghost-protocol → Variables
2. Find `PREDICTION_DUAL_WRITE`
3. Change value from `1` to `0`
4. Redeploy

**Expected Results**:
- Health endpoint: < 1 second (currently 7-14s)
- Zero "database is locked" errors
- Predictions write successfully to PostgreSQL only
- System fully operational

## Why This is Guardian Mode Compliant

✅ **Zero code changes** - environment variable only  
✅ **No baseline modification** - system behavior adjusted via config  
✅ **Reversible** - can re-enable dual-write anytime  
✅ **Non-destructive** - PostgreSQL remains fully functional  
✅ **Production-safe** - PostgreSQL is more reliable than SQLite for production

## Verification

After applying fix, verify:

```bash
# Health should respond instantly
curl -w "Time: %{time_total}s\n" https://ghost-protocol-production.up.railway.app/health

# Check logs for "database is locked" (should be zero)
railway logs | grep "database is locked"

# Verify predictions working
curl https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=BTC
```

## Long-term Consideration

If dual-write backup is desired in future:
- Consider async queue-based replication instead of synchronous dual-write
- Use PostgreSQL replication rather than SQLite
- Implement write-ahead logging (WAL) mode for SQLite if needed

But for now: **PostgreSQL alone is sufficient and more reliable**.
