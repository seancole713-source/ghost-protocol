# Backend Hardening Deployment Guide

## Changes Made (Commit d6ac7f8)

### 1. Enhanced Middleware
**Location**: `wolf_app.py` line ~9696

**New Features**:
- Guards against `None` responses (returns JSON 500)
- Catches all `BaseException` (Exception + CancelledError + ExceptionGroup)
- Safe logging (never crashes request path)
- Always returns JSON response

```python
@APP.middleware("http")
async def _log_requests(request, call_next):
    from starlette.responses import JSONResponse
    try:
        response = await call_next(request)
        if response is None:
            LOGGER.error("call_next returned None for %s %s", request.method, request.url.path)
            return JSONResponse({"error": "internal_error"}, status_code=500)
        return response
    except BaseException as e:
        try:
            LOGGER.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=e)
        except Exception:
            pass
        return JSONResponse({"error": "internal_error"}, status_code=500)
```

### 2. Global Exception Handlers
**Location**: `wolf_app.py` line ~205

**Handlers Added**:
- `RuntimeError` handler (catches "No response returned." specifically)
- `Exception` handler (catches all unhandled exceptions)
- `BaseException` handler (catches CancelledError, ExceptionGroup)

All handlers return JSON 500 with structured error messages.

### 3. Canary Crash Route
**Location**: `wolf_app.py` line ~16421

**Purpose**: Verify exception handling works correctly

```python
@APP.get("/api/_crash")
async def _crash():
    raise RuntimeError("boom")
```

Expected response: `{"error": "internal_error", "detail": "runtime_error"}` with HTTP 500

### 4. Verified Routes Exist
- `/api/tick` ✅ (line 16438)
- `/api/regime/current` ✅ (line 16443)

Both routes already exist in code. 404s in production were due to Railway not deploying.

---

## Deployment Steps

### Step 1: Trigger Railway Deployment

**Option A: Railway Dashboard** (Recommended)
1. Open https://railway.app
2. Navigate to "Ghost Sniper Bot" service
3. Click "Deployments" tab
4. Find commit `d6ac7f8` or click "Deploy Latest"
5. Wait for "Healthcheck succeeded!" (~2-3 minutes)

**Option B: Railway CLI**
```bash
railway up --detach
```

### Step 2: Verify Deployment

Wait for Railway to show "Deployment successful" with green checkmark.

Check version in logs or via endpoint:
```bash
export GHOST_BASE_URL="https://ghost-sniper-bot-seancole713-production.up.railway.app"
curl -s "$GHOST_BASE_URL/api/status" | jq .
```

Expected: `version` field should be updated

### Step 3: Test Exception Handling (Canary)

```bash
curl -s "$GHOST_BASE_URL/api/_crash"
```

**Expected Response**:
```json
{
  "error": "internal_error",
  "detail": "runtime_error"
}
```

**HTTP Status**: 500

**What This Proves**:
- Exception handlers are working
- Middleware catches exceptions
- No stack traces leak to clients
- Always returns JSON (never HTML error pages)

### Step 4: Test Previously Failing Routes

```bash
# Test /api/tick
curl -s "$GHOST_BASE_URL/api/tick"
# Expected: {"tick": N, "ts": 1731456789000}

# Test /api/regime/current
curl -s "$GHOST_BASE_URL/api/regime/current"
# Expected: {"regime": "neutral", "confidence": 0.5, ...}
```

### Step 5: Run Full Smoke Tests

```bash
cd /app
bash deployment_smoke_test.sh
```

**Expected Result**: 83-100% operational (8-9 endpoints passing)

### Step 6: Monitor Railway Logs

```bash
railway logs --tail 100
```

**Look for**:
- ✅ No 499 errors (middleware catches all exceptions)
- ✅ No "No response returned." errors
- ✅ All errors return JSON 500 (not HTML or empty responses)
- ✅ Structured error logging: "Unhandled error on GET /api/..."

**Watch for 5 minutes** to confirm stability.

---

## Expected Improvements

### Before Deployment
- ❌ 499 errors (middleware dropped responses)
- ❌ "No response returned." errors (Starlette bug)
- ❌ 6 endpoints returning 404 (routes not loaded)
- ❌ SSE stream timeout (0 events)
- ⚠️ 33% operational (3/9 endpoints)

### After Deployment
- ✅ No 499 errors (middleware always returns JSON 500)
- ✅ No "No response returned." (BaseException + None guard)
- ✅ 5 endpoints fixed (404 → 200)
- ⚠️ SSE stream may still need investigation
- ✅ 83-100% operational (8-9/9 endpoints)

---

## Troubleshooting

### Issue: /api/_crash returns HTML error page
**Cause**: Exception handlers not loaded or middleware not applied
**Fix**: Check Railway logs for startup errors, ensure deployment completed

### Issue: /api/tick still returns 404
**Cause**: Railway didn't deploy new code
**Fix**: 
1. Check Railway dashboard for deployment status
2. Verify commit hash in logs: `git log --oneline -1`
3. Manually trigger redeploy with "Clear Cache" option

### Issue: SSE stream still times out
**Cause**: Separate issue from 404s (async generator not yielding)
**Next Steps**:
1. Check Railway logs for "SSE:" debug messages
2. Read `/app/wolf_app.py` lines 11889-11938
3. Add immediate yield before `api_cockpit()` call
4. See SSE_STREAM_DEBUG.md (to be created)

### Issue: 500 errors in production logs
**Expected**: This is correct behavior!
**What to check**:
- Are responses JSON? ✅ Good
- Do logs show "Unhandled error on..."? ✅ Good
- Are stack traces in logs (not responses)? ✅ Good

**Action**: Investigate root cause of exceptions in application logic

---

## Validation Checklist

After deployment, verify:

- [ ] Railway shows "Deployment successful" (green checkmark)
- [ ] `/api/status` returns 200 OK
- [ ] `/api/_crash` returns JSON 500 (not HTML)
- [ ] `/api/tick` returns `{"tick": N, "ts": ...}` (not 404)
- [ ] `/api/regime/current` returns `{"regime": "neutral", ...}` (not 404)
- [ ] Railway logs show NO 499 errors for 5 minutes
- [ ] Railway logs show NO "No response returned." errors
- [ ] `deployment_smoke_test.sh` shows 83-100% operational

---

## Next Steps

1. **If all checks pass**: Monitor production for 24 hours
2. **If SSE still fails**: Create SSE_STREAM_DEBUG.md and investigate
3. **If new errors appear**: Check Railway logs for "Unhandled error on..." messages
4. **Enable Railway auto-deploy**: Settings → GitHub Integration → Auto-deploy on push

---

## Commit Details

**Commit**: d6ac7f8
**Branch**: main
**GitHub**: https://github.com/seancole713-source/ghost-protocol
**Previous**: 77e4971 (initial middleware fix)

**Files Changed**:
- `wolf_app.py` (+47 lines, -9 lines)

**Changes**:
1. Enhanced middleware with None guard + BaseException
2. Added 4 global exception handlers
3. Added `/api/_crash` canary route
4. Verified `/api/tick` and `/api/regime/current` exist

---

**Deployment Target**: Railway (https://ghost-sniper-bot-seancole713-production.up.railway.app)

**Expected Downtime**: 30-60 seconds during redeploy

**Rollback Plan**: Revert to commit 77e4971 if critical issues occur
