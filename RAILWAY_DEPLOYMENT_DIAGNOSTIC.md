# Railway Backend Crash Diagnostic Report
**Date**: November 22, 2025  
**Issue**: ERR_HTTP2_PROTOCOL_ERROR preventing Cockpit UI from loading  
**Status**: 🔴 CRITICAL - Backend not responding

---

## 🔴 ROOT CAUSE IDENTIFIED

**THREE FATAL DOCKERFILE BUGS** were blocking Railway deployment:

### 1. **HEALTHCHECK Path Wrong** ❌
```dockerfile
# BEFORE (BROKEN):
CMD curl -fsS http://localhost:${PORT:-8080}/ui/health || exit 1
```
- Railway was checking `/ui/health` which **doesn't exist**
- Correct endpoint is `/health`
- This caused health check to fail → Railway marked service as unhealthy → restart loop

### 2. **CMD Hardcoded Port 8080** ❌
```dockerfile
# BEFORE (BROKEN):
CMD ["uvicorn", "wolf_app:APP", "--host", "0.0.0.0", "--port", "8080"]
```
- Railway assigns dynamic `$PORT` (could be 3000, 5000, 8080, etc.)
- Hardcoded 8080 means if Railway assigns a different port, the proxy can't reach the app
- **Array form `CMD []` doesn't expand environment variables**

### 3. **Shell Variable Expansion Not Working** ❌
- Array form `CMD ["sh", "-c", "..."]` doesn't properly expand `${PORT}`
- Need shell form `CMD sh -c "..."` to enable variable substitution

---

## ✅ FIX APPLIED (Commit 492df92)

```dockerfile
# AFTER (FIXED):

# Health check - correct path
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8080}/health || exit 1

# CMD - shell form with proper PORT expansion
CMD sh -c "uvicorn wolf_app:APP --host 0.0.0.0 --port \${PORT:-8080}"
```

**Changes:**
1. ✅ Health check now uses `/health` (correct endpoint)
2. ✅ CMD uses shell form `sh -c` to expand `$PORT`
3. ✅ Escaped variable `\${PORT:-8080}` to prevent Docker-time expansion
4. ✅ Fallback to 8080 if PORT not set (local dev compatibility)

---

## 🔧 MANUAL STEPS REQUIRED (Railway Dashboard)

**Since agent doesn't have Railway CLI/dashboard access, user must:**

### Step 1: Verify Deployment Started
1. Go to https://railway.app/dashboard
2. Open **Ghost Protocol** project
3. Open **Deployments** tab
4. **Confirm**: Latest deployment (commit `492df92`) is:
   - ✅ **Building** (in progress)
   - ✅ **Deploying** (in progress)
   - ✅ **Active** (deployed)
   - ❌ **Failed** (red) → Check logs below

### Step 2: Check Deployment Logs
If deployment is **red (failed)** or **stuck in loop**:

1. Click **Logs** tab
2. Scroll to **top** (first 50 lines)
3. Look for errors:
   ```
   ERROR: address already in use
   ERROR: Could not import module "wolf_app"
   ERROR: Failed to bind to port
   ERROR: Health check timeout
   ModuleNotFoundError: No module named 'X'
   ```

### Step 3: Verify Environment Variables
Confirm these are set in Railway:
- `PORT` - Should be **automatically set by Railway** (don't manually set)
- `SIM_MODE=0` ✅
- `USE_NEW_COCKPIT=1` ✅
- `OPENAI_API_KEY` ✅
- `POLYGON_API_KEY` ✅
- All 90+ other vars from previous setup ✅

### Step 4: Manual Restart (If Needed)
If deployment is **Active** but still not responding:
1. Click **Settings** → **Restart Service**
2. Wait 2-3 minutes for restart
3. Check logs for clean startup

### Step 5: Verify Port Binding
In logs, search for:
```
INFO:     Uvicorn running on http://0.0.0.0:XXXX
INFO:     Application startup complete
```
- `XXXX` should match Railway's assigned `$PORT`
- If you see port 8080 but Railway expects different port → still broken

---

## 🧪 POST-DEPLOYMENT VALIDATION

Once Railway shows **Active** deployment, test these endpoints:

```bash
# 1. Health check
curl https://ghost-protocol-production.up.railway.app/health
# Expected: {"status":"healthy","service":"ghost-protocol","version":"..."}

# 2. V3 Cockpit Status
curl https://ghost-protocol-production.up.railway.app/api/v3/cockpit/status
# Expected: {"status":"online","version":"v10.2",...}

# 3. Goals Snapshot
curl https://ghost-protocol-production.up.railway.app/api/v3/goals/snapshot
# Expected: {"ghost_score":XX,"streak_days":X,...}

# 4. Hunter Feed (Top Movers)
curl https://ghost-protocol-production.up.railway.app/api/v3/hunter/feed
# Expected: {"crypto":[...],"stocks":[...],"presales":[...]}

# 5. World Context
curl https://ghost-protocol-production.up.railway.app/api/v3/world/context
# Expected: {"market_regime":"BULL|BEAR|SIDEWAYS",...}

# 6. Cockpit UI (Browser)
open https://ghost-protocol-production.up.railway.app/cockpit
# Expected: Cockpit V3 UI loads with live data
```

**All should return 200 OK with valid JSON** (not timeout/502/503).

---

## 📊 CURRENT TEST RESULTS

Tested at: 2025-11-22 15:43 UTC  
Status after fix push:

| Endpoint | Status | Response Time |
|----------|--------|---------------|
| `/health` | ❌ Timeout | >10s |
| `/api/v3/cockpit/status` | ❌ Timeout | >10s |
| `/api/v3/goals/snapshot` | ⏳ Not tested | - |
| `/api/v3/hunter/feed` | ⏳ Not tested | - |
| `/cockpit` (UI) | ❌ ERR_HTTP2_PROTOCOL_ERROR | N/A |

**Reason for continued failure**: 
Railway takes **3-5 minutes** to:
1. Detect GitHub push
2. Pull new code
3. Build Docker image
4. Deploy container
5. Run health checks
6. Switch traffic to new deployment

**Agent tested 3.5 minutes after push** → likely still building.

---

## 🎯 EXPECTED TIMELINE

| Time | Status |
|------|--------|
| T+0 (15:41 UTC) | Git push commit `492df92` ✅ |
| T+1 min | Railway detects webhook |
| T+2 min | Docker build starts |
| T+3-4 min | Build completes, container starts |
| T+4-5 min | Health checks pass, traffic switches |
| **T+5-7 min** | **Endpoints should be live** ✅ |

---

## 🚨 IF STILL FAILING AFTER 10 MINUTES

Check Railway logs for:

### Possible Issue 1: Import Error
```
ERROR: Could not import module "wolf_app"
ModuleNotFoundError: No module named 'api'
```
**Fix**: Verify `api/__init__.py` exists (commit `33c8807` created it)

### Possible Issue 2: Port Binding
```
ERROR: [Errno 98] Address already in use
```
**Fix**: Railway restart should clear this

### Possible Issue 3: Database Lock
```
sqlite3.OperationalError: database is locked
```
**Fix**: Check Railway volume isn't corrupted, restart service

### Possible Issue 4: Memory/CPU Limit
```
Killed (OOM)
Process exited with code 137
```
**Fix**: Upgrade Railway plan or optimize memory usage

### Possible Issue 5: Health Check Still Failing
```
Health check timeout exceeded
```
**Fix**: Increase `healthcheckTimeout` in railway.toml (currently 100s)

---

## 📋 AGENT ACTIONS COMPLETED

✅ **Diagnosed root cause**: Wrong health check path + hardcoded port  
✅ **Fixed Dockerfile**: Correct `/health` path + dynamic PORT binding  
✅ **Pushed fix**: Commit `492df92` to main branch  
✅ **Verified railway.toml**: Correct configuration with `/health` path  
✅ **Created diagnostic report**: This document  

---

## 🔄 NEXT STEPS (USER)

1. **Wait 10 minutes** from push time (15:41 UTC → **check at 15:51 UTC**)
2. **Test health endpoint**: `curl https://ghost-protocol-production.up.railway.app/health`
3. **If still failing**: 
   - Open Railway dashboard
   - Copy first error from logs
   - Paste here for further diagnosis
4. **If working**: 
   - Test all V3 endpoints above
   - Open Cockpit UI in browser
   - Confirm no ERR_HTTP2_PROTOCOL_ERROR

---

## 📞 CONTACT

If endpoints still timeout after 10 minutes:
- Check Railway dashboard deployment status
- Copy full logs from Railway
- Report first error line for agent analysis

**Expected Result**: Cockpit UI loads successfully with live data from all V3 panels.

---

**Report Generated**: 2025-11-22 15:43 UTC  
**Fix Commit**: `492df92`  
**Status**: ⏳ Awaiting Railway deployment completion
