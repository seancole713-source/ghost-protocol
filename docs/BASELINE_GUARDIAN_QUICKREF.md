# 🛡️ BASELINE GUARDIAN - QUICK REFERENCE

**Production Baseline**: December 11, 2025 | Commit: `7740c6f6` | Service: tender-benevolence (Railway)

---

## ✅ Current Baseline Status

**ALL SYSTEMS OPERATIONAL** as of 2025-12-11:

- ✅ Core APIs: 10/10 returning HTTP 200
- ✅ Response times: <1s typical, <8s maximum
- ✅ Background workers: VIP scanner, pre-market predictor, auto-prediction loop all active
- ✅ No 499 timeouts or 30s+ delays (previously observed, now resolved)

---

## 🚀 Quick Commands

### Run Regression Test (Before ANY Deployment)
```bash
bash scripts/ghost_regression.sh
```

**Expected Output**: `✅ ALL TESTS PASSED - Baseline is healthy`

**On Failure**: DO NOT DEPLOY. Investigate and fix before proceeding.

### Check Railway Production Logs
```bash
railway logs --tail 100 | grep -E "VIP scan|Pre-market predictor|ERROR|FAIL"
```

**Look for**:
- ✅ `VIP scan #XX: ...` (every ~60s)
- ✅ `🌅 Pre-market predictor starting ...` (scheduled times)
- ✅ `Stored prediction for WOLF|NVDA|BTC...` (auto-prediction loop)
- ❌ Any `ERROR` or `Traceback` lines

### Force Railway Redeploy
```bash
railway up --detach
```

### Rollback to Baseline
```bash
# In Railway dashboard: Deployments → 7740c6f6 → Rollback
# Or via CLI:
railway up --detach  # (after checking out 7740c6f6 in git)
```

---

## 🔒 Protected Components (HIGH RISK)

**DO NOT MODIFY** without explicit approval + regression testing:

| Component | File | Risk | Regression Check |
|-----------|------|------|------------------|
| Auto-Prediction Loop | `core/auto_prediction_loop.py` | 🔴 | Predictions stop generating |
| Stage 1 (RSS/Mood) | `core/stage1_integration.py` | 🔴 | Prediction quality degrades |
| Hunter Feed | `api/cockpit_v3_live_endpoints.py` | 🔴 | 499 timeouts, 30s+ delays |
| Personal Watchlist | `api/personal_watchlist_endpoints.py` | 🔴 | Cockpit UI breaks |
| XRP Tracker | `core/xrp_tracker.py` | 🔴 | VIP alerts stop |
| Presale Watcher | `core/presale_watcher.py` | 🔴 | Presale tracking fails |
| VIP Scanner | `core/orchestrator.py` → VIP loop | 🔴 | Telegram alerts stop |
| Pre-Market Predictor | `core/orchestrator.py` → scheduler | 🔴 | Morning predictions skip |

---

## 📊 Baseline Endpoints (Must Return 200)

Test these manually if regression script fails:

```bash
BASE="https://ghost-protocol-production.up.railway.app"

# Core health
curl "$BASE/health"
curl "$BASE/api/v3/health/metrics"

# Cockpit APIs
curl "$BASE/api/v3/cockpit/status"
curl "$BASE/api/v3/watchlist/user"
curl "$BASE/api/v3/predictions/latest?symbol=BTC&limit=3"

# Trading systems
curl "$BASE/api/v3/goals/snapshot"
curl "$BASE/api/v3/accuracy/summary"

# Live feeds
curl "$BASE/api/v3/hunter/feed"

# Trackers
curl "$BASE/api/xrp/tracker"
curl "$BASE/api/presale/watch"
```

**All must return JSON with `"ok": true` or valid data structure.**

---

## 🚨 Emergency Procedures

### Scenario 1: Regression Test Fails After Deployment

1. **Check Railway logs** for errors:
   ```bash
   railway logs --tail 200 | grep -E "ERROR|Traceback|FAIL"
   ```

2. **Rollback immediately** to 7740c6f6:
   - Railway Dashboard → Deployments → Find 7740c6f6 → Rollback
   - Or redeploy from git after checkout

3. **Verify baseline restored**:
   ```bash
   bash scripts/ghost_regression.sh
   ```

4. **If still failing**: Check Railway service health, database connections, environment variables

### Scenario 2: Background Workers Not Running

**Symptoms**: No "VIP scan" or "Pre-market predictor" logs

**Diagnosis**:
```bash
railway logs --tail 500 | grep -i "orchestrator\|background\|startup"
```

**Check for**:
- Orchestrator initialization errors
- Database connection failures
- Missing environment variables

**Fix**: Restart Railway service or check database connectivity

### Scenario 3: 499 Timeouts Return on Hunter Feed

**Symptoms**: `/api/v3/hunter/feed` returns 499 or >30s response times

**Diagnosis**:
```bash
railway logs --tail 100 | grep "hunter/feed"
```

**Likely Causes**:
- RSS feed source is down or slow
- Database query timeout
- Concurrent request overload

**Fix**:
1. Check Stage 1 RSS feeds are reachable
2. Review database query performance
3. Consider rate limiting or caching

---

## 📝 Change Request Template

Before modifying protected components, fill out:

**Component**: [e.g., Auto-Prediction Loop]  
**File(s)**: [e.g., core/auto_prediction_loop.py]  
**Risk Level**: [🟢 LOW / 🟡 MEDIUM / 🔴 HIGH / 🔴 CRITICAL]  
**Change Description**: [Brief summary]  
**Regression Tests**: [Which endpoints could break?]  
**Rollback Plan**: [How to revert if failure?]  
**Approval**: [Operator signature/timestamp]

---

## 🔍 Baseline Health Checklist (Daily)

Run this daily to confirm production health:

- [ ] `bash scripts/ghost_regression.sh` → All pass
- [ ] Railway logs show "VIP scan #XX" in last 5 minutes
- [ ] Railway logs show predictions for BTC, WOLF, NVDA in last hour
- [ ] `/api/v3/hunter/feed` returns in <5 seconds
- [ ] No 499 timeouts in last 24 hours
- [ ] No ERROR/Traceback lines in last 100 log entries

**If ANY fail**: Investigate immediately before making further changes.

---

**Document Version**: 1.0  
**Last Updated**: December 11, 2025  
**Baseline Commit**: 7740c6f6  
**Guardian Status**: 🛡️ ACTIVE
