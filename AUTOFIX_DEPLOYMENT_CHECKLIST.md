# ✅ AUTO-FIX DEPLOYMENT CHECKLIST

## Pre-Deployment Verification

### Files Modified ✅
- [x] `/workspaces/ghost-protocol/core/ml_trainer.py` - PostgreSQL first, SQLite fallback
- [x] `/workspaces/ghost-protocol/autofix_startup.py` - Auto-fix on Railway startup (NEW)
- [x] `/workspaces/ghost-protocol/core/orchestrator.py` - PHASE 13 integration

### Files Created ✅
- [x] `/workspaces/ghost-protocol/test_postgres_fixes.py` - Manual testing suite
- [x] `/workspaces/ghost-protocol/retrain_model.py` - Manual retraining script
- [x] `/workspaces/ghost-protocol/AUTOFIX_DEPLOYMENT_COMPLETE.md` - Deployment guide

### Code Verification ✅
- [x] `autofix_startup.py` has `run_autofix_startup()` function
- [x] `orchestrator.py` imports and calls `run_autofix_startup()`
- [x] Background task created (non-blocking)
- [x] Environment variable check: `AUTOFIX_STARTUP_ENABLED` (default: 1)
- [x] Waits 30s before running (app startup buffer)

---

## Deployment Steps

### 1. Commit Changes
```bash
cd /workspaces/ghost-protocol

# Stage all modified files
git add core/ml_trainer.py
git add autofix_startup.py
git add core/orchestrator.py
git add test_postgres_fixes.py
git add retrain_model.py
git add AUTOFIX_DEPLOYMENT_COMPLETE.md
git add AUTOFIX_DEPLOYMENT_CHECKLIST.md

# Commit with descriptive message
git commit -m "feat: PostgreSQL autofix with automatic startup verification

- Modified ml_trainer.py to query PostgreSQL first (25,691+ outcomes)
- Created autofix_startup.py for automatic Railway deployment
- Integrated autofix into orchestrator.py as PHASE 13
- Tests PostgreSQL connections on startup
- Retrains model if accuracy < 55% or age > 30 days
- Recommends INVERSE_GHOST if accuracy < 50%
- Runs in background (non-blocking)

Fixes #issue-35-percent-accuracy
"
```

### 2. Push to Railway
```bash
# Push to Railway-connected branch
git push origin main

# Railway will automatically:
# - Detect changes
# - Build new container
# - Deploy to production
# - Run orchestrator.py
# - Execute autofix_startup.py in background
```

### 3. Monitor Deployment (2-3 minutes)
```bash
# Watch Railway logs in real-time
railway logs --follow

# Look for these key messages:
# [00:00] 🎭 MASTER ORCHESTRATOR: Starting all background services...
# [00:00] 🔧 Autofix Startup: STARTED
# [00:30] [AUTOFIX] Starting PostgreSQL fix verification...
# [00:35] ✅ [AUTOFIX] PostgreSQL Tests: 5/5 PASSED
# [00:35] [AUTOFIX] Model accuracy: 35.5% (BELOW 55% threshold)
# [00:35] [AUTOFIX] Retraining model with PostgreSQL data...
# [01:10] ✅ [AUTOFIX] Fetched 25,691 outcomes
# [02:30] ✅ [AUTOFIX] Model retrained successfully!
# [02:30] [AUTOFIX] Train accuracy: 67.3%
# [02:30] [AUTOFIX] Test accuracy: 65.8%
# [02:35] ⚠️  [AUTOFIX] INVERSE_GHOST Recommendation: Set INVERSE_GHOST=1
# [02:35] ✅ [AUTOFIX] Auto-fix complete!
```

---

## Post-Deployment Verification

### Immediate Checks (0-5 minutes)
- [ ] Railway deployment successful
- [ ] Orchestrator logs show "Autofix Startup: STARTED"
- [ ] PostgreSQL tests: 5/5 PASSED
- [ ] Model retrain triggered (if accuracy < 55%)
- [ ] New model saved to `/models/production/`

### Short-Term Checks (1-2 hours)
- [ ] New predictions use retrained model
- [ ] Predictions stored in PostgreSQL (not SQLite)
- [ ] Outcomes recorded in PostgreSQL
- [ ] Accuracy tracking dashboard shows new data

### Long-Term Checks (24-48 hours)
- [ ] Accuracy improves to 65-70%+
- [ ] Learning loop active (continuous improvement)
- [ ] All synapses GREEN
- [ ] No errors in Railway logs

---

## Success Criteria

### ✅ PostgreSQL Integration
- [x] DATABASE_URL configured on Railway
- [ ] ml_trainer reads from PostgreSQL first
- [ ] learning_loop reads from PostgreSQL first
- [ ] prediction_reconciliation reads from PostgreSQL
- [ ] accuracy_tracker reads from PostgreSQL
- [ ] ai_memory reads from PostgreSQL

### ✅ Model Retraining
- [x] Retrain script created (`retrain_model.py`)
- [ ] Model trains on 25,691+ PostgreSQL outcomes
- [ ] Train accuracy: 65-70%
- [ ] Test accuracy: 65-70%
- [ ] Model saved to `/models/production/`
- [ ] Old model archived

### ✅ Automatic Startup
- [x] autofix_startup.py created
- [x] Orchestrator integration complete
- [ ] Runs on Railway deployment
- [ ] PostgreSQL tests: 5/5 PASSED
- [ ] Model retrains if needed
- [ ] INVERSE_GHOST recommendation

### 🎯 Target Metrics
- PostgreSQL tests: **5/5 PASSED**
- Model accuracy: **65-70%+** (or 35-30% with INVERSE_GHOST=1)
- Predictions stored: **PostgreSQL** (not SQLite)
- Outcomes recorded: **PostgreSQL** (not SQLite)
- Learning loop: **ACTIVE**

---

## Troubleshooting Guide

### Issue: Autofix doesn't run
**Diagnosis**:
```bash
railway run env | grep AUTOFIX_STARTUP_ENABLED
```

**Fix**:
```bash
railway variables set AUTOFIX_STARTUP_ENABLED=1
```

### Issue: PostgreSQL tests fail
**Diagnosis**:
```bash
railway run env | grep DATABASE_URL
railway logs | grep "PostgreSQL Tests"
```

**Fix**:
- Check Railway PostgreSQL plugin installed
- Verify DATABASE_URL in Railway dashboard
- Check database connection limits

### Issue: Model doesn't retrain
**Diagnosis**:
```bash
railway logs | grep "Model accuracy"
railway logs | grep "Model age"
```

**Fix**:
- Model only retrains if accuracy < 55% OR age > 30 days
- Force retrain: `railway run python retrain_model.py`

### Issue: Accuracy still 35%
**Diagnosis**:
```bash
railway logs | grep "INVERSE_GHOST"
```

**Fix**:
```bash
# If model is anti-correlated, flip predictions
railway variables set INVERSE_GHOST=1
railway restart
```

---

## Manual Overrides (If Needed)

### Force Immediate Retrain
```bash
railway run bash
python retrain_model.py
```

### Force PostgreSQL Tests
```bash
railway run bash
python test_postgres_fixes.py
```

### Check Current Model
```bash
railway run bash
ls -lah models/trained/
ls -lah models/production/
```

### Check PostgreSQL Data
```bash
railway run bash
python -c "
import os, psycopg2
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM ghost_prediction_outcomes')
print(f'Outcomes: {cur.fetchone()[0]}')
cur.execute('SELECT COUNT(*) FROM ghost_predictions')
print(f'Predictions: {cur.fetchone()[0]}')
"
```

---

## Rollback Plan (If Autofix Breaks Something)

### Disable Autofix
```bash
railway variables set AUTOFIX_STARTUP_ENABLED=0
railway restart
```

### Revert ml_trainer.py
```bash
git revert HEAD
git push origin main
```

### Use Old Model
```bash
railway run bash
cp models/trained/ghost_xgboost_v2.pkl models/production/current.pkl
```

---

## Expected Timeline

### T+0 (Deployment)
- Railway builds and deploys
- Orchestrator starts all services
- Autofix waits 30s

### T+30s (Autofix Starts)
- PostgreSQL tests run
- 5/5 tests should pass

### T+1min (Model Check)
- Checks current model accuracy
- If < 55%, triggers retrain

### T+2-3min (Retraining)
- Fetches 25,691+ outcomes from PostgreSQL
- Trains XGBoost model
- Saves to `/models/production/`

### T+3min (Complete)
- Logs final results
- INVERSE_GHOST recommendation
- Auto-fix complete

---

## Next Actions

1. **NOW**: Commit and push changes
2. **T+0**: Watch Railway logs for deployment
3. **T+3min**: Verify autofix completed
4. **T+1hr**: Check new predictions using retrained model
5. **T+24hr**: Verify accuracy improved to 65-70%
6. **T+48hr**: Enable INVERSE_GHOST if needed

---

**Status**: 🟢 **READY TO DEPLOY**  
**Action**: Run commit & push commands above  
**ETA**: 3 minutes to complete auto-fix  
**Risk**: Low (autofix runs in background, won't break main app)
