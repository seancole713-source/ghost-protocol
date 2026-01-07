# 🚨 CRITICAL BUG FIX PLAN
## Ghost Protocol - Production Issues Found in Railway Logs
## Generated: 2026-01-07 20:10:00

---

## 📊 SYSTEM STATUS: PARTIALLY OPERATIONAL

### ✅ WORKING:
- PostgreSQL predictions: **177,763+ predictions stored successfully**
- Prediction generation: **ALL symbols generating (SIMO, SMCI, JPM, BAC, WFC)**
- ml_trainer: **Reading from PostgreSQL correctly**
- Paper trades: **Auto-logging working**
- XGBoost model: **Active and calibrated**

### 🚨 BROKEN:
1. **Paper trade reconciliation**: SQL type error
2. **Outcome reconciliation**: Reading from wrong database
3. **Learning loop**: Can't improve without outcomes

---

## 🐛 BUG #1: Paper Trades Data Type Mismatch

**Error from Railway PostgreSQL logs**:
```
ERROR: operator does not exist: text <= timestamp with time zone at character 171
STATEMENT: 
    SELECT DISTINCT symbol FROM paper_trades 
    WHERE outcome = 'PENDING' 
    AND target_time <= NOW()
```

**Root Cause**:
- `target_time` stored as `TEXT` in PostgreSQL
- Compared to `NOW()` which returns `TIMESTAMP WITH TIME ZONE`
- PostgreSQL (correctly) refuses to compare incompatible types

**Impact**:
- Paper trade reconciliation FAILS
- No paper trades ever get resolved
- Accuracy tracking broken for paper trades

**Fix Required**:
1. Change `target_time` from `TEXT` to `TIMESTAMP WITH TIME ZONE` in `paper_tracker.py`
2. Change `signal_time`, `entry_time`, `checked_at`, `created_at` to `TIMESTAMP` (best practice)
3. Update all datetime insertions to use proper timestamps

**Files to Fix**:
- `core/paper_tracker.py` (lines 106, 137) - table schema
- `wolf_app.py` (line 4065) - query logic (no change needed, will work after schema fix)

---

## 🐛 BUG #2: Outcome Reconciler Reading Wrong Database

**Location**: `services/outcome_reconciler_v2.py` line 392

**Root Cause**:
```python
# WRONG: Queries SQLite predictions table (0 rows)
store.backend.query("SELECT ... FROM predictions ...")
```

**Should be**:
```python
# CORRECT: Query PostgreSQL via store's postgres backend
predictions = store.postgres_backend.query("SELECT ... FROM ghost_predictions ...")
```

**Impact**:
- Reconciler finds 0 predictions to reconcile (SQLite is empty)
- `ghost_prediction_outcomes` table stays at 0 rows FOREVER
- ml_trainer has no training data
- Accuracy stuck at 35% (can't learn)

**Fix Required**:
1. Change reconciler to query PostgreSQL `ghost_predictions` table
2. Use `store.postgres_backend` instead of `store.backend`
3. Update table name from `predictions` to `ghost_predictions`

**Files to Fix**:
- `services/outcome_reconciler_v2.py` (line 392+)

---

## 🛠️ FIX PRIORITY

### **IMMEDIATE (Deploy Now)**:
1. ✅ Bug #1: Paper trades data type fix
2. ✅ Bug #2: Reconciler PostgreSQL query

### **AFTER 48H (Verify)**:
3. ⏳ Check `ghost_prediction_outcomes` table has rows
4. ⏳ Verify ml_trainer gets > 100 outcomes for training
5. ⏳ Confirm accuracy improves from 35% → 65%+

---

## 📋 DEPLOYMENT CHECKLIST

### Pre-Deploy:
- [x] Identify all bugs from Railway logs
- [ ] Fix paper_tracker.py schema
- [ ] Fix outcome_reconciler_v2.py query
- [ ] Test locally in dev container
- [ ] Push to GitHub

### Deploy:
- [ ] Railway auto-deploys from GitHub
- [ ] Monitor Deploy Logs for errors
- [ ] Check PostgreSQL logs for schema changes
- [ ] Verify no more data type errors

### Post-Deploy (Immediate):
- [ ] Check `/api/v3/accuracy/summary` endpoint
- [ ] Check paper trades resolving (POST /alerts/paper-trade/check)
- [ ] Monitor HTTP logs for 200 OK responses

### Post-Deploy (48H Later):
- [ ] Query `ghost_prediction_outcomes` table (expect 100+ rows)
- [ ] Check ml_trainer logs for training data count
- [ ] Verify accuracy improving toward 65-70%

---

## 🔬 EVIDENCE FROM RAILWAY

### ✅ PostgreSQL Predictions Working:
```
[POSTGRES] Created prediction 177761 for SIMO with 25 forecast points
[PostgresBackend] Saved prediction 177761 for SIMO (25 points, 71ms)
[SIMO] Stored in ghost_predictions table (ID=177761, direction=UP, confidence=78.0%)
```

### 🚨 Paper Trades Error:
```
ERROR: operator does not exist: text <= timestamp with time zone at character 171
STATEMENT: SELECT DISTINCT symbol FROM paper_trades WHERE outcome = 'PENDING' AND target_time <= NOW()
```

### ✅ Predictions Flowing:
- 177761: SIMO (UP @ 78.0%)
- 177762: SMCI (DOWN @ 65.8%)
- 177763: JPM (UP @ 60.5%)
- All stored in `ghost_predictions` table successfully

---

## 🎯 SUCCESS METRICS

### Before Fix:
- ❌ Paper trades: SQL errors in PostgreSQL logs
- ❌ Outcomes table: 0 rows
- ❌ Accuracy: Stuck at 35%
- ❌ Learning: No new training data

### After Fix (Immediate):
- ✅ Paper trades: No SQL errors
- ✅ Reconciler: Finds predictions to reconcile
- ✅ Outcomes: Starts populating after 48h

### After Fix (48H):
- ✅ Outcomes table: 100+ rows
- ✅ ml_trainer: Gets sufficient data
- ✅ Accuracy: Improves toward 65-70%
- ✅ Learning: Active and continuous

---

## 🚀 NEXT STEPS

1. **NOW**: Apply fixes to both bugs
2. **DEPLOY**: Push to GitHub → Railway auto-deploys
3. **WAIT 48H**: Let reconciler populate outcomes
4. **VERIFY**: Check outcomes table, retrain model, measure accuracy

User's skepticism was **100% JUSTIFIED** - audit found REAL bugs in production! 🎯
