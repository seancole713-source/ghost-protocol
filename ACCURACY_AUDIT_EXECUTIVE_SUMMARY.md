# GHOST ACCURACY AUDIT - EXECUTIVE SUMMARY
## Status: BLOCKED - Awaiting Production Data Access
## Date: December 8, 2025 | Auditor: GitHub Copilot (Claude Sonnet 4.5)

---

## 🎯 MISSION OBJECTIVE

**Determine with statistical rigor whether Ghost can achieve 70%+ prediction accuracy**

---

## 📊 CURRENT STATUS: **BLOCKED**

**Confidence in 70% Target: 20/100** ⬇️ DOWN from 55/100

### Critical Blocker
Cannot access production database to determine if predictions exist that are ready for reconciliation. All attempts to validate historical predictions have failed due to data limitations.

---

## 🔍 WHAT WE DISCOVERED

### 1. ✅ Prediction Infrastructure (WORKING)
- **Storage**: PostgreSQL (production) + SQLite (local fallback)
- **Schema**: Complete with `ghost_predictions` table (20+ columns)
- **Reconciliation**: `services/outcome_reconciler_v2.py` (448 lines)
- **Outcome Tracking**: `ghost_prediction_outcomes` table with 6 indexes + 4 accuracy views
- **API Endpoints**: `/api/v3/accuracy/summary`, `/api/v3/predictions/latest`

**Status**: ✅ Architecture is sound

### 2. ✅ Local Historical Data (FOUND)
- **Location**: `data/ghost_predictions.db` (536KB SQLite)
- **Count**: 195 predictions from Nov 22 - Dec 3, 2025
- **Age**: 6-16 days old (all past 48h horizon)
- **Symbols**:
  - WOLF: 24 predictions (Nov 22)
  - AAPL: 6 predictions (Nov 22)
  - BTC: 5 predictions (Dec 2-3)
  - 30+ crypto: 3 predictions each (Dec 2-3)

**Status**: ✅ Historical data exists locally

### 3. ❌ Historical Price Data (MISSING)
- **Required**: Prices at `prediction_time + 48h` for each prediction
- **Available**: Current prices only (Dec 8)
- **Problem**: Cannot validate predictions without knowing what happened 48h after they were made

**Status**: ❌ Cannot reconcile without historical prices

### 4. ❌ Production Reconciler (NOT WORKING)
- **Expected**: Reconciler runs hourly via orchestrator
- **Reality**: `ghost_prediction_outcomes` table is EMPTY (0 rows)
- **Evidence**: `/api/v3/accuracy/summary` returns "No reconciled predictions found"
- **Unknown**: Why reconciler not populating outcomes

**Status**: ❌ Production system not generating accuracy data

### 5. ❌ Production Database Access (BLOCKED)
- **Problem**: `DATABASE_URL=postgresql://...@postgres.railway.internal` uses private DNS
- **Impact**: Cannot query production database from local machine
- **Attempted Solutions**:
  - ✗ `railway run python3 script.py` - still can't resolve hostname
  - ✗ Admin API endpoint - requires authentication
  - ✗ Direct connection - requires Railway internal network

**Status**: ❌ Cannot inspect production data

---

## 🧪 RECONCILIATION ATTEMPT (INVALID)

### Methodology
Created script to reconcile 195 local predictions using CoinGecko API for current crypto prices.

### Results
```
✅ Correct:   12
❌ Wrong:    110
⚠️  No Data:   73
────────────────────────────────────────
📊 ACCURACY: 9.8% (12/122)
📈 95% CI:   [4.6%, 15.1%]
```

### ⚠️ WHY THIS IS INVALID
1. **Wrong Time Window**: Measured 5-6 days after predictions instead of exactly 48 hours
2. **Current Prices**: Used Dec 8 prices, not historical prices from Dec 5 (t+48h)
3. **Market Volatility**: Crypto moves dramatically in 5-6 days vs 48h
4. **Methodology Flaw**: Predictions targeted 48h, we measured 143h

**Example:**
- RUNE prediction (Dec 3): FLAT (±0.25%)
- Price at t0: $0.6717
- Price at t+48h (Dec 5): **UNKNOWN**
- Price at t+143h (Dec 8): $0.6707 (-0.14%)

Ghost predicted 48h movement. We measured 6-day movement. **Completely different timeframes.**

### Why 9.8% Is Concerning
Even with methodological flaws, 9.8% accuracy is **far below random chance (33%)**:
- **Random baseline**: 33.3% (UP/DOWN/FLAT equally likely)
- **Observed**: 9.8% (statistically significant in wrong direction, p < 0.0001)

**Possible explanations:**
1. Measurement methodology completely wrong (most likely)
2. Predictions are inversely correlated (model bug)
3. Crypto market shift Nov-Dec 2025 invalidated model
4. Local predictions are test data, not real predictions

---

## 🚨 THREE CRITICAL UNKNOWNS

### Unknown #1: Production Predictions
**Question**: How many predictions exist in production Postgres that are >48h old?

**Why Critical**: If zero, we must wait 2-3 days for new predictions to age. If hundreds, we can reconcile immediately.

**How to Check**: 
```sql
SELECT COUNT(*) FROM ghost_predictions 
WHERE run_at < NOW() - INTERVAL '48 hours'
```

### Unknown #2: Why Reconciler Not Working
**Question**: Why is `ghost_prediction_outcomes` table empty despite reconciler running hourly?

**Why Critical**: Production system should be self-reconciling. If broken, accuracy tracking impossible.

**Possible Causes**:
- No predictions old enough (age < 48h)
- Reconciler not actually running
- Price fetching failures (API rate limits, wrong providers)
- Silent errors being caught by crash protection

### Unknown #3: Historical Price Availability
**Question**: Can we access historical prices at `t+48h` for 195 local predictions?

**Why Critical**: Without historical prices, cannot validate local predictions.

**Options**:
- CoinGecko Pro API ($129/month)
- Historical data in production database
- Wait for new predictions and measure prospectively

---

## 📋 ACTION PLAN

### OPTION A: Fix Production Reconciler (RECOMMENDED) ⭐
**Objective**: Get production system working to generate ongoing accuracy data

**Steps**:
1. Access Railway shell: `railway shell ghost-protocol`
2. Run Python script to check predictions:
   ```python
   from sqlalchemy import create_engine, text
   import os, time
   
   engine = create_engine(os.getenv('DATABASE_URL'))
   with engine.connect() as conn:
       now = time.time()
       cutoff = now - (48 * 3600)
       
       total = conn.execute(text("SELECT COUNT(*) FROM ghost_predictions")).scalar()
       ready = conn.execute(text("SELECT COUNT(*) FROM ghost_predictions WHERE run_at < :c"), {"c": cutoff}).scalar()
       outcomes = conn.execute(text("SELECT COUNT(*) FROM ghost_prediction_outcomes")).scalar()
       
       print(f"Total predictions: {total}")
       print(f"Ready for reconciliation: {ready}")
       print(f"Reconciled outcomes: {outcomes}")
   ```
3. If predictions exist but outcomes empty: **reconciler is broken**
4. Check Railway logs for reconciler errors: `railway logs --service ghost-protocol | grep reconcil`
5. Manually run reconciler with logging:
   ```bash
   python3 -c "from services.outcome_reconciler_v2 import reconcile_outcomes_v2; reconcile_outcomes_v2()"
   ```
6. Fix any errors discovered
7. Verify outcomes appear in database
8. Query accuracy via `/api/v3/accuracy/summary`

**Timeline**: 2-4 hours  
**Confidence Gain**: +30-50 points (if accuracy is good)  
**Risk**: LOW (just debugging existing system)

### OPTION B: Wait for New Predictions
**Objective**: Let current predictions age and reconcile prospectively

**Steps**:
1. Current predictions: Dec 8 @ various times
2. Resolution time: Dec 10 @ various times (t+48h)
3. Reconciliation: Dec 10-11 (once past horizon)
4. Check `/api/v3/accuracy/summary` on Dec 11

**Timeline**: 3+ days  
**Confidence Gain**: +30-50 points (if accuracy is good)  
**Risk**: LOW (passive waiting)  
**Downside**: Slow, no immediate data

### OPTION C: Purchase Historical Price Data
**Objective**: Validate 195 local predictions retroactively

**Steps**:
1. Sign up for CoinGecko Pro ($129/month)
2. Use `/coins/{id}/market_chart/range` API
3. For each prediction: fetch price at `run_at + 48h`
4. Calculate realized movement and accuracy
5. Generate statistical analysis

**Timeline**: 4-6 hours (API integration)  
**Cost**: $129/month  
**Confidence Gain**: +30-50 points (if accuracy is good)  
**Risk**: MEDIUM (may reveal accuracy is actually poor)

### OPTION D: Manual Sample Validation
**Objective**: Hand-check 10-20 predictions to verify methodology

**Steps**:
1. Select 10 predictions from different dates
2. Manually look up historical prices from Dec 3-5
3. Calculate if predictions were correct
4. Verify calculation methodology matches Ghost's
5. Extrapolate to full dataset if methodology confirms

**Timeline**: 2-3 hours  
**Confidence Gain**: +10-20 points (directional only, small sample)  
**Risk**: LOW (just validation)

---

## 🎯 RECOMMENDED PATH

**Execute OPTION A (Fix Production Reconciler) + OPTION D (Manual Validation)**

**Rationale**:
1. Option A fixes the system permanently (future-proof)
2. Option D provides immediate confidence signal
3. Combined timeline: 4-6 hours
4. No additional cost
5. If Option A reveals production has no old predictions, fall back to Option B (wait)

**Next Immediate Action**:
```bash
# 1. Access Railway production shell
railway shell ghost-protocol

# 2. Run diagnostic query
python3 -c "
from sqlalchemy import create_engine, text
import os, time
from datetime import datetime

engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    now = time.time()
    cutoff_48h = now - (48 * 3600)
    
    total = conn.execute(text('SELECT COUNT(*) FROM ghost_predictions')).scalar()
    ready = conn.execute(text('SELECT COUNT(*) FROM ghost_predictions WHERE run_at < :c'), {'c': cutoff_48h}).scalar()
    outcomes = conn.execute(text('SELECT COUNT(*) FROM ghost_prediction_outcomes')).scalar()
    
    oldest = conn.execute(text('SELECT MIN(run_at) FROM ghost_predictions')).scalar()
    newest = conn.execute(text('SELECT MAX(run_at) FROM ghost_predictions')).scalar()
    
    print(f'Total predictions: {total}')
    print(f'Ready for reconciliation (>48h old): {ready}')
    print(f'Reconciled outcomes: {outcomes}')
    print(f'Oldest prediction: {datetime.fromtimestamp(oldest) if oldest else None}')
    print(f'Newest prediction: {datetime.fromtimestamp(newest) if newest else None}')
    
    if ready > 0 and outcomes == 0:
        print()
        print('⚠️  WARNING: Reconciler NOT working!')
        print(f'   {ready} predictions ready but 0 outcomes')
"
```

---

## 💡 MANUAL VALIDATION SAMPLE

To validate methodology, manually check these predictions:

### Sample 1: WOLF (Nov 22)
- **Prediction Time**: Nov 22, 2025 @ 23:34 UTC
- **Resolution Time**: Nov 24, 2025 @ 23:34 UTC (t+48h)
- **Predicted**: [Need to query local DB]
- **Actual**: [Look up WOLF price on Nov 24]
- **Result**: ✅ Correct / ❌ Wrong

### Sample 2: BTC (Dec 3)
- **Prediction Time**: Dec 3, 2025 @ 07:40 UTC
- **Resolution Time**: Dec 5, 2025 @ 07:40 UTC (t+48h)
- **Predicted**: [Need to query local DB]
- **Actual**: [Look up BTC price on Dec 5]
- **Result**: ✅ Correct / ❌ Wrong

Query to get prediction details:
```sql
SELECT id, symbol, datetime(run_at, 'unixepoch') as time, 
       direction, confidence, horizon_h
FROM predictions 
WHERE symbol IN ('WOLF', 'BTC')
ORDER BY run_at DESC 
LIMIT 5;
```

---

## 📈 CONFIDENCE TRAJECTORY

### Current: 20/100
**Why**: Cannot measure accuracy, invalid reconciliation attempt, production reconciler broken

### After Production Diagnostics: 30-40/100
**If**: Production has predictions ready for reconciliation

### After Reconciler Fix: 45-55/100
**If**: Reconciler successfully populates outcomes

### After First Accuracy Data: 50-90/100
**Depends on actual accuracy:**
- 70-75% accuracy → confidence jumps to 85/100
- 65-70% accuracy → confidence to 75/100
- 60-65% accuracy → confidence to 55/100
- <60% accuracy → confidence drops to 20/100

### After 7 Days of Data: 60-95/100
**With**: 50+ reconciled predictions, tight confidence intervals, pattern analysis

---

## 🎬 NEXT STEPS (Immediate)

**YOU (User) must do:**
1. Run `railway shell ghost-protocol` (requires manual interaction)
2. Paste the diagnostic Python script above
3. Share the output with me
4. Based on results, I'll provide specific fix instructions

**I (Agent) will:**
1. Create ACCURACY_AUDIT_EXECUTIVE_SUMMARY.md (this document) ✅
2. Document all findings in ACCURACY_AUDIT_CRITICAL_FINDINGS.md ✅
3. Prepare reconciler fix scripts (ready to deploy once diagnostics complete)
4. Stand by for production diagnostics results

---

## 📝 FILES CREATED

1. **ACCURACY_AUDIT_CRITICAL_FINDINGS.md** - Technical deep dive
2. **ACCURACY_AUDIT_EXECUTIVE_SUMMARY.md** - This document
3. **reconcile_with_coingecko.py** - CoinGecko reconciliation script (invalid results)
4. **check_production_predictions.py** - Production diagnostic script (can't run locally)
5. **railway_check_predictions.sh** - Railway shell helper (requires manual use)

---

## ⚖️ AUDIT VERDICT

**ACCURACY MEASUREMENT: INCOMPLETE**

**Current Evidence**:
- ❌ Cannot validate historical predictions (no t+48h price data)
- ❌ Cannot access production data (Railway hostname issue)
- ❌ Production reconciler not working (empty outcomes table)
- ❌ Invalid reconciliation attempt shows 9.8% (wrong methodology)

**Verdict**:
**INSUFFICIENT DATA TO DETERMINE IF GHOST MEETS 70% TARGET**

**Required to Complete Audit**:
1. Access production database
2. Fix production reconciler OR obtain historical prices
3. Generate 30+ validated outcomes
4. Calculate accuracy with 95% confidence intervals

**Estimated Time to Valid Results**: 2-7 days
- **Fast path (Option A)**: 2-4 hours if production has old predictions
- **Slow path (Option B)**: 3+ days waiting for new predictions to age

---

**Status**: ⏸️ AWAITING PRODUCTION DIAGNOSTICS  
**Blocker**: Railway database access requires manual shell interaction  
**Next Action**: User runs diagnostic script in Railway shell  

---

*Baseline: Ghost v3.0-DECEMBER-3-STABLE (commit 54510e4)*  
*Audit Date: December 8, 2025*  
*Auditor: GitHub Copilot (Claude Sonnet 4.5)*
