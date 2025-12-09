# GHOST ACCURACY AUDIT - CRITICAL FINDINGS
## Date: December 8, 2025

---

## ⚠️ EXECUTIVE SUMMARY

**ACCURACY MEASUREMENT FAILED DUE TO DATA LIMITATIONS**

Initial reconciliation of 195 historical predictions (Nov 22 - Dec 3) yielded **9.8% accuracy**, but this result is **INVALID** due to methodological flaws in the measurement approach.

## 🔍 WHAT WENT WRONG

### Problem 1: Incorrect Time Window
- **Predictions made**: Dec 3, 2025 @ 07:39 UTC
- **Prediction horizon**: 48 hours (should resolve Dec 5 @ 07:39 UTC)
- **Measurement time**: Dec 8, 2025 (6 days later, not 48h later)
- **Impact**: We measured prices **5.9 days** after predictions instead of **exactly 48 hours**

**Example:**
```
RUNE prediction Dec 3:
- Predicted: FLAT (±0.25%)
- Price at t0: $0.6717
- Price at t+48h (Dec 5): UNKNOWN
- Price at t+143h (Dec 8): $0.6707 (-0.14%)
```

Ghost predicted what would happen in 48 hours. We measured what happened in 143 hours. These are **completely different** timeframes.

### Problem 2: Historical Price Data Unavailable
- **What we need**: Price at `run_at + 48h` for each prediction
- **What we have**: Current prices (Dec 8, 2025)
- **Data source attempted**: CoinGecko free API (no historical data)
- **Result**: Cannot validate predictions accurately

### Problem 3: Local vs Production Data Mismatch
- **Local SQLite**: 195 predictions (Nov 22 - Dec 3)
- **Production Postgres**: Different predictions (99505-99753+ IDs)
- **Outcomes table**: EXISTS but EMPTY (0 reconciled outcomes)
- **Reconciler**: Runs hourly but not populating outcomes

## 📊 INVALID RESULTS (For Reference Only)

**Reconciliation Attempt Using Current Prices:**
- ✅ Correct: 12
- ❌ Wrong: 110
- ⏸️ No Data: 73
- **Apparent Accuracy: 9.8%** (95% CI: [4.6%, 15.1%])

**Why This Is Meaningless:**
1. Measuring wrong time window (6 days vs 48h)
2. Crypto markets highly volatile - 6-day moves ≠ 48h moves
3. Many predictions show massive moves (-10% to -13%) which may not have occurred at t+48h
4. Using current prices assumes linear movement (false assumption)

## 🎯 ROOT CAUSE ANALYSIS

### Why Outcomes Table Empty
Investigation needed to determine why production reconciler not working:

**Possible Causes:**
1. **No predictions old enough**: Production may only have recent predictions (<48h old)
2. **Reconciler not running**: Orchestrator may not be triggering hourly reconciliation
3. **Silent failures**: Price fetching may be failing and reconciler is skipping
4. **Data isolation**: Predictions may be in different database than expected

### Why Local SQLite Has No Historical t+48h Data
The `prediction_points` table stores forecast prices but NOT actual prices at resolution:
```sql
CREATE TABLE prediction_points (
    prediction_id, ts, kind CHECK(forecast/actual), price
)
```
- Only `kind='forecast'` points exist (prediction time prices)
- No `kind='actual'` points exist (resolution time prices)
- Reconciler was meant to populate this via outcome_reconciler_v2.py

## 🚨 CRITICAL BLOCKER

**WE CANNOT DETERMINE GHOST'S ACCURACY WITHOUT HISTORICAL PRICE DATA**

The 195 predictions exist, but without knowing what prices were **exactly 48 hours later**, we cannot validate them.

## 📋 REQUIRED ACTIONS TO UNBLOCK

### Option A: Fix Production Reconciler (RECOMMENDED)
1. **Investigate why reconciler not populating outcomes**
   - Check production logs for reconciler execution
   - Verify predictions exist with `run_at + 48h < NOW()`
   - Test price fetching in production environment
   
2. **Manually trigger reconciliation**
   - Use outcome_reconciler_v2.py on production data
   - Populate ghost_prediction_outcomes table
   - Query accuracy via pre-built views

3. **Timeline**: 2-4 hours (investigation + fix + validation)

### Option B: Fetch Historical Prices via Premium API
1. **Purchase CoinGecko Pro** ($129/month for historical data)
   - API endpoint: `/coins/{id}/market_chart/range`
   - Can query exact prices at specific timestamps
   - Supports 3+ years of historical data

2. **Reconstruct 48h prices**
   - For each prediction: query price at `run_at + 172800 seconds`
   - Calculate realized_move_pct accurately
   - Validate all 195 predictions

3. **Timeline**: 4-6 hours (API integration + reconciliation)

### Option C: Wait for New Predictions to Age
1. **Current production predictions**: Dec 8 @ various times
2. **Resolution time**: Dec 10 @ various times (48h later)
3. **Reconciliation**: Dec 10 onwards (once past horizon)
4. **First accuracy data**: Dec 11 (allowing reconciler to run)
5. **Timeline**: 3+ days

### Option D: Use Production Database Historical Data
1. **Check if production Postgres has historical predictions**
   - Query `ghost_predictions` for predictions with `run_at < NOW() - 172800`
   - Verify these are different from local SQLite 195 predictions
   
2. **Query historical market data if stored**
   - Check if Ghost stores price history anywhere
   - Look for `price_history` or similar tables
   - Check Redis cache for historical data

3. **Timeline**: 1-2 hours (database exploration)

## 🎯 RECOMMENDATION

**OPTION A (Fix Production Reconciler) is the BEST PATH** because:

1. **Fixes the system permanently** - reconciler should work, currently broken
2. **Uses production infrastructure** - proper APIs, proper data
3. **Enables ongoing accuracy tracking** - not just one-time measurement
4. **Faster than waiting** - 2-4 hours vs 3+ days
5. **Cheaper than API subscription** - no $129/month cost

**Next Steps:**
1. SSH into Railway production environment
2. Check production Postgres for predictions WHERE `run_at < NOW() - INTERVAL '48 hours'`
3. Check Railway logs for reconciler execution and errors
4. Run outcome_reconciler_v2.py manually with verbose logging
5. Identify why it's not populating ghost_prediction_outcomes
6. Fix the issue and verify outcomes appear
7. Query `/api/v3/accuracy/summary` for actual accuracy

## 📊 CONFIDENCE ASSESSMENT

**Current Confidence in 70% Target: 20/100** ⬇️ DOWN from 55/100

**Reasoning:**
- ❌ Cannot validate historical predictions (no data)
- ❌ Production reconciler broken (outcomes table empty)
- ❌ No accuracy measurement infrastructure working
- ❌ Invalid 9.8% result raises concerns (even if methodologically flawed)
- ⚠️ Even with wrong time window, 9.8% is FAR below random (33%)
- ⚠️ Suggests either: (a) predictions getting worse over time, or (b) methodology error

**What Could Restore Confidence:**
- ✅ Fix reconciler + get real accuracy data
- ✅ If real accuracy shows 65-75% → confidence jumps to 85/100
- ✅ If real accuracy shows 60-65% → confidence stays at 50/100
- ✅ If real accuracy shows <55% → confidence drops to 10/100 (failure)

## 🔬 TECHNICAL NOTES

### Why 9.8% Result Is Suspicious
Even with wrong time window, 9.8% accuracy is **statistically impossible** if predictions have any edge:

1. **Random baseline**: 33.3% (UP/DOWN/FLAT equally likely)
2. **Observed**: 9.8% (12 correct out of 122)
3. **Statistical significance**: p < 0.0001 (HIGHLY significant, but in WRONG direction)

This suggests:
- **Either**: Predictions are anti-correlated (inverse of reality) → model inversion bug
- **Or**: Measurement methodology is completely wrong
- **Or**: Crypto market structure changed dramatically (Nov-Dec 2025 market shift)

### Symbol Breakdown from Invalid Results
Top failures (wrong time window, but showing patterns):
```
❌ QNT:    -9.02% move (predicted FLAT)
❌ HBAR:  -10.58% move (predicted FLAT)
❌ ICP:   -11.16% move (predicted FLAT)
❌ FLOW:  -10.17% move (predicted UP, went DOWN)
❌ 1INCH: -13.54% move (predicted FLAT)
```

**Pattern**: Most failures predicted FLAT, but crypto moved significantly down. This could mean:
- Ghost's FLAT threshold (±0.25%) too narrow for crypto volatility
- Predictions made during high volatility period (Nov-Dec 2025)
- Bear market phase not detected by features

### Forecast Accuracy Database
The `data/forecast_accuracy.db` with 4537 forecasts is likely a **different system** or **abandoned feature**:
- Schema has outcome columns but all NULL
- Not referenced in production code
- May be from earlier Ghost version (pre-v3.0)
- Low priority to investigate until primary system working

---

## 📝 AUDIT STATUS

**Phase 1: Infrastructure Discovery** ✅ COMPLETE
- Located prediction storage (Postgres + SQLite)
- Found outcome tracking schema (ghost_prediction_outcomes)
- Identified reconciliation system (outcome_reconciler_v2.py)

**Phase 2: Historical Data Discovery** ✅ COMPLETE  
- Found 195 local predictions (Nov 22 - Dec 3)
- Found 4537 forecasts in forecast_accuracy.db
- Confirmed all past 48h horizon

**Phase 3: Outcome Reconciliation** ❌ FAILED
- Attempted reconciliation using current prices
- Invalid methodology (wrong time window)
- Cannot proceed without historical price data

**Phase 4: Accuracy Measurement** ⏸️ BLOCKED
- Waiting on one of four options (A/B/C/D above)
- Cannot calculate valid accuracy until data available

**Phase 5: Statistical Analysis** ⏸️ PENDING
- Will run once valid outcome data obtained
- Will calculate accuracy with Wilson score 95% CI
- Will break down by symbol, horizon, confidence

**Phase 6: Final Verdict** ⏸️ PENDING
- Will determine if Ghost meets 70% target
- Will provide confidence interval and sample size recommendations
- Will deliver go/no-go recommendation for production use

---

## 🎯 MISSION STATUS

**ACCURACY AUDIT: BLOCKED - AWAITING DATA**

**Confidence in 70% Target: 20/100** (down from 55/100)

**Blocker**: No historical price data at t+48h for prediction validation

**Recommended Action**: Fix production reconciler (Option A)

**ETA for Valid Results**: 2-4 hours (if Option A pursued)

---

*Auditor: GitHub Copilot (Claude Sonnet 4.5)*  
*Baseline: Ghost v3.0-DECEMBER-3-STABLE (commit 54510e4)*  
*Report Date: December 8, 2025*
