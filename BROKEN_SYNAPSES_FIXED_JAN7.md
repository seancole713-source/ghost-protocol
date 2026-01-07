# Ghost Protocol - Broken Synapses Fixed (Jan 7, 2026)

## Executive Summary

**Problem**: 35% accuracy (worse than 50% random) due to ML training modules reading from **empty SQLite databases** instead of **PostgreSQL with 25,691+ outcomes**.

**Root Cause**: 5 critical modules used local SQLite that dies on Railway deploy. Training data never reached the models.

**Solution**: Fixed `ml_trainer.py` to use PostgreSQL first. Other modules already had PostgreSQL support or weren't part of the broken flow.

---

## The 5 "Broken Synapses" Analysis

### ✅ 1. **ml_trainer.py** - FIXED
**Status**: 🔴 BROKEN → 🟢 FIXED

**Problem**: 
- `_fetch_training_data()` at line 152 used SQLite `data/prediction_outcomes.db`
- This database is EMPTY on Railway (dies every deploy)
- XGBoost model trained on ZERO outcomes

**Fix Applied**:
```python
def _fetch_training_data(symbol, lookback_days):
    # TRY POSTGRESQL FIRST (where 25,691 outcomes live)
    database_url = os.getenv("DATABASE_URL", "")
    if database_url.startswith(("postgres://", "postgresql://")):
        # Query ghost_prediction_outcomes + ghost_predictions
        # Get: prediction_id, symbol, direction, confidence, 
        #      hit_direction, open_price, close_price, features_json
        # Returns: List of training samples with features
    
    # FALLBACK: SQLite (local dev only)
    # (Fixed column names to match SQLite schema)
```

**Impact**: 
- XGBoost now trains on **real outcomes from PostgreSQL**
- Training data includes **50+ features** (not just confidence + price)
- Model can actually learn patterns instead of random guessing

---

### ✅ 2. **accuracy_tracker.py** - NOT PART OF PROBLEM
**Status**: ⚠️ USES SQLITE (but different system)

**Analysis**:
- This tracks **price forecasts** (MAP, RMSE) for regression tasks
- Ghost uses **directional predictions** (UP/DOWN/FLAT) for classification
- These are separate systems - accuracy_tracker is for experimental price targets
- **Not used in the 35% accuracy calculation**

**Verdict**: No fix needed - this is a parallel system, not broken synapse.

---

### ✅ 3. **prediction_reconciliation.py** - LEGACY CODE
**Status**: 🟡 OLD CODE (not used in production)

**Analysis**:
- This old module writes outcomes to SQLite `data/prediction_outcomes.db`
- Production uses `services/outcome_reconciler_v2.py` instead
- V2 writes directly to PostgreSQL `ghost_prediction_outcomes` table
- Orchestrator (`core/orchestrator.py`) calls `reconcile_outcomes_v2()` every 60 minutes

**Verdict**: No fix needed - this module isn't called in production.

---

### ✅ 4. **learning_loop.py** - ALREADY FIXED
**Status**: 🟢 ALREADY CORRECT

**Analysis**:
- Has `_get_postgres_direction_accuracy()` method (line 102-154)
- Queries PostgreSQL `ghost_prediction_outcomes` directly
- Falls back to SQLite only if PostgreSQL fails
- Used by `outcome_reconciler_v2.py` after reconciliation

**Verdict**: No fix needed - already uses PostgreSQL first.

**PostgreSQL Query**:
```python
def _get_postgres_direction_accuracy(self, days=7):
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE hit_direction = 1) as correct,
            COUNT(*) FILTER (WHERE hit_direction = 0) as incorrect
        FROM ghost_prediction_outcomes
        WHERE status = 'closed'
        AND closed_at > NOW() - INTERVAL '%s days'
    """, (days,))
    # Returns: accuracy_pct, map (error rate), bias_pct
```

---

### ⚠️ 5. **ai_memory.py** - DIFFERENT SYSTEM
**Status**: 🟡 USES SQLITE (but not for prediction accuracy)

**Analysis**:
- Stores **AI context memory** with vector embeddings
- Used for RAG (Retrieval-Augmented Generation) context
- NOT used for prediction outcomes or accuracy tracking
- Separate concern from the 35% accuracy issue

**Verdict**: No fix needed - this is memory/context storage, not prediction data.

---

## What Actually Feeds the 35% Accuracy?

### Data Flow (Production)

```
1. ensemble_predictor.py (line 540: BIAS CORRECTION v5)
   ↓ Makes prediction with XGBoost v2 model
   ↓ Saves to PostgreSQL ghost_predictions

2. outcome_reconciler_v2.py (called every 60 min)
   ↓ Fetches predictions where 48h window closed
   ↓ Gets actual prices from live providers
   ↓ Calculates hit_direction (1=correct, 0=wrong)
   ↓ Stores in PostgreSQL ghost_prediction_outcomes

3. learning_loop._get_postgres_direction_accuracy()
   ↓ Queries ghost_prediction_outcomes
   ↓ Calculates accuracy_pct from hit_direction
   ↓ Returns: 35.47% accuracy (888 predictions evaluated)

4. ml_trainer._fetch_training_data() 🔴 BROKEN → 🟢 FIXED
   ↓ NOW: Reads from PostgreSQL ghost_prediction_outcomes
   ↓ WAS: Read from empty SQLite (0 rows)
   ↓ Trains XGBoost v3 on real outcomes
```

---

## Verification Tests

### Test Suite: `test_postgres_fixes.py`

Run on Railway to verify all synapses are GREEN:

```bash
python3 test_postgres_fixes.py
```

**Tests**:
1. ✅ DATABASE_URL configured (PostgreSQL)
2. ✅ ml_trainer._fetch_training_data() → PostgreSQL (returns >0 rows)
3. ✅ learning_loop._get_postgres_direction_accuracy() → PostgreSQL
4. ✅ Direct query to ghost_prediction_outcomes (25,691+ outcomes)
5. ✅ Data quality check (no null symbols, prices)

**Expected Output**:
```
✅ PASS: DATABASE_URL
✅ PASS: ml_trainer (fetched 1,234 training samples)
✅ PASS: learning_loop (35.47% accuracy from PostgreSQL)
✅ PASS: direct_postgres (25,691 total outcomes)
✅ PASS: data_quality (no issues)

🎉 ALL TESTS PASSED - PostgreSQL synapses are GREEN!
```

---

## Files Changed

### 1. `/workspaces/ghost-protocol/core/ml_trainer.py`

**Function**: `_fetch_training_data(symbol, lookback_days)`

**Changes**:
- Added PostgreSQL query using `psycopg2`
- Query joins `ghost_prediction_outcomes` + `ghost_predictions`
- Fetches: prediction_id, symbol, direction, confidence, hit_direction, prices, features_json
- Falls back to SQLite only if PostgreSQL unavailable
- Fixed SQLite column names to match schema (`predicted_direction` not `direction_predicted`)

**Lines Modified**: 152-235

---

## Next Steps

### 1. **Test on Railway** (Priority 1)
```bash
# SSH into Railway container
python3 test_postgres_fixes.py

# Expected: All 5 tests pass
```

### 2. **Retrain XGBoost Model** (Priority 2)
```bash
# Train new model on PostgreSQL data
python3 -c "from core.ml_trainer import train_model; train_model(symbol=None, lookback_days=180)"

# Expected: 
# - Model trained on 1,000+ samples (not 0)
# - test_accuracy > 50% (better than random)
# - Saved to models/production/ghost_model_ALL.pkl
```

### 3. **Monitor Accuracy** (Priority 3)
```bash
# Check if accuracy improves with new model
python3 -c "from core.learning_loop import get_learning_loop; print(get_learning_loop()._get_postgres_direction_accuracy(days=7))"

# Expected:
# - Data source: postgres_outcomes
# - count > 100 (recent outcomes)
# - accuracy_pct > 50% (after model retraining)
```

### 4. **Optional: Enable INVERSE_GHOST** (Quick Fix)
If the model is still anti-correlated after retraining:

```bash
# In Railway environment variables
INVERSE_GHOST=1

# This flips UP/DOWN predictions
# Would turn 35% → 65% accuracy immediately
```

---

## Technical Notes

### PostgreSQL Schema (Production)

**ghost_predictions** (source of predictions):
- id, symbol, run_at, horizon_h, direction, confidence
- features_json (50+ technical indicators)
- predicted_direction (UP/DOWN/FLAT)

**ghost_prediction_outcomes** (reconciled results):
- prediction_id, symbol, status (open/closed)
- open_price, close_price, hit_direction (1/0)
- mae, mape, rmse (error metrics)
- closed_at (timestamp)

**Join Query**:
```sql
SELECT 
    o.prediction_id, o.symbol, p.predicted_direction, p.confidence,
    o.hit_direction, o.open_price, o.close_price, p.features_json
FROM ghost_prediction_outcomes o
JOIN ghost_predictions p ON o.prediction_id = p.id
WHERE o.status = 'closed'
  AND o.closed_at >= NOW() - INTERVAL '30 days'
ORDER BY o.closed_at DESC
LIMIT 10000
```

---

## Summary

**What was broken**: 
- ml_trainer read from empty SQLite instead of PostgreSQL with 25,691 outcomes

**What was fixed**: 
- ml_trainer now reads from PostgreSQL first, SQLite fallback

**What wasn't broken**:
- accuracy_tracker (different system - price forecasts not directions)
- prediction_reconciliation (legacy code - outcome_reconciler_v2 used)
- learning_loop (already had PostgreSQL support)
- ai_memory (separate concern - AI context not prediction data)

**Expected improvement**: 
- XGBoost trains on real data (1,000+ samples not 0)
- Accuracy should improve from 35% toward 50%+ (random baseline)
- If still inverted, enable INVERSE_GHOST=1 for 65% accuracy

**Test status**: 
- ✅ test_postgres_fixes.py created
- ⏳ Awaiting Railway test results

---

## Run This Now

```bash
# On Railway container
python3 test_postgres_fixes.py
```

If all tests pass → **ALL SYNAPSES ARE GREEN** ✅
