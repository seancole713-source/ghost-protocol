# GHOST REBUILD CHECKLIST
## "Fixed = Tested and Verified Working"

---

## ROOT CAUSE SUMMARY

**THE SEED**: Bootstrap commit `23b10bb` contained placeholder prediction code with comment:
> "Replace with actual Ghost forecast logic if available"

This was never replaced. Everything built on top of scaffolding.

**Current State**:
- ~46% accuracy (worse than coin flip)
- 4+ fragmented databases
- Fake ML models (LSTM = momentum calc, Transformer = 3 hardcoded weights)
- XGBoost trained on DAILY data for 48h predictions
- Random confidence values: `random.uniform(0.65, 0.85)`
- INVERSE_GHOST hack exists because predictions were so wrong

---

# PHASE 1: DATABASE CONSOLIDATION (Difficulty: 4/10)

## Task 1.1: Audit Current Database State ✅ COMPLETE

**What**: Document what's in each database

### FINDINGS (Jan 5, 2026):

**27 SQLite databases found** - far worse than expected:

| Database | Size | Tables | Key Row Counts |
|----------|------|--------|----------------|
| data/wolf.db | 2.5M | 23 | price_history: 19,998, forecast_48h: 614, ghost_predictions: 60 |
| data/ghost_predictions.db | 660K | 11 | predictions: 195, prediction_points: 5,226, outcomes: 38 |
| data/forecast_accuracy.db | 588K | 1 | forecasts: 4,601 |
| watchlist.db | 44K | 2 | watchlist: 82 |
| data/active_tracking.db | 36K | 2 | active_picks: 5 |
| data/ai_memory.db | 32K | 2 | ai_memory: 0, calibration_metrics: 0 |
| + 21 more databases | various | various | mostly empty |

**392 sqlite3.connect calls** in production Python code

**Top offenders**:
- wolf_app.py: 66 connections
- core/smart_watcher.py: 15 connections
- core/prediction_store.py: 13 connections
- core/cascading_predictor.py: 11 connections

**Verification Script Created**: `./database_audit.sh`

**Status**: [x] COMPLETE - Audit done, situation worse than expected

---

## Task 1.2: Enable PostgreSQL as Primary Store ✅ COMPLETE

**What**: Set PREDICTION_STORE_ENGINE=postgres to use PostgreSQL

**THE ROOT CAUSE OF "insufficient aligned points"**:
- `PREDICTION_STORE_ENGINE` was NOT SET
- Default was `sqlite` → predictions went to ephemeral SQLite
- SQLite wiped on every Railway deploy
- Reconciler found empty database → "0 aligned points"

**Fix Applied**: Added to `Dockerfile`:
```dockerfile
PREDICTION_STORE_ENGINE=postgres
```

**Verification Test**:
```bash
# In Railway environment, verify backend selection:
PREDICTION_STORE_ENGINE=postgres python3 -c "
from core.prediction_store import PREDICTION_STORE_ENGINE, IS_POSTGRES_AVAILABLE
print(f'Engine: {PREDICTION_STORE_ENGINE}')  # Should be: postgres
print(f'PG Available: {IS_POSTGRES_AVAILABLE}')  # Should be: True
"
```

**PASS CRITERIA**: 
- PREDICTION_STORE_ENGINE=postgres
- IS_POSTGRES_AVAILABLE=True
- Backend is PostgresBackend

**Status**: [x] COMPLETE - Fixed in Dockerfile

---

## Task 1.3: Migrate Existing SQLite Predictions to PostgreSQL ✅ SCRIPT CREATED

**What**: One-time migration of predictions from SQLite to PostgreSQL

**Why**: The 259 predictions + 38 outcomes in SQLite need to move to PostgreSQL.

**Script Created**: `scripts/migrate_sqlite_to_postgres.py`
- Handles both old and new outcome schemas
- Dry-run mode for testing
- Verifies migration with counts

**Verification Test**:
```bash
# 1. Run migration (in Railway)
railway run python scripts/migrate_sqlite_to_postgres.py

# 2. Verify
railway run python scripts/migrate_sqlite_to_postgres.py --dry-run
# Should show: "Found 0 predictions in SQLite" (already migrated)
```

**PASS CRITERIA**: 
- [x] Script created and tested (dry-run)
- [ ] Run in Railway (deploy required)
- [ ] Verify counts match

**Status**: [x] SCRIPT READY - Awaiting deployment

---

## Task 1.4: Verify PostgreSQL Predictions End-to-End

**What**: Full round-trip test - create prediction, store, reconcile

**Verification Test**:
```bash
# In Railway environment after deployment:
railway logs | grep -E "PostgreSQL|predictions"
# Expected: "Using PostgreSQL backend for predictions"

# Create a test prediction and verify it's in PostgreSQL
curl -X POST $RAILWAY_URL/api/debug/test-prediction
# Check logs for "Created prediction X for BTC"
```

**PASS CRITERIA**: 
- [ ] Predictions stored in PostgreSQL
- [ ] Reconciler finds pending outcomes
- [ ] No "insufficient aligned points" errors

**Status**: [ ] NOT STARTED
conn = get_db_connection()
result = conn.execute('SELECT COUNT(*) FROM predictions WHERE created_at > NOW() - INTERVAL 1 minute').fetchone()
print(f'Recent predictions in PostgreSQL: {result[0]}')
assert result[0] >= 1, 'FAIL: Prediction not in PostgreSQL'
print('PASS: Single database verified')
"
```

**PASS CRITERIA**: New predictions appear ONLY in PostgreSQL

**Status**: [ ] NOT STARTED

---

# PHASE 2: CLEAN PREDICTOR MODULE (Difficulty: 8/10)

## Task 2.1: Extract Current Prediction Interface

**What**: Document the exact interface the system expects from predictions

**Current Interface** (from `wolf_app.py` lines 7373-7404):
```python
{
    'direction': str,       # "UP" or "DOWN"
    'confidence': float,    # 0.0 to 1.0
    'entry_price': float,
    'target_price': float,
    'stop_loss': float,
    'timeframe': str,       # "48h"
}
```

**Verification Test**:
```bash
# Run interface validator
python test_prediction_interface.py
# Expected: All required fields documented with types
```

**PASS CRITERIA**: Interface documented with all required fields

**Status**: [ ] NOT STARTED

---

## Task 2.2: Create Isolated Predictor Module

**What**: New `core/predictor_v2.py` with clean architecture

**Structure**:
```python
class GhostPredictor:
    def __init__(self, model_path: str):
        """Load trained model"""
        
    def predict(self, symbol: str, price_data: pd.DataFrame) -> PredictionResult:
        """Single entry point for predictions"""
        
    def get_confidence(self) -> float:
        """Model's calibrated confidence - NOT random"""
```

**Verification Test**:
```bash
# Unit test for new predictor
python -m pytest tests/test_predictor_v2.py -v
# Expected: All tests pass
```

**PASS CRITERIA**: 
- [ ] No random.uniform in confidence
- [ ] No hardcoded values
- [ ] Clear input/output interface
- [ ] Unit tests pass

**Status**: [ ] NOT STARTED

---

## Task 2.3: Train XGBoost on Correct Granularity

**What**: Retrain model on HOURLY data (not daily) for 48h predictions

**Current Problem**: Daily bars → 48h prediction = only ~2 data points of info

**Fix**:
```python
# In train_ml_models.py, change:
# FROM: fetch_daily_bars(symbol, limit=365)
# TO:   fetch_hourly_bars(symbol, limit=365*24)
```

**Verification Test**:
```bash
# 1. Retrain model
python train_ml_models.py --granularity=hourly

# 2. Verify training data granularity
python -c "
import joblib
model = joblib.load('models/xgboost_predictor.pkl')
print(f'Training data points: {model.n_training_samples_}')
# Should be ~8760 for 1 year of hourly data, not ~365
assert model.n_training_samples_ > 1000, 'FAIL: Still using daily data'
print('PASS: Hourly granularity confirmed')
"
```

**PASS CRITERIA**: Model trained on 1000+ data points (hourly, not daily)

**Status**: [ ] NOT STARTED

---

## Task 2.4: Remove Fake ML Models

**What**: Delete LSTM and Transformer implementations that are just indicator wrappers

**Files**:
- [ ] `core/ensemble_predictor.py` - remove fake LSTM/Transformer
- [ ] Any ensemble that averages their outputs

**Verification Test**:
```bash
# Search for fake implementations
grep -r "10-period momentum\|hardcoded weights" --include="*.py" . | wc -l
# Expected: 0

# Search for the fake LSTM
grep -r "class.*LSTM.*Predictor" --include="*.py" . | wc -l
# Expected: 0 (unless using actual PyTorch LSTM)
```

**PASS CRITERIA**: No fake ML model code remains

**Status**: [ ] NOT STARTED

---

## Task 2.5: Remove Random Confidence

**What**: Replace all `random.uniform(0.65, 0.85)` with model-based confidence

**Search Command**:
```bash
grep -rn "random.uniform" --include="*.py" .
```

**Verification Test**:
```bash
# After fix, this should return 0
grep -r "random.uniform.*0\.[67].*0\.[89]" --include="*.py" . | wc -l
# Expected: 0
```

**PASS CRITERIA**: Zero random confidence generation in prediction paths

**Status**: [ ] NOT STARTED

---

## Task 2.6: Remove INVERSE_GHOST Hack

**What**: Delete the code that flips predictions because they were wrong

**Current Code** (example):
```python
if INVERSE_GHOST:
    direction = "DOWN" if direction == "UP" else "UP"
```

**Verification Test**:
```bash
grep -ri "inverse.*ghost\|flip.*prediction" --include="*.py" . | wc -l
# Expected: 0
```

**PASS CRITERIA**: No prediction flipping hacks remain

**Status**: [ ] NOT STARTED

---

## Task 2.7: Wire New Predictor to Wolf App

**What**: Replace old prediction calls with new `GhostPredictor`

**Verification Test**:
```bash
# 1. Start system
python wolf_app.py &

# 2. Make prediction
curl -X POST http://localhost:5000/api/predict -d '{"symbol":"BTC"}'

# 3. Check logs for new predictor
grep "GhostPredictor" logs/wolf.log | tail -1
# Expected: Shows new predictor being used
```

**PASS CRITERIA**: API calls use new predictor module

**Status**: [ ] NOT STARTED

---

# PHASE 3: ACCURACY VERIFICATION (Difficulty: 6/10)

## Task 3.1: Backtest New Predictor

**What**: Run historical backtest to establish baseline accuracy

**Verification Test**:
```bash
python backtest_predictor.py --period=30d --symbol=BTC

# Expected output format:
# Total predictions: 720
# Correct: 432
# Accuracy: 60.0%
# Baseline (50%): BEAT by 10%
```

**PASS CRITERIA**: Accuracy > 50% on 30-day backtest

**Status**: [ ] NOT STARTED

---

## Task 3.2: Live Paper Trading Test

**What**: Run 24h of paper trades with new predictor

**Verification Test**:
```bash
# Run paper trading for 24h
python paper_trade.py --duration=24h --predictor=v2

# Check results
python analyze_paper_trades.py --last=24h

# Expected:
# Predictions made: > 10
# Accuracy: > 50%
# No errors in logs
```

**PASS CRITERIA**: 
- [ ] At least 10 predictions in 24h
- [ ] Accuracy > 50%
- [ ] No errors in prediction path

**Status**: [ ] NOT STARTED

---

## Task 3.3: Confidence Calibration Test

**What**: Verify confidence scores correlate with actual accuracy

**Verification Test**:
```bash
python test_confidence_calibration.py

# Expected output:
# High confidence (>0.7): 65% accurate
# Medium confidence (0.5-0.7): 55% accurate
# Low confidence (<0.5): 45% accurate
# Calibration: GOOD (higher confidence = higher accuracy)
```

**PASS CRITERIA**: Higher confidence = higher accuracy (monotonic)

**Status**: [ ] NOT STARTED

---

## Task 3.4: Production Deployment Test

**What**: Deploy to Railway and verify predictions flow correctly

**Verification Test**:
```bash
# 1. Deploy
railway up

# 2. Check Railway logs for predictions
railway logs | grep "prediction"

# 3. Verify no "insufficient aligned points" errors
railway logs | grep -c "insufficient aligned points"
# Expected: 0
```

**PASS CRITERIA**: 
- [ ] Predictions appear in Railway logs
- [ ] No "insufficient aligned points" errors
- [ ] Reconciler finds predictions to process

**Status**: [ ] NOT STARTED

---

# FINAL ACCEPTANCE TEST

## All-in-One Verification

```bash
#!/bin/bash
# acceptance_test.sh

echo "=== GHOST REBUILD ACCEPTANCE TEST ==="

# Database
echo -n "1. Single database (PostgreSQL only): "
sqlite_count=$(grep -r "sqlite3.connect" --include="*.py" . 2>/dev/null | wc -l)
[ "$sqlite_count" -eq 0 ] && echo "PASS" || echo "FAIL ($sqlite_count SQLite connections)"

# No fake ML
echo -n "2. No fake ML models: "
fake_count=$(grep -r "10-period momentum\|hardcoded weights" --include="*.py" . 2>/dev/null | wc -l)
[ "$fake_count" -eq 0 ] && echo "PASS" || echo "FAIL ($fake_count fake models)"

# No random confidence
echo -n "3. No random confidence: "
random_count=$(grep -r "random.uniform.*0\.[67].*0\.[89]" --include="*.py" . 2>/dev/null | wc -l)
[ "$random_count" -eq 0 ] && echo "PASS" || echo "FAIL ($random_count random confidences)"

# No inverse hack
echo -n "4. No inverse hack: "
inverse_count=$(grep -ri "inverse.*ghost" --include="*.py" . 2>/dev/null | wc -l)
[ "$inverse_count" -eq 0 ] && echo "PASS" || echo "FAIL ($inverse_count inverse hacks)"

# Accuracy > 50%
echo -n "5. Backtest accuracy > 50%: "
# Run backtest and check result
accuracy=$(python backtest_predictor.py --quick 2>/dev/null | grep "Accuracy" | awk '{print $2}' | tr -d '%')
[ "$(echo "$accuracy > 50" | bc)" -eq 1 ] && echo "PASS ($accuracy%)" || echo "FAIL ($accuracy%)"

echo "=== END ACCEPTANCE TEST ==="
```

**FINAL PASS CRITERIA**: All 5 tests pass

---

# EXECUTION ORDER

1. **Phase 1** (Do First): Database consolidation
   - Lower risk (4/10 difficulty)
   - Creates stable foundation
   - Easy to verify

2. **Phase 2** (After Phase 1): Clean predictor
   - Higher risk (8/10 difficulty)
   - Depends on stable database
   - Core logic rebuild

3. **Phase 3** (After Phase 2): Verification
   - Proves system works
   - Establishes baseline metrics
   - Production deployment

---

# CHECKBOXES FOR TRACKING

## Phase 1: Databases
- [x] 1.1 Audit current database state (27 DBs, 392 connections found)
- [x] 1.2 Enable PostgreSQL as primary store (Dockerfile updated)
- [x] 1.3 Create migration script (scripts/migrate_sqlite_to_postgres.py)
- [ ] 1.4 Deploy and verify PostgreSQL working

## Phase 2: Predictor
- [ ] 2.1 Extract prediction interface
- [ ] 2.2 Create isolated predictor module
- [ ] 2.3 Train on correct granularity
- [ ] 2.4 Remove fake ML models
- [ ] 2.5 Remove random confidence
- [ ] 2.6 Remove INVERSE_GHOST hack
- [ ] 2.7 Wire new predictor to wolf_app

## Phase 3: Verification
- [ ] 3.1 Backtest new predictor
- [ ] 3.2 Live paper trading test
- [ ] 3.3 Confidence calibration test
- [ ] 3.4 Production deployment test

## Final
- [ ] All acceptance tests pass
- [ ] Deployed to production
- [ ] 24h monitoring shows no errors

---

*Last Updated: Jan 5, 2026*
*Status: PHASE 1 - 3/4 TASKS COMPLETE - DEPLOY REQUIRED*
