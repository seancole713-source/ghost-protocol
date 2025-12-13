# 🧠 Ghost ML Model Training Guide

## Current Status

**Infrastructure**: ✅ Complete
- ML trainer code exists (`core/ml_trainer.py`)
- Ensemble predictor has model loading logic
- Training endpoint available (`POST /api/v3/ml/train`)
- Outcome reconciliation system running

**Models**: ❌ **NOT YET TRAINED**
- `models/production/` directory is empty
- Ensemble predictor falls back to heuristics (40-50% confidence)
- Need to execute training workflow to create model files

---

## Why Ghost Uses Heuristics Instead of ML Models

From `core/ensemble_predictor.py` lines 113-170:

```python
# Try to load trained model
model_data = load_model()
if model_data:
    # Use trained XGBoost model (65-75% confidence)
    prediction = self.model.predict(features)
else:
    # FALLBACK: Use heuristic rules (40-50% confidence)
    rsi = features.get("rsi", 50)
    if rsi < 30: score += 2  # Hardcoded oversold signal
    elif rsi > 70: score -= 2  # Hardcoded overbought signal
```

**Result**: No trained models → Always uses fallback → 40-50% base confidence → Final 55-65% (below 70% target)

---

## How to Train Models

### Option 1: Railway Production (Recommended)

```bash
# Trigger training via API endpoint
curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/ml/train?min_predictions=50"

# Response:
{
  "ok": true,
  "symbols_trained": 15,
  "total_predictions": 2847,
  "models": {
    "BTC": {"accuracy": 0.68, "train_samples": 380},
    "ETH": {"accuracy": 0.65, "train_samples": 290},
    "WOLF": {"accuracy": 0.72, "train_samples": 156}
  }
}
```

**Requirements**:
- PostgreSQL must have reconciled predictions (`actual_direction IS NOT NULL`)
- Min 50 predictions per symbol to train
- Database query: `SELECT COUNT(*) FROM ghost_predictions WHERE actual_direction IS NOT NULL`

### Option 2: Local Training Script

```bash
# Set Railway database URL
export DATABASE_URL="postgresql://user:pass@hostname:5432/railway"

# Run training script
python train_models_now.py 50

# Output:
# ✅ Training complete!
#    Symbols trained: 8
#    Total predictions: 1247
#    
# 📈 Model Performance:
#    BTC   : 68.5% accuracy ( 380 samples)
#    ETH   : 65.2% accuracy ( 290 samples)
#    WOLF  : 72.1% accuracy ( 156 samples)
```

### Option 3: Manual Python

```python
from core.ml_trainer import get_ml_trainer
import asyncio

trainer = get_ml_trainer()
results = asyncio.run(trainer.train_from_postgres(min_predictions=50))

print(f"Trained {results['symbols_trained']} symbols")
print(f"Models saved to: {results['model_dir']}")
```

---

## Training Data Requirements

### What Ghost Needs

1. **Reconciled Predictions** (PostgreSQL `ghost_predictions` table):
   ```sql
   SELECT 
     symbol,
     features_json,        -- Feature vector (150+ features)
     predicted_direction,  -- UP/DOWN/FLAT
     actual_direction,     -- Actual outcome (NOT NULL = reconciled)
     was_correct,          -- Boolean accuracy flag
     confidence
   FROM ghost_predictions
   WHERE actual_direction IS NOT NULL  -- Key filter
   ORDER BY run_at DESC
   ```

2. **Minimum Data**:
   - 50+ predictions per symbol to train (configurable via `min_predictions`)
   - Features must be available (`features_json IS NOT NULL`)
   - Outcomes must be reconciled (`actual_direction IS NOT NULL`)

3. **Data Collection Process**:
   - Ghost generates predictions every 6h/24h/48h
   - Reconciler waits for prediction horizon (e.g., 48 hours)
   - Fetches actual price at t+48h
   - Computes `was_correct = (predicted_direction == actual_direction)`
   - Stores outcome in database

### Check Data Availability

```bash
# Check total reconciled predictions
curl "https://ghost-protocol-production.up.railway.app/api/admin/reconcile/outcomes"

# Response shows:
# {
#   "success": 124,
#   "no_data": 3,
#   "error": 0,
#   "reconciled_count": 124
# }
```

Or query PostgreSQL directly:

```sql
-- Count reconciled predictions by symbol
SELECT 
  symbol, 
  COUNT(*) as predictions,
  SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
  ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) * 100, 1) as accuracy_pct
FROM ghost_predictions
WHERE actual_direction IS NOT NULL
GROUP BY symbol
ORDER BY predictions DESC;

-- Example output:
--  symbol | predictions | correct | accuracy_pct
-- --------+-------------+---------+--------------
--  BTC    |         380 |     260 |         68.4
--  ETH    |         290 |     189 |         65.2
--  WOLF   |         156 |     112 |         71.8
--  SOL    |         142 |      89 |         62.7
```

---

## Expected Results

### Before Training (Current State)

```python
# Ensemble predictor returns:
{
  "direction": "UP",
  "confidence": 0.45,           # Heuristic fallback (40-50%)
  "method": "technical_rules"
}

# After signal calibration:
{
  "confidence": 0.62,           # Boosted to 55-65%
  "status": "MONITOR ONLY"      # Below 70% threshold
}
```

### After Training (Target State)

```python
# Ensemble predictor returns:
{
  "direction": "UP",
  "confidence": 0.70,           # Trained XGBoost (65-75%)
  "method": "xgboost_trained",
  "model_accuracy": 0.68        # Historical test accuracy
}

# After signal calibration:
{
  "confidence": 0.78,           # Boosted to 70-85%
  "status": "TRADE ELIGIBLE"    # Above 70% threshold ✅
}
```

### Confidence Improvement

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Ensemble Base | 40-50% | 65-75% | **+20-25%** |
| Signal Calibration | +10-15% | +5-10% | More accurate |
| Final Confidence | 55-65% | 70-85% | **+15-20%** |
| Trade Eligibility | ❌ Monitor Only | ✅ Trade Signals | **Enabled** |

---

## Troubleshooting

### Error: "No training data available"

**Cause**: No reconciled predictions in database

**Solution**:
```bash
# 1. Check prediction count
curl "https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest"

# 2. Run reconciliation manually
curl -X POST "https://ghost-protocol-production.up.railway.app/api/admin/reconcile/outcomes"

# 3. Wait 48 hours for predictions to mature
# 4. Try training again
```

### Error: "PostgreSQL not configured"

**Cause**: Missing `DATABASE_URL` environment variable

**Solution**:
```bash
# Get Railway database URL
railway variables

# Set locally
export DATABASE_URL="postgresql://postgres:pass@hostname:5432/railway"

# Or add to .env file
echo "DATABASE_URL=postgresql://..." >> .env
```

### Error: "Insufficient training data (X < 50 samples)"

**Cause**: Not enough predictions per symbol

**Solution**:
```bash
# Lower minimum threshold
curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/ml/train?min_predictions=20"

# Or wait for more predictions to accumulate
```

---

## Verification

### Check if Models Exist

```bash
# On Railway (via API)
curl "https://ghost-protocol-production.up.railway.app/api/v3/ml/models"

# Response:
{
  "ok": true,
  "models_found": 8,
  "models": [
    {
      "symbol": "BTC",
      "accuracy": 0.685,
      "samples": 380,
      "trained_at": "2025-12-13T14:30:00Z"
    }
  ]
}
```

```bash
# Locally
ls -lh models/production/

# Output:
# ghost_model_BTC.pkl   (245KB)
# ghost_model_ETH.pkl   (198KB)
# ghost_model_WOLF.pkl  (156KB)
```

### Verify Ensemble Uses Trained Models

```bash
# Generate prediction and check metadata
curl "https://ghost-protocol-production.up.railway.app/api/predictions/run?symbol=BTC" | jq '.confidence_metadata'

# Look for:
{
  "method": "ensemble_xgboost_trained",  # ✅ Using trained model
  "base_confidence": 0.70,               # ✅ Higher than heuristic
  "model_accuracy": 0.685,               # ✅ Historical performance
  "calibrated": 0.78                     # ✅ Above 70% threshold
}

# vs old output:
{
  "method": "technical_rules",           # ❌ Heuristic fallback
  "base_confidence": 0.45,               # ❌ Low baseline
  "calibrated": 0.62                     # ❌ Below 70%
}
```

---

## Next Steps

1. **Check current data**: How many reconciled predictions exist?
   ```bash
   curl "https://ghost-protocol-production.up.railway.app/api/admin/reconcile/outcomes"
   ```

2. **If data available (>50 per symbol)**: Train models now
   ```bash
   curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/ml/train?min_predictions=50"
   ```

3. **If insufficient data**: Wait for reconciliation to collect more outcomes
   - Reconciler runs every 5 minutes automatically
   - Need 2+ weeks of predictions for good training data
   - Each prediction takes 48h to reconcile

4. **Once trained**: Verify confidence improvement
   ```bash
   # Generate new prediction
   curl "https://ghost-protocol-production.up.railway.app/api/predictions/run?symbol=BTC"
   
   # Check confidence metadata
   # Should see 70%+ confidence instead of 55-65%
   ```

5. **Retrain weekly**: Keep models updated with new data
   ```bash
   # Add to cron
   0 2 * * 0 curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/ml/train"
   ```

---

## Summary

Ghost has **complete ML infrastructure** but models haven't been trained yet:

- ✅ Training code exists (`core/ml_trainer.py`)
- ✅ Outcome reconciliation running (collecting data)
- ✅ API endpoint available (`/api/v3/ml/train`)
- ✅ Ensemble predictor ready to load models
- ❌ **No trained model files** in `models/production/`
- ❌ **Currently using heuristic fallbacks** (40-50% confidence)

**Action Required**: Execute training workflow once sufficient reconciled predictions exist (50+ per symbol). This will create model files and enable ensemble predictor to use trained XGBoost instead of hardcoded rules, boosting confidence from 55-65% → 70-85%.
