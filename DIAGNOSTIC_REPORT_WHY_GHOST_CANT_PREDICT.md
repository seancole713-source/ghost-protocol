# 🔍 COMPLETE DIAGNOSTIC REPORT: Why Ghost Can't Predict Accurately

## Executive Summary

**Ghost's prediction system was never fully wired together.** The individual components exist (prediction creation, accuracy tracking, outcome reconciliation), but the **critical data pipeline that connects them is broken or missing**.

The error "insufficient aligned points (0)" isn't a bug in the reconciler - it's telling us that **no actual price data was ever collected** to compare against predictions.

---

## THE ROOT CAUSE ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GHOST PREDICTION ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   STEP 1: Create Prediction                                             │
│   ┌──────────────┐         ┌──────────────┐                            │
│   │ wolf_app.py  │────────▶│  PostgreSQL  │  (PREDICTION_STORE_ENGINE) │
│   │ Line 7865    │         │  predictions │                            │
│   └──────────────┘         └──────────────┘                            │
│                                    │                                    │
│                                    │ Predictions stored here ✅         │
│                                    ▼                                    │
│   STEP 2: Collect Actual Prices (BROKEN!)                              │
│   ┌──────────────────────┐         ┌──────────────┐                    │
│   │ _append_actual_prices│────────▶│   SQLite     │                    │
│   │ Line 18389-18417     │ READS   │  (EMPTY!)    │                    │
│   └──────────────────────┘         └──────────────┘                    │
│           │                                │                           │
│           │ Queries SQLite for predictions │                           │
│           │ SQLite is EMPTY (predictions   │                           │
│           │ are in PostgreSQL!)            │                           │
│           │                                │                           │
│           ▼                                ▼                           │
│   "No predictions found" ────────▶ "No actual points added"           │
│                                                                         │
│   STEP 3: Reconcile Outcomes (FAILS!)                                  │
│   ┌──────────────────────┐                                              │
│   │ outcome_reconciler   │                                              │
│   │ Line 85-87           │                                              │
│   └──────────────────────┘                                              │
│           │                                                             │
│           │ Calls get_prediction_points(pred_id, kind="actual")        │
│           │ Returns: [] (empty list)                                   │
│           │                                                             │
│           ▼                                                             │
│   "insufficient aligned points (0)" ◀─── THE ERROR YOU SEE             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ISSUE #1: Database Mismatch (The Immediate Cause)

### The Code

**File: `wolf_app.py` Line 18388-18417**
```python
def _append_actual_prices():
    """Append current live prices to active predictions"""
    import sqlite3

    conn = sqlite3.connect(predictor.DB_PATH)  # <-- HARDCODED SQLITE!
    try:
        # Get active predictions (not yet closed)
        rows = conn.execute("""
            SELECT p.id, p.symbol, p.run_at, p.horizon_h
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            WHERE o.prediction_id IS NULL
              AND (p.run_at + (p.horizon_h * 3600)) > ?
        """, (now,)).fetchall()
```

### The Problem

1. `PREDICTION_STORE_ENGINE=postgres` → Predictions saved to **PostgreSQL**
2. `_append_actual_prices()` reads from **SQLite** (hardcoded!)
3. SQLite `predictions` table is **empty**
4. `rows = []` → No actual prices collected
5. Reconciler finds 0 actual points → "insufficient aligned points (0)"

### Impact

**100% of predictions fail to reconcile.** The accuracy tracking system never receives any actual price data to compare against forecasts.

---

## ISSUE #2: Crypto Actual Points Table Never Populated

### The Code

**File: `core/crypto/crypto_predictor.py` Line 78**
```python
# Table is CREATED...
c.execute("""
    CREATE TABLE IF NOT EXISTS crypto_actual_points (
        prediction_id TEXT NOT NULL,
        ts REAL NOT NULL,
        price REAL NOT NULL,
        provider TEXT,
        FOREIGN KEY (prediction_id) REFERENCES crypto_predictions(id)
    )
""")
```

**File: `wolf_app.py` Lines 13630, 13644**
```python
# Table is QUERIED...
JOIN crypto_actual_points ap ON cp.id = ap.prediction_id
```

### The Problem

**The `crypto_actual_points` table is:**
- ✅ Created in `_init_tables()`
- ✅ Joined in accuracy queries
- ❌ **NEVER POPULATED** - No INSERT statement exists anywhere

### Search Results
```bash
$ grep -rn "INSERT INTO crypto_actual" --include="*.py"
# NO RESULTS - Table is never written to!
```

### Impact

**0% of crypto predictions can ever be evaluated.** The accuracy calculation joins against an empty table → always returns 0 accurate predictions.

---

## ISSUE #3: The "Fake ML" Models

### The Bootstrap Problem

The `ensemble_predictor.py` was created in commit `12e627f` ("feat: SUPERMAN MODE - Add 5 advanced AI trading modules") with this comment:

```python
# Simulate LSTM prediction (TODO: Train actual LSTM)
# For now, use price momentum as proxy
```

This "TODO" was never completed. The "LSTM" and "Transformer" models were:

| Model | Claimed | Reality |
|-------|---------|---------|
| LSTM | "Deep learning for temporal patterns" | `if momentum > 0.5: direction = "UP"` |
| Transformer | "Attention mechanisms" | `attention_score = confidence*0.5 + (1-volatility/0.1)*0.3` |
| XGBoost | "Real ML model" | Actually trained, but on wrong data granularity |

### The Confidence Problem

**File: `core/daily_top_10_scanner.py` Line 301** (before fix)
```python
confidence = random.uniform(0.60, 0.75)  # RANDOM!
```

Multiple files used `random.uniform()` for confidence values, making predictions statistically meaningless.

---

## ISSUE #4: XGBoost Trained on Wrong Granularity

### The Training Code

**Original `train_ml_models.py`:**
```python
# Fetch daily bars for training
url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit=365"
```

### The Problem

- Model trained on **365 daily bars** 
- Predictions made for **48 hours ahead**
- Daily bars have **~2 data points** in a 48h window
- Insufficient data resolution for the prediction horizon

### Impact

Model was effectively guessing - not enough temporal resolution to learn meaningful patterns.

---

## SUMMARY: The Real Root Causes

| # | Issue | Impact | Line(s) |
|---|-------|--------|---------|
| 1 | `_append_actual_prices()` reads SQLite, not PostgreSQL | 100% predictions unreconcilable | wolf_app.py:18393 |
| 2 | `crypto_actual_points` table never populated | 0% crypto predictions evaluable | N/A (missing code) |
| 3 | LSTM/Transformer were fake implementations | "Ensemble" was just weighted momentum | ensemble_predictor.py |
| 4 | XGBoost trained on daily data for 48h predictions | Model couldn't learn useful patterns | train_ml_models.py |
| 5 | Random confidence values in fallback paths | Predictions statistically meaningless | daily_top_10_scanner.py:301 |

---

## WHAT "FIXING THE DATABASE" AND "FIXING THE MODEL" DON'T ADDRESS

Even after:
- ✅ Setting `PREDICTION_STORE_ENGINE=postgres` 
- ✅ Removing fake LSTM/Transformer
- ✅ Retraining XGBoost on hourly data

**The core issue remains:** `_append_actual_prices()` is hardcoded to read from SQLite.

Until that function is modified to use `get_prediction_store()` abstraction, **actual prices will never be collected**, and **accuracy will never be calculated**.

---

## THE FIX NEEDED

**File: `wolf_app.py` Line 18388-18417**

Change from:
```python
def _append_actual_prices():
    import sqlite3
    conn = sqlite3.connect(predictor.DB_PATH)  # HARDCODED!
```

To:
```python
def _append_actual_prices():
    from core.prediction_store import get_prediction_store
    store = get_prediction_store()  # Uses PostgreSQL when configured
    rows = store.get_pending_predictions()  # Or equivalent method
```

---

## CONCLUSION

**Ghost wasn't predicting badly. Ghost wasn't predicting at all.**

The error "insufficient aligned points (0)" was the system correctly reporting that it had **zero data points** to evaluate accuracy against. The predictions were being made and stored, but the **feedback loop was never connected**.

This is a classic case of building components without integration testing. Each piece worked in isolation:
- Predictions → stored ✅
- Actual price fetcher → runs ✅ (but reads wrong database)
- Reconciler → runs ✅ (but finds no data)
- Accuracy calculator → runs ✅ (but joins empty tables)

The system gave the illusion of working while actually producing no valid accuracy data.
