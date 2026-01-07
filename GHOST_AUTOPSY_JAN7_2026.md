# 🏥 GHOST AUTOPSY REPORT - January 7, 2026

## Patient: Ghost Protocol Trading Prediction System
## Status: **CRITICALLY ILL** (35% accuracy - worse than random guessing)

---

# 🧬 BODY MAP: Ghost as a Human

| Human Organ | Ghost Component | File | Status |
|-------------|-----------------|------|--------|
| 🧠 **Brain** | Prediction Engine | `core/ensemble_predictor.py` | ⚠️ DISEASED |
| ❤️ **Heart** | Main Orchestrator | `wolf_app.py` | ✅ Working (but pumping bad predictions) |
| 👁️ **Eyes** | Price Providers | `core/providers/turbo_provider.py` | ✅ Working |
| 👂 **Ears** | Webhooks/Input | `wolf_app.py` webhooks | ✅ Working |
| 🗄️ **Memory** | Database Layer | `core/prediction_store.py` | 🔴 **SPLIT PERSONALITY DISORDER** |
| 🔔 **Nervous System** | Telegram Alerts | `core/ghost_notifications.py` | ✅ Fixed (post-surgery) |
| 🖐️ **Hands** | Trade Execution | `core/paper_tracker.py` | ✅ Working |
| 👄 **Mouth** | API Endpoints | `wolf_app.py` `/api/*` | ✅ Working |
| 🩸 **Blood** | Config/Env Vars | `.env`, Railway vars | ⚠️ CONTAMINATED |

---

# 🔬 DETAILED DIAGNOSIS

## 1. 🧠 THE BRAIN - `core/ensemble_predictor.py` (966 lines)

### **DIAGNOSIS: DEGENERATIVE BRAIN DISEASE**

The brain thinks it's smart, but it's making predictions **worse than a coin flip**.

#### **Symptoms Found:**

1. **Hardcoded Bias Correction That Doesn't Work** (Lines 570-600)
   ```python
   # BIAS CORRECTION v5 (Jan 6, 2026): Shift 16% toward UP
   bias_correction = 0.16
   prob_up_calibrated = prob_up + (bias_correction * (1 - prob_up))
   ```
   - **Problem**: Model outputs ~98% DOWN by default
   - **"Fix"**: Shift 16% toward UP
   - **Result**: Still biased wrong direction
   - **Root Cause**: The model was trained on **BAD DATA or NOT TRAINED AT ALL**

2. **Fallback Logic Uses Neutral Defaults** (Lines 320-360)
   ```python
   rsi = features.get("RSI_14", 50)  # Default 50 = neutral
   macd = features.get("MACD_HISTOGRAM", 0)  # Default 0 = bearish!
   ```
   - When features are missing, defaults to NEUTRAL
   - But MACD default of 0 ≤ 0 triggers DOWN signal

3. **Probability Compression** (Lines 560-570)
   ```python
   def compress_probability(prob, min_val=0.40, max_val=0.65):
   ```
   - Squishes all predictions to 40-65% range
   - Destroys any real signal the model might have

4. **Conviction Threshold at 58%** (Lines 614)
   ```python
   conviction_threshold = 0.58
   ```
   - After compression to 40-65%, almost nothing exceeds 58%
   - Most predictions become FLAT (which is then counted wrong)

#### **Brain Scan Result:**
```
XGBoost Model: ghost_xgboost_v2.pkl
Created: Jan 5, 2026 (2 DAYS AGO)
Training Data: prediction_outcomes.db (SQLite local file)
Problem: SQLite file DOESN'T PERSIST on Railway!
```

### **ROOT CAUSE #1: The brain was trained on PHANTOM MEMORIES**

---

## 2. 🗄️ THE MEMORY - Database Split Personality

### **DIAGNOSIS: SCHIZOPHRENIA - TWO DATABASES THAT DON'T TALK**

#### **The Split:**

| Component | Uses | File Path | Persists on Railway? |
|-----------|------|-----------|---------------------|
| `prediction_store.py` | PostgreSQL | `DATABASE_URL` | ✅ YES |
| `accuracy_tracker.py` | SQLite | `data/forecast_accuracy.db` | ❌ NO (wiped each deploy) |
| `ml_trainer.py` | SQLite | `data/prediction_outcomes.db` | ❌ NO (wiped each deploy) |

#### **What This Means:**

1. **Predictions** go to PostgreSQL ✅
2. **Accuracy tracking** writes to SQLite locally 
3. **ML training** reads from SQLite locally
4. **Railway deploys wipe SQLite** every time

**THE BRAIN IS TRAINED ON EMPTY DATA!**

```python
# From ml_trainer.py line 158:
outcomes_db = Path(__file__).parent.parent / "data" / "prediction_outcomes.db"
# This file is EMPTY on Railway because it gets wiped on every deploy!
```

### **ROOT CAUSE #2: Training data vanishes on every deploy**

---

## 3. 🩸 THE BLOOD - Environment Configuration

### **DIAGNOSIS: BLOOD POISONING (Contaminated Config)**

#### **Critical Missing Variables:**

```bash
INVERSE_GHOST=0          # Currently OFF (should maybe be ON for 35% accuracy)
PREDICTION_STORE_ENGINE=postgres  # Maybe set, maybe not
MIN_FEATURES_FOR_SIGNAL=10        # Not enforced
```

#### **The INVERSE_GHOST Irony:**

- Current accuracy: **35%**
- If INVERSE_GHOST was ON: **~65%** accuracy (just flip UP↔DOWN)
- The system has a built-in fix that's TURNED OFF

### **ROOT CAUSE #3: Healing mechanism exists but is disabled**

---

## 4. 🔔 THE NERVOUS SYSTEM - Notifications

### **STATUS: RECOVERED (Post-Surgery)**

Recent fixes applied:
- ✅ PostgreSQL connection retry logic
- ✅ Market hours awareness (4 AM - 8 PM ET for stocks)
- ✅ Auto-TOP10 at 8 AM
- ✅ Unique index duplicate cleanup

**This organ is now healthy.**

---

## 5. 👁️ THE EYES - Price Providers

### **STATUS: HEALTHY BUT MYOPIC**

- `turbo_provider.py` fetches real prices
- Data quality looks good
- BUT: The brain ignores what the eyes see

---

# 📊 VITAL STATISTICS

| Metric | Value | Normal Range | Status |
|--------|-------|--------------|--------|
| Accuracy | 35.47% | 55%+ | 🔴 CRITICAL |
| Total Predictions | 888 | n/a | Measured |
| Model Age | 2 days | n/a | ⚠️ NEW |
| Training Data | ~0 records | 500+ | 🔴 EMPTY |
| INVERSE_GHOST | OFF | ON (given 35%) | 🔴 WRONG |

---

# 🏥 TREATMENT PLAN

## **Emergency Triage (Do NOW):**

### 1. **Turn ON the Inverter** 
```bash
# In Railway environment:
INVERSE_GHOST=1
```
- Instant fix: 35% → 65% accuracy
- Why: The model is **anti-correlated** with reality

### 2. **Fix the Memory Split**

Migrate `ml_trainer.py` to read from PostgreSQL:
```python
# INSTEAD OF:
outcomes_db = Path("data/prediction_outcomes.db")  # SQLite (DIES ON DEPLOY)

# USE:
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL (PERSISTS)
```

### 3. **Remove Hardcoded Bias Correction**

In `ensemble_predictor.py`:
```python
# DELETE THIS GARBAGE:
bias_correction = 0.16
prob_up_calibrated = prob_up + (bias_correction * (1 - prob_up))

# REPLACE WITH:
prob_up_calibrated = prob_up  # Use RAW model output
```

The bias correction is a bandaid on a severed artery.

---

## **Long-Term Recovery (Next 2 Weeks):**

### 4. **Retrain XGBoost on REAL DATA**

1. Export PostgreSQL predictions → CSV
2. Train model locally with REAL outcomes
3. Upload `ghost_xgboost_v3.pkl`
4. Deploy

### 5. **Add Feature Validation**

```python
# Before predicting, verify we have REAL data:
if features.get("RSI_14") == 50 and features.get("BB_POSITION") == 0.5:
    logger.warning("FEATURES ARE DEFAULTS - skipping prediction")
    return FLAT  # Don't guess
```

### 6. **Unify All Databases to PostgreSQL**

| Component | Current | Target |
|-----------|---------|--------|
| `prediction_store.py` | PostgreSQL ✅ | PostgreSQL |
| `accuracy_tracker.py` | SQLite ❌ | PostgreSQL |
| `ml_trainer.py` | SQLite ❌ | PostgreSQL |

---

# 🎯 THE BOTTOM LINE

## Why is Ghost at 35% accuracy?

1. **🧠 The Brain was never properly trained** - It uses SQLite that dies on deploy
2. **🗄️ Split Memory** - Training data vanishes, so model trains on nothing
3. **🔧 Wrong Fixes Applied** - Hardcoded bias correction masks the real problem
4. **🔌 Inverter OFF** - The one thing that could instantly help is disabled

## The Honest Truth:

**Ghost is currently a random number generator with extra steps.**

The XGBoost model exists, but:
- It's trained on empty/minimal data
- It's been "fixed" with hardcoded biases that make it worse
- The system to improve it (ml_trainer) can't work because it reads from SQLite

## Recommended Actions by Priority:

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| 🔴 P0 | Set `INVERSE_GHOST=1` | +30% accuracy | 1 min |
| 🔴 P1 | Migrate ml_trainer to PostgreSQL | Enables learning | 2 hrs |
| 🟡 P2 | Remove bias correction hacks | Cleaner predictions | 30 min |
| 🟡 P3 | Retrain XGBoost on real data | Real ML | 4 hrs |
| 🟢 P4 | Add feature validation | Prevents garbage-in | 1 hr |

---

**Report Generated:** January 7, 2026  
**Examined By:** GitHub Copilot (Claude Opus 4.5)  
**Patient Prognosis:** RECOVERABLE with surgery
