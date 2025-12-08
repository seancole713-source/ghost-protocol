# 🔧 GHOST V3 PREDICTION REPAIR PLAN

## Generated: 2025-11-21

## Status: Phase 1 RECON Complete → Executing Surgical Fixes

---

## RECON SUMMARY

### Prediction Pipeline Architecture

**Entry Points**:

1. **HTTP Endpoint**: `/api/predict/run` (POST) - wolf_app.py:5779
   - Accepts: `{"symbol": "WOLF"}`
   - Calls: `predictor.create_prediction()`
   - Stores: `./data/ghost_predictions.db`
   - Returns: `{ok, prediction_id, symbol, run_at, confidence, direction}`

1. **Scheduled System**: `core/beast_scheduler.py`
   - Stocks: 07:55, 09:35, 12:00, 15:10 CT
   - Crypto: Every 2 hours
   - Calls: `RUN_PREDICTION_FUNC` (injected dependency)
   - **CRITICAL**: Must verify this connects to predictor.create_prediction()


### Database Architecture

**PRIMARY SYSTEM**: `services/predictor.py` → `./data/ghost_predictions.db`

```sql
predictions: id, symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag
prediction_points: id, prediction_id, ts, kind (forecast/actual), price
outcomes: prediction_id, closed_at, mae, mape, rmse, hit_direction, hit_ratio_window

```text

- ✅ WORKING: Predictions ARE being written via /api/predict/run
- ✅ CONFIRMED: predictor.create_prediction() returns prediction_id successfully


**PARALLEL SYSTEM**: `core/prediction_tracker.py` → `data/wolf.db`

```sql

ghost_predictions: id, symbol, predicted_at, check_at, predicted_price, predicted_direction, ...
ghost_accuracy_stats: period, total_predictions, correct_predictions, accuracy_pct, ...

```text

- ⚠️ UNCLEAR: Has `calculate_accuracy()` function but not connected to V3 endpoint
- ⚠️ UNCLEAR: Used by scheduled predictions or just legacy?


### The 0.0% Accuracy Bug 🚨

**ROOT CAUSE CONFIRMED**: `api/cockpit_v3_live_endpoints.py` Line 661-673

```python

@router.get("/accuracy/summary")
async def get_accuracy_summary():
    try:

        # TODO: Query accuracy ledger  # ← NEVER IMPLEMENTED

        return {
            "daily_accuracy_pct": 0.0,     # ← HARDCODED
            "weekly_accuracy_pct": 0.0,    # ← HARDCODED
            "monthly_accuracy_pct": 0.0,   # ← HARDCODED
            "correct": 0,                  # ← HARDCODED
            "wrong": 0,                    # ← HARDCODED
            "pending": 0,                  # ← HARDCODED
            ...
        }

```text

**Impact**:

- Telegram reports: "Ghost Accuracy = 0.0%"
- Cockpit V3 accuracy panel: Shows "--" or 0%
- User perception: Ghost isn't making predictions (FALSE - predictions ARE running!)


### Cockpit V3 Current State

**File**: `static/cockpit_v3.js`

- Cash-App style UI ✅
- Minimal design ✅
- Update interval: 5 seconds ✅


**Panels**:

1. Header: System time, status indicator, mode selector
2. Top Movers: Stock/Crypto tabs (needs wiring)
3. Forecast Strip: 24-48h, 2-5d, 7-14d (needs wiring)
4. Accuracy/Graph: Shows accuracy (BROKEN - needs data)
5. Providers/System: Health status (partially working)


**V3 Endpoints Status**:

- ✅ `/api/v3/cockpit/status` - EXISTS
- ❌ `/api/v3/accuracy/summary` - HARDCODED ZEROS
- ❓ `/api/v3/predictions/latest` - NEED TO VERIFY
- ❓ `/api/v3/hunter/feed` - NEED TO VERIFY
- ❓ `/api/v3/providers/health` - NEED TO VERIFY


---

## SURGICAL FIX PLAN

### Phase 2: Prediction Pipeline Normalization ✅

**Goal**: Ensure all prediction paths write to same database

**Tasks**:

1. ✅ Verify `/api/predict/run` schema (currently: `{"symbol": "WOLF"}`)
2. ⚠️ Find `run_prediction` function that beast_scheduler uses
3. ⚠️ Confirm it calls `predictor.create_prediction()`
4. ⚠️ If not, create wrapper function


**Expected Function Signature**:

```python

def run_prediction(symbol: str, market: str = "stock", horizon: str = "SHORT") -> dict:
    """
    Generate prediction and store in predictor DB
    Called by beast_scheduler and other scheduled systems

    Returns:
        {
            'ok': True,
            'prediction_id': int,
            'symbol': str,
            'direction': 'UP'|'DOWN'|'FLAT',
            'confidence': float,
            'horizon_h': int
        }
    """

```text

### Phase 3: Accuracy Engine Repair 🚨 CRITICAL

**Goal**: Fix 0.0% accuracy display by querying real data

**File to Fix**: `api/cockpit_v3_live_endpoints.py:661-673`

**New Implementation**:

```python

@router.get("/accuracy/summary")
async def get_accuracy_summary():
    """Get prediction accuracy metrics from database"""
    try:
        import sqlite3
        from services import predictor

        conn = sqlite3.connect(predictor.DB_PATH)

        # Query predictions from last 7/30 days

        now = time.time()
        day_ago = now - (24 * 3600)
        week_ago = now - (7 *24* 3600)
        month_ago = now - (30 *24* 3600)

        # Get predictions with outcomes

        predictions = conn.execute("""
            SELECT
                p.id,
                p.symbol,
                p.run_at,
                p.direction,
                p.confidence,
                o.hit_direction,
                o.hit_ratio_window
            FROM predictions p
            LEFT JOIN outcomes o ON p.id = o.prediction_id
            WHERE p.run_at >= ?
            ORDER BY p.run_at DESC
        """, (month_ago,)).fetchall()

        conn.close()

        # Calculate accuracy by time window

        daily = [p for p in predictions if p[2] >= day_ago]
        weekly = [p for p in predictions if p[2] >= week_ago]
        monthly = predictions

        def calc_accuracy(preds):
            if not preds:
                return 0.0, 0, 0, 0

            with_outcomes = [p for p in preds if p[5] is not None]
            if not with_outcomes:
                return 0.0, 0, 0, len(preds)

            correct = sum(1 for p in with_outcomes if p[5] == 1)
            wrong = sum(1 for p in with_outcomes if p[5] == 0)
            pending = len(preds) - len(with_outcomes)

            accuracy = (correct / len(with_outcomes) * 100) if with_outcomes else 0.0
            return accuracy, correct, wrong, pending

        daily_acc, daily_corr, daily_wrong, daily_pend = calc_accuracy(daily)
        weekly_acc, weekly_corr, weekly_wrong, weekly_pend = calc_accuracy(weekly)
        monthly_acc, monthly_corr, monthly_wrong, monthly_pend = calc_accuracy(monthly)

        # Get last tune timestamp (from latest prediction)

        last_tune = max([p[2] for p in predictions]) if predictions else None

        return {
            "daily_accuracy_pct": round(daily_acc, 1),
            "weekly_accuracy_pct": round(weekly_acc, 1),
            "monthly_accuracy_pct": round(monthly_acc, 1),
            "correct": daily_corr,
            "warning": 0,  # Can add warning threshold logic
            "wrong": daily_wrong,
            "pending": daily_pend,
            "last_tune_ts": int(last_tune) if last_tune else None,
            "config_name": "ghost-av1",
            "total_predictions": len(daily)
        }

    except Exception as e:
        LOGGER.error(f"Accuracy summary failed: {e}", exc_info=True)
        return {
            "daily_accuracy_pct": 0.0,
            "weekly_accuracy_pct": 0.0,
            "monthly_accuracy_pct": 0.0,
            "correct": 0,
            "warning": 0,
            "wrong": 0,
            "pending": 0,
            "last_tune_ts": None,
            "config_name": "error",
            "error": str(e)
        }

```text

### Phase 4: Hunter/Top Movers UI 📊

**Goal**: Wire Cockpit V3 to show real top movers

**Endpoints Needed**:

1. `/api/v3/hunter/feed` - Top stock/crypto movers
2. `/api/v3/predictions/latest` - Recent predictions with outcomes
3. `/api/v3/providers/health` - Provider status


**Implementation Strategy**:

- Reuse existing `/api/movers` endpoint logic
- Format for V3 minimal UI
- Add GPS/confidence scores


### Phase 5: V3 UI Integration 🎨

**Goal**: Complete Cockpit V3 JavaScript wiring

**Files to Update**:

- `static/cockpit_v3.js` - Add data fetching functions
- Update panels to show real data
- Handle error states gracefully


### Phase 6: Data Reliability 📡

**Goal**: Ensure Ghost knows when data is degraded

**Use**: `core/price_reliability.py` and `core/feature_diagnostics.py`

**Implementation**:

- Price fetches return: `{price, provider, stale, last_update}`
- Predictions with stale data: confidence = 0% or skip
- Provider health endpoint shows status


### Phase 7: Testing & Validation ✅

**Local Tests**:

1. Hit `/api/predict/run` with `{"symbol": "AAPL"}`
2. Verify prediction_id returned
3. Query `/api/v3/accuracy/summary` - should show non-zero after enough predictions
4. Load `/cockpit` - verify no JS errors
5. Check panels populate with data


**Railway Tests**:

1. Deploy to Railway
2. Monitor logs for prediction scheduler
3. Verify Telegram alerts show real accuracy
4. Test all V3 endpoints return 200


---

## CRITICAL FINDINGS

1. **Predictions ARE Working**: `/api/predict/run` successfully writes to database
2. **Root Bug Located**: `/api/v3/accuracy/summary` never queries database
3. **Scheduler Status**: Beast scheduler is PRIMARY, others disabled
4. **Database**: `ghost_predictions.db` is authoritative source
5. **UI**: Cockpit V3 exists but needs data wiring


---

## NEXT STEPS

1. ✅ Complete RECON (DONE)
2. 🔄 Find/verify `run_prediction` function for beast_scheduler
3. 🚨 Fix accuracy endpoint (HIGHEST PRIORITY)
4. 📊 Wire hunter/top movers endpoints
5. 🎨 Complete V3 JS integration
6. ✅ Test locally + Railway
7. 📝 Create completion report


---

**STATUS**: Phase 1 Complete → Beginning Phase 2 Surgical Fixes
