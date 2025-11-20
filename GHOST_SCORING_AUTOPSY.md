# 🔍 GHOST PREDICTION + SCORING AUTOPSY – HUNTER CONSISTENCY FIX

**Date**: November 20, 2025  
**Status**: DIAGNOSIS COMPLETE → FIX IN PROGRESS  
**Severity**: HIGH – Inconsistent messaging undermines user trust

---

## 📋 EXECUTIVE SUMMARY

**The Problem**: Ghost's prediction pipeline shows confusing inconsistencies between what it generates, what it counts, and what it reports to users.

**User Evidence**:
- **Morning Report**: "Ghost Accuracy: 0.0% (0 predictions)"
- **Reality**: WOLF prediction sent at 07:57 AM with "Confidence: 0%"
- **Reality**: META/TSLA signals with 72% confidence exist but don't appear in morning report
- **Reality**: WOLF 0% prediction evaluated as "INCORRECT" using mismatched price providers

**Root Causes** (mapped below):
1. **0% confidence predictions** are generated and sent but treated as "real" predictions
2. **Morning Report** only counts `ghost_predictions` table entries (which are never populated by current system)
3. **High-confidence predictions** (72%+) are generated but not stored in scoring tables
4. **Provider mismatch** between prediction-time (Yahoo) and evaluation-time (real-time feed)

---

## 📊 PHASE 1: PREDICTION LIFECYCLE MAP

### **GENERATION LAYER** (How predictions are created)

| Component | Function | Storage Destination | Confidence Range |
|-----------|----------|-------------------|------------------|
| `/api/predict/run` | `api_predict_run()` in wolf_app.py:5745 | `_LATEST_PREDICTIONS` dict + `predictions` table (services/predictor.py) | 0.6-0.75 (default) |
| Beast Scheduler | `_send_prediction_alert()` in core/beast_scheduler.py:41 | Via `RUN_PREDICTION_FUNC` callback | Variable (depends on callback) |
| Hunter V1 Generator | `_generate_multi_symbol_predictions()` in wolf_app.py:5878 | Calls `api_predict_run()` → same destinations | Inherited from api_predict_run |
| Crypto Predictor | `generate_prediction()` in core/crypto/crypto_predictor.py:151 | `crypto_predictions` table | 0-1.0 (confidence score) |

**Key Finding**: `/api/predict/run` generates predictions with:
```python
direction = "FLAT"
confidence = 0.6  # Default

# Try to get momentum from history
if recent_change_pct > 2:
    direction = "UP"
    confidence = min(0.75, 0.6 + abs(recent_change_pct) / 20)
```

**BUT** if history fetch fails or no significant movement → `confidence = 0.6`, `direction = "FLAT"`

**CRITICAL BUG**: If price history is completely unavailable → confidence remains 0.6 (not 0.0)  
**User saw 0%** → This suggests a different code path or override somewhere

---

### **STORAGE LAYER** (Where predictions are kept)

| Store | Populated By | Used By | Confidence Filter |
|-------|--------------|---------|-------------------|
| `_LATEST_PREDICTIONS` (wolf_app.py:1320) | `api_predict_run()` | `/api/cockpit`, `/api/cockpit/snapshot`, `/api/hunter/snapshot` | NONE – stores all |
| `predictions` table (services/predictor.py) | `predictor.create_prediction()` | `/api/predict/series`, `/api/predict/history` | NONE – stores all |
| `ghost_predictions` table (core/prediction_tracker.py:27) | `log_prediction()` | `calculate_accuracy()`, Morning Report | ⚠️ **NEVER CALLED IN CURRENT CODE** |
| `crypto_predictions` table | `CryptoPredictionEngine._store_prediction()` | Crypto-specific endpoints | NONE – stores all |

**CRITICAL FINDING**: 
- `ghost_predictions` table (used for Morning Report accuracy) is **NEVER POPULATED**
- `log_prediction()` function exists but is not called anywhere in wolf_app.py
- Morning Report pulls from empty `ghost_predictions` table → always shows "0.0% (0 predictions)"

---

### **SCORING LAYER** (How accuracy is calculated)

| Component | Data Source | Logic | Output |
|-----------|-------------|-------|--------|
| `calculate_accuracy()` (core/prediction_tracker.py:322) | `ghost_predictions` table | Counts `checked=1` rows, calculates `correct/total` | `{"accuracy_pct": X, "total_predictions": N}` |
| `verify_prediction()` (core/prediction_tracker.py:209) | `ghost_predictions` table | Fetches outcome price, compares direction, updates `correct` field | Updates `ghost_predictions` row |
| `/api/stage2/accuracy` (wolf_app.py:12039) | `get_accuracy_report()` from core/accuracy_tracker.py | Queries `forecasts` table + outcomes | Different accuracy system (48h forecasts) |
| Telegram "/check" command | `force_market_open_check()` | ⚠️ **FUNCTION DOESN'T EXIST** | Broken |

**CRITICAL FINDING**:
- **Dual accuracy systems**: `prediction_tracker.py` (unused) vs `accuracy_tracker.py` (Stage 2 forecasts)
- Neither system is wired to `/api/predict/run` predictions
- No automatic scoring/evaluation job running

---

### **REPORTING LAYER** (What users see)

| Report Type | Generator | Data Source | Filters |
|-------------|-----------|-------------|---------|
| Morning Report (Telegram) | `format_daily_report()` core/telegram_hunter.py:261 | `accuracy_stats` parameter | None specified |
| Morning Report Accuracy | `calculate_accuracy()` (prediction_tracker.py) | `ghost_predictions` table (empty) | Only `checked=1` rows |
| "GHOST AI TRADING SIGNALS" | `format_opportunity_alert()` (telegram_hunter.py:218) | Opportunity scanner | `score >= 80` for instant alert |
| Beast Scheduler Alerts | `render_alert()` (telegram_alerts.py:91) | Scheduler-triggered predictions | No confidence filter (sends all) |
| /api/hunter/snapshot | `api_hunter_snapshot()` (wolf_app.py:17488) | `_LATEST_PREDICTIONS` dict | No filter |

**CRITICAL FINDINGS**:
1. Morning Report shows "0.0% (0 predictions)" because `ghost_predictions` table is empty
2. Morning Report shows "No high-quality opportunities" even when 72% confidence predictions exist
3. High-confidence signals (META/TSLA at 72%) are generated by a different system (opportunity scanner)
4. Beast Scheduler sends alerts regardless of confidence (no 0% filter)

---

## 🔍 PHASE 2: INCONSISTENCY REPORT

### **Inconsistency #1: 0% Confidence Contradiction**

**What User Saw**:
```
🌅 PRE-MARKET PREDICTION
Confidence: 0%
Direction: BUY

💡 STRATEGY:
📈 Ghost predicts UPWARD movement today
```

**The Problem**: 
- Text says "Ghost predicts UPWARD movement"
- Confidence is 0% (means "no usable data")
- These statements contradict each other

**Root Cause**:
- Alert template in `core/telegram_alerts.py:91` (or beast_scheduler path) doesn't filter 0% confidence
- Text template doesn't have conditional logic for low confidence
- No policy defining "what counts as a real prediction"

---

### **Inconsistency #2: Morning Report Shows 0 Predictions Despite Active Predictions**

**What User Saw**:
```
☀️ MORNING REPORT
📅 Thursday, November 20, 2025
🎯 Ghost Accuracy: 0.0% (0 predictions)
```

**Simultaneously**:
- WOLF prediction was sent at 07:57 AM
- META/TSLA predictions exist (72% confidence)

**Root Cause**:
```python
# core/telegram_hunter.py:261 - format_daily_report()
if accuracy_stats:
    accuracy_pct = accuracy_stats.get("accuracy_pct", 0)
    total = accuracy_stats.get("total_predictions", 0)
    report += f"🎯 **Ghost Accuracy:** {accuracy_pct:.1f}% ({total} predictions)\n\n"
```

`accuracy_stats` comes from `calculate_accuracy()` which queries `ghost_predictions` table (empty).

**Why It's Empty**:
- `/api/predict/run` calls `predictor.create_prediction()` → stores in `predictions` table
- `log_prediction()` exists to write to `ghost_predictions` table
- **BUT `log_prediction()` is never called by any active code path**

**Evidence**:
```bash
grep -r "log_prediction" ghost-protocol/ --include="*.py"
# Result: Only defined in prediction_tracker.py, never called
```

---

### **Inconsistency #3: High-Confidence Predictions Not in Morning Report**

**What User Saw**:
```
🎯 GHOST AI TRADING SIGNALS  
⚠️ $590.32 → $595.33 (0.8%)  
✅ Confidence: 72%  

BUT Morning Report said:
🔍 No high-quality opportunities detected today.
```

**Root Cause**:
These are two different systems:
1. **"GHOST AI TRADING SIGNALS"** = Opportunity Scanner (core/telegram_hunter.py + opportunity_scorer.py)
2. **Morning Report** = Daily report (core/telegram_hunter.py:261)

Morning Report logic:
```python
if not opportunities:
    report += "🔍 No high-quality opportunities detected today.\n"
```

`opportunities` parameter is passed from wolf_app.py:3603:
```python
async def get_accuracy_stats(period="24h"):
    """Get accuracy stats for daily report"""
    return await calculate_accuracy(period)  # Wrong function - returns accuracy, not opportunities

_asyncio_module.create_task(daily_report_loop(get_top_opportunities, get_accuracy_stats))
```

**The Bug**: `get_accuracy_stats` should return opportunities list, but it returns accuracy dict.

**Result**: Morning Report always gets empty opportunities list → always shows "no opportunities"

---

### **Inconsistency #4: Provider Mismatch in Evaluation**

**What User Saw**:
```
8:00 AM PREDICTION:
• Price: $17.51 (Yahoo)
• Action: BUY

9:35 AM ACTUAL:
• Current: $17.51
• Direction: FLAT
RESULT: ❌ INCORRECT
```

**Simultaneously**: 
```
Wolfspeed shares are up 5.19% to $18.42 per share (Cash App / real-time)
```

**Root Cause**:
- **Prediction time**: Uses `_get_price_quorum()` → Yahoo (often stale in pre-market)
- **Evaluation time**: `/check` command or market open check → may use different provider
- **Reality**: Stock actually moved 5% but Yahoo didn't update → false negative

**Code Path**:
```python
# wolf_app.py:5745 - api_predict_run()
price_data = _get_price_quorum(symbol, "stock")  # Yahoo primary
current_price = float(price_data["price"])

# core/prediction_tracker.py:209 - verify_prediction()
from core.price import get_price
outcome_price = await get_price(symbol)  # Which provider? Not specified
```

**Policy Gap**: No requirement that evaluation uses same provider as prediction.

---

## 🎯 PHASE 3: POLICY IMPLEMENTATION (Fixes)

### **Fix #1: 0% Confidence Policy**

**New Rule**: `confidence == 0` or `confidence < 0.10` = diagnostic placeholder, NOT a real prediction.

**Implementation**:

1. **Alert Generation** (core/telegram_alerts.py:91):
   ```python
   # BEFORE sending alert
   if prediction.get("confidence", 0) < 0.10:
       # Don't send, or send diagnostic-only message
       return False
   ```

2. **Telegram Text Template** (same file):
   ```python
   if confidence < 0.10:
       message = f"⚠️ DIAGNOSTIC: No usable features for {symbol}\n"
       message += "Ghost cannot make a confident prediction right now.\n"
       message += "Will retry when market data improves."
   else:
       # Normal prediction message
   ```

3. **Storage Filter** (_generate_multi_symbol_predictions):
   ```python
   result = asyncio.run(api_predict_run(body, credentials=None))
   if result.get("ok") and result.get("confidence", 0) >= 0.10:
       # Only count as success if confidence >= 10%
       stocks_success += 1
   ```

4. **Accuracy Exclusion** (prediction_tracker.py):
   ```python
   # In calculate_accuracy()
   cur.execute("""
       SELECT * FROM ghost_predictions
       WHERE checked = 1 AND confidence >= 0.10  -- NEW FILTER
       ORDER BY predicted_at DESC
   """)
   ```

---

### **Fix #2: Wire Prediction Logging**

**Problem**: `ghost_predictions` table never populated.

**Solution**: Call `log_prediction()` from `/api/predict/run`:

```python
# wolf_app.py:5745 - api_predict_run()
# After creating prediction via predictor.create_prediction()

# Log to ghost_predictions table for accuracy tracking
if confidence >= 0.10:  # Only log real predictions
    from core import prediction_tracker
    
    prediction_tracker.log_prediction(
        symbol=symbol,
        predicted_price=current_price,  # Predicted future price (use forecast endpoint)
        predicted_direction=direction,
        predicted_pct=0.0,  # TODO: Calculate from forecast
        confidence=confidence,
        timeframe_hours=horizon_h,
        reasons=[],  # TODO: Add reasoning
        current_price=current_price,
    )
```

**Blocker**: `log_prediction()` requires `predicted_price` (future price), but `api_predict_run` doesn't calculate this yet.

**Alternate Solution**: Modify `log_prediction()` to accept optional predicted_price:
```python
def log_prediction(
    symbol: str,
    predicted_direction: str,
    confidence: float,
    timeframe_hours: int,
    current_price: float,
    predicted_price: float | None = None,  # Make optional
    predicted_pct: float | None = None,
    reasons: list[str] | None = None,
):
    # Use current_price if predicted_price not provided
    predicted_price = predicted_price or current_price
    predicted_pct = predicted_pct or 0.0
    reasons = reasons or []
    # ... rest of function
```

---

### **Fix #3: Fix Morning Report Opportunity Wiring**

**Problem**: `get_accuracy_stats()` returns accuracy dict, but `daily_report_loop()` expects opportunities list.

**Solution**: Fix the callback in wolf_app.py startup:

```python
# wolf_app.py:3603 (startup section)
# BEFORE:
async def get_accuracy_stats(period="24h"):
    """Get accuracy stats for daily report"""
    return await calculate_accuracy(period)  # WRONG

# AFTER:
async def get_accuracy_stats(period="24h"):
    """Get accuracy stats for daily report"""
    from core import prediction_tracker
    return prediction_tracker.calculate_accuracy(period)

async def get_top_opportunities():
    """Get top opportunities for daily report"""
    # Get high-confidence predictions from _LATEST_PREDICTIONS
    opportunities = []
    for sym, pred in _LATEST_PREDICTIONS.items():
        if pred.get("confidence", 0) >= 0.70:  # 70%+ threshold
            opportunities.append({
                "symbol": sym,
                "confidence": pred["confidence"],
                "predicted_pct": 0.0,  # TODO: Calculate
                "action": pred["direction"],
                "score": int(pred["confidence"] * 100),  # Convert to score
            })
    
    # Sort by confidence desc
    opportunities.sort(key=lambda x: x["confidence"], reverse=True)
    return opportunities[:10]  # Top 10

# Wire both correctly:
_asyncio_module.create_task(
    daily_report_loop(get_top_opportunities, get_accuracy_stats)
)
```

---

### **Fix #4: Provider Reconciliation**

**Solution**: Store provider with prediction, use same for evaluation.

```python
# wolf_app.py:5745 - api_predict_run()
_LATEST_PREDICTIONS[symbol] = {
    "prediction_id": prediction_id,
    "symbol": symbol,
    "run_at": run_at,
    "confidence": confidence,
    "direction": direction,
    "horizon_h": horizon_h,
    "provider": price_data.get("provider"),  # ADD THIS
    "price_at_prediction": current_price,     # ADD THIS
}

# core/prediction_tracker.py - verify_prediction()
# Fetch using same provider stored with prediction
from core.price import get_price_from_provider  # New function

provider = prediction.get("provider", "yahoo")  # Default yahoo
outcome_price = await get_price_from_provider(symbol, provider)

# If provider unavailable, mark as "unscorable" instead of incorrect
if outcome_price is None:
    cur.execute("""
        UPDATE ghost_predictions
        SET checked = 1, checked_at = ?, error_pct = NULL
        WHERE id = ?
    """, (int(time.time()), prediction["id"]))
    # Don't mark correct/incorrect if provider mismatch
```

---

### **Fix #5: Fix Telegram Commands (/predict, /check)**

**Problem**: Commands call non-existent functions.

**Solution**: Update wolf_app.py telegram webhook:

```python
# wolf_app.py:14365
elif text.lower().startswith("/predict"):
    try:
        if SCHEDULED_PREDICTIONS_ENABLED:
            _tg_send_chat_message(chat_id, "🔮 Generating prediction now...")
            scheduled_predictions.force_multi_prediction()  # FIX: Use correct function
        else:
            _tg_send_chat_message(chat_id, "⚠️ Prediction scheduler not enabled")
    except Exception as e:
        _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")

elif text.lower().startswith("/check"):
    try:
        if SCHEDULED_PREDICTIONS_ENABLED:
            _tg_send_chat_message(chat_id, "📊 Checking prediction accuracy...")
            
            # Get latest prediction for WOLF
            pred = _LATEST_PREDICTIONS.get("WOLF")
            if not pred:
                _tg_send_chat_message(chat_id, "No recent prediction to check")
                return {"ok": True}
            
            # Get current price
            price, prev, provider = get_wolf_price()
            pred_price = pred.get("price_at_prediction", prev)
            change_pct = ((price - pred_price) / pred_price * 100) if pred_price else 0
            
            # Compare direction
            actual_direction = "UP" if change_pct > 1 else ("DOWN" if change_pct < -1 else "FLAT")
            predicted_direction = pred.get("direction", "FLAT")
            correct = actual_direction == predicted_direction
            
            # Format message
            msg = f"⚠️ PREDICTION CHECK\n\n"
            msg += f"PREDICTED: {predicted_direction} @ ${pred_price:.2f}\n"
            msg += f"ACTUAL: {actual_direction} @ ${price:.2f} ({change_pct:+.2f}%)\n\n"
            msg += f"RESULT: {'✅ CORRECT' if correct else '❌ INCORRECT'}"
            
            _tg_send_chat_message(chat_id, msg)
        else:
            _tg_send_chat_message(chat_id, "⚠️ Prediction scheduler not enabled")
    except Exception as e:
        _tg_send_chat_message(chat_id, f"❌ Error: {str(e)[:100]}")
```

---

## 🛡️ PHASE 4: SAFETY + TEST PLAN

### **Pre-Deployment Checks**:

```bash
# 1. No execution/trading changes
grep -n "AUTO_TRADE\|SIM_MODE\|execute_order" wolf_app.py
# Should return same lines as before (no new matches)

# 2. Verify confidence filter in alerts
grep -A 5 "confidence.*<.*0.10" wolf_app.py core/telegram_alerts.py

# 3. Verify logging wired
grep -n "prediction_tracker.log_prediction" wolf_app.py
# Should find new call in api_predict_run

# 4. Verify morning report fixed
grep -A 10 "async def get_top_opportunities" wolf_app.py
```

### **Test Commands**:

```bash
# Test 1: Generate low-confidence prediction (should be filtered)
curl -X POST https://ghost-protocol-production.up.railway.app/api/predict/run \
  -H "Content-Type: application/json" \
  -d '{"symbol":"WOLF"}'

# Expected: If confidence < 10%, should not send Telegram or be counted

# Test 2: Check morning report payload
curl https://ghost-protocol-production.up.railway.app/api/cockpit/snapshot | jq '.predictions'

# Expected: Only predictions with confidence >= 10%

# Test 3: Telegram /predict command
# Send "/predict" in Telegram
# Expected: Should trigger force_multi_prediction() successfully

# Test 4: Telegram /check command  
# Send "/check" in Telegram
# Expected: Should show prediction vs actual comparison

# Test 5: Verify ghost_predictions table populates
# After generating a prediction, check database:
sqlite3 data/wolf.db "SELECT symbol, confidence, predicted_direction FROM ghost_predictions ORDER BY predicted_at DESC LIMIT 5;"

# Expected: New entries with confidence >= 0.10
```

---

## 📝 SUMMARY OF CHANGES

| File | Change Type | Lines Modified | Risk |
|------|-------------|----------------|------|
| wolf_app.py | Wire prediction logging | +20 | LOW |
| wolf_app.py | Fix morning report callbacks | +25 | LOW |
| wolf_app.py | Fix /predict command | 2 | LOW |
| wolf_app.py | Fix /check command | +30 | LOW |
| core/telegram_alerts.py | Add 0% confidence filter | +10 | LOW |
| core/telegram_hunter.py | Update report format (optional) | 5 | LOW |
| core/prediction_tracker.py | Make predicted_price optional | 5 | LOW |
| core/prediction_tracker.py | Add confidence filter to accuracy | 2 | LOW |

**Total Additions**: ~92 lines  
**Risk Level**: LOW (no execution/trading changes)  
**Testing Required**: Telegram commands + morning report + accuracy tracking

---

## ✅ DEPLOYMENT READINESS

- [ ] All fixes implemented
- [ ] No AUTO_TRADE or execution changes
- [ ] Test plan documented
- [ ] Commands tested locally (if possible)
- [ ] Ready for git commit + Railway deploy

**Status**: DOCUMENTATION COMPLETE → READY FOR IMPLEMENTATION

