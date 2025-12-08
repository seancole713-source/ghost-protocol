# Ghost Hunter V1 - Developer Notes & Test Plan

## Overview

Ghost Hunter V1 extends the single-symbol WOLF prediction system to multi-symbol hunting across stocks and crypto. The
system maintains **execution OFF**and focuses solely on prediction layer enhancements.

---

## Implementation Summary

### Phase 1: Data Structure**Decision**: Keep flat `_LATEST_PREDICTIONS` structure with classification helper

**Internal Structure**:

```python

# wolf_app.py line ~1320

_LATEST_PREDICTIONS: dict[str, dict[str, Any]] = {}

# Structure: Flat dictionary

{
  "WOLF": {
    "prediction_id": "uuid",
    "symbol": "WOLF",
    "run_at": 1763647539.693815,  # Float timestamp
    "confidence": 0.6,             # 0.0-1.0 range
    "direction": "FLAT",           # "UP", "DOWN", "FLAT"
    "horizon_h": 48                # Hours
  },
  "AAPL": {...},
  "WEPE": {...}
}

```text

**Classification Logic**:

- `_classify_symbol_category(symbol)` → Returns "stocks", "crypto", or "vip"
- VIP coins (WEPE, LILPEPE, DORKL, SLOTH, APC) → `"vip"`
- Crypto symbols (BTC, ETH, etc.) → `"crypto"`
- All others → `"stocks"`


---

### Phase 2: Hunter Universe

**Stock Symbols**(wolf_app.py line ~1328):

```python

HUNTER_STOCK_SYMBOLS = ["WOLF", "AAPL", "MSFT", "NVDA"]

```text**Crypto Symbols**(wolf_app.py line ~1330):

```python

HUNTER_CRYPTO_SYMBOLS = ["WEPE", "LILPEPE", "DORKL", "SLOTH", "APC", "BTC"]

```text**Scheduler Integration**:

- `_generate_multi_symbol_predictions()` loops through hunter universe
- Calls `api_predict_run()` for each symbol
- Updates `_LATEST_PREDICTIONS` in-memory store
- Returns summary: `{stocks: N, crypto: N, total: N, errors: []}`


**Schedule**(core/scheduled_predictions.py):

- 8:00 AM ET: Pre-market predictions
- 12:00 PM ET: Mid-day update
- 4:00 PM ET: End-of-day summary
- Runs Mon-Fri only (market days)


---

### Phase 3: Hunter View API**Endpoint**: `GET /api/hunter/snapshot`

**JSON Schema**:

```json

{
  "timestamp": 1763647539,  // Latest prediction run_at (int seconds)
  "stocks": [
    {
      "symbol": "WOLF",
      "direction": "FLAT",
      "confidence": 0.6,
      "horizon_h": 48
    }
  ],
  "crypto": [
    {
      "symbol": "WEPE",
      "direction": "UP",
      "confidence": 0.72,
      "horizon_h": 24
    }
  ]
}

```text

**Design Notes**:

- Compact format (no prediction_id, run_at for UI simplicity)
- Omits symbols with no predictions (not `null` values)
- Reads directly from `_LATEST_PREDICTIONS` (no separate storage)
- No authentication required (public read-only)


---

### Phase 4: Legacy Endpoint Updates

**Modified**: `GET /api/cockpit/snapshot`

**Changes**:

- Now classifies predictions using `_classify_symbol_category()`
- Populates both `predictions.stocks` and `predictions.crypto` arrays
- Updates `timestamp` from latest prediction `run_at`


**Example Response**:

```json

{
  "timestamp": 1763647539,
  "predictions": {
    "stocks": [
      {
        "symbol": "WOLF",
        "prediction_id": 1,
        "run_at": 1763647539,
        "confidence": 60.0,          // Percentage (0-100)
        "direction": "FLAT",
        "horizon_h": 48
      }
    ],
    "crypto": [
      {
        "symbol": "WEPE",
        "prediction_id": 2,
        "run_at": 1763647540,
        "confidence": 72.0,          // Percentage (0-100)
        "direction": "UP",
        "horizon_h": 24
      }
    ]
  }
}

```text

---

## Execution Safety Verification

### ✅ Confirmed NO Changes To

1. **Order Placement**: No modifications to `POST /api/orders/place`
2. **Trade Execution**: No changes to execution engines
3. **SL/TP Monitors**: Stop-loss/take-profit logic untouched
4. **Risk Guard**: Risk management engines unchanged
5. **Auto-Trading Toggles**: No environment variable changes (SIM_MODE, EXECUTION_ENABLED, etc.)


### Files Modified

- `wolf_app.py`: +221 lines, -4 lines
  - Added: `_classify_symbol_category()`, `_generate_multi_symbol_predictions()`, `_send_multi_symbol_telegram_alert()`
  - Added: `GET /api/hunter/snapshot` endpoint
  - Added: `HUNTER_STOCK_SYMBOLS`, `HUNTER_CRYPTO_SYMBOLS` constants
  - Modified: `GET /api/cockpit/snapshot` prediction population logic


**Git Diff Summary**:

```text

 wolf_app.py | 225 +++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 221 insertions(+), 4 deletions(-)

```text

---

## Test Plan

### Manual Testing Sequence

#### Test 1: Single Symbol Prediction (Baseline)

```bash

# Generate WOLF prediction

curl -X POST <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/predict/run>>>>> \
  -H "Content-Type: application/json" \
  -d '{"symbol":"WOLF"}' | jq

# Expected

# {

#   "ok": true

#   "prediction_id": 1

#   "symbol": "WOLF"

#   "confidence": 0.6

#   "direction": "FLAT"

# }

```text

#### Test 2: Debug Store (Verify Write Path)

```bash

# Check in-memory predictions store

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/debug/predictions>>>>> | jq

# Expected

# {

#   "count": 1

#   "keys": ["WOLF"]

#   "store": {

#     "WOLF": {

#       "prediction_id": 1

#       "symbol": "WOLF"

#       "run_at": 1763647539.693815

#       "confidence": 0.6

#       "direction": "FLAT"

#       "horizon_h": 48

#     }

#   }

# }

```text

#### Test 3: Multi-Symbol Predictions

```bash

# Generate predictions for additional symbols

curl -X POST <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/predict/run>>>>> \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL"}' | jq

curl -X POST <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/predict/run>>>>> \
  -H "Content-Type: application/json" \
  -d '{"symbol":"WEPE"}' | jq

curl -X POST <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/predict/run>>>>> \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC"}' | jq

# Verify store now has 4 symbols

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/debug/predictions>>>>> | jq '.count'

# Expected: 4

```text

#### Test 4: Hunter Snapshot (New Endpoint)

```bash

# Test new compact hunter view

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/hunter/snapshot>>>>> | jq

# Expected

# {

#   "timestamp": 1763647540

#   "stocks": [

#     {"symbol": "WOLF", "direction": "FLAT", "confidence": 0.6, "horizon_h": 48}

#     {"symbol": "AAPL", "direction": "UP", "confidence": 0.72, "horizon_h": 48}

#   ]

#   "crypto": [

#     {"symbol": "WEPE", "direction": "UP", "confidence": 0.68, "horizon_h": 24}

#     {"symbol": "BTC", "direction": "DOWN", "confidence": 0.55, "horizon_h": 24}

#   ]

# }

```text

#### Test 5: Legacy Cockpit (Verify Backward Compatibility)

```bash

# Check cockpit predictions field

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/cockpit>>>>> | jq '.predictions'

# Expected: Hash of all symbols

# {

#   "WOLF": {"prediction_id": 1, "run_at": 1763647539.693815, "confidence": 0.6, ...}

#   "AAPL": {...}

#   "WEPE": {...}

#   "BTC": {...}

# }

# Check snapshot predictions classification

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/cockpit/snapshot>>>>> | jq '.predictions'

# Expected: Classified into stocks/crypto

# {

#   "stocks": [

#     {"symbol": "WOLF", "prediction_id": 1, "confidence": 60.0, ...}

#     {"symbol": "AAPL", ...}

#   ]

#   "crypto": [

#     {"symbol": "WEPE", ...}

#     {"symbol": "BTC", ...}

#   ]

# }

```text

#### Test 6: Timestamp Validation

```bash

# Verify snapshot timestamp is non-null and updated

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/cockpit/snapshot>>>>> | jq '.timestamp'

# Expected: Integer timestamp (e.g., 1763647540)

# Should NOT be null

# Verify hunter timestamp matches

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/hunter/snapshot>>>>> | jq '.timestamp'

# Expected: Same timestamp as snapshot (latest prediction)

```text

---

## Validation Checklist

### ✅ Functionality

- [ ] Single symbol predictions work (WOLF)
- [ ] Multi-symbol predictions work (WOLF + AAPL + WEPE + BTC)
- [ ] `_LATEST_PREDICTIONS` store populates correctly
- [ ] `/api/hunter/snapshot` returns classified predictions
- [ ] `/api/cockpit` shows all predictions in `.predictions` field
- [ ] `/api/cockpit/snapshot` classifies into stocks/crypto arrays
- [ ] Timestamp is non-null in all responses


### ✅ Data Integrity

- [ ] Predictions have correct structure (prediction_id, symbol, run_at, confidence, direction, horizon_h)
- [ ] Confidence values are in correct range (0.0-1.0 for hunter, 0-100 for snapshot)
- [ ] Timestamps are integers (Unix seconds)
- [ ] Symbol classification works (WOLF/AAPL → stocks, WEPE/BTC → crypto)


### ✅ Backward Compatibility

- [ ] Existing `/api/cockpit` endpoint still works
- [ ] Existing `/api/cockpit/snapshot` endpoint still works
- [ ] Single WOLF predictions still function as before
- [ ] No breaking changes to response schemas


### ✅ Safety Verification

- [ ] No order placement code modified
- [ ] No trade execution code modified
- [ ] No SL/TP monitor code modified
- [ ] No risk guard code modified
- [ ] No auto-trading toggles changed
- [ ] SIM_MODE environment unchanged
- [ ] Scheduler only generates predictions (no execution)


---

## Next Steps for Production

### After Testing Passes

1. **Monitor Scheduler**: Wait for 8am/12pm/4pm ET scheduled runs
2. **Check Logs**: Verify `_generate_multi_symbol_predictions()` logs success
3. **Telegram Alerts**: Confirm hunter summary messages sent
4. **UI Integration**: Wire `/api/hunter/snapshot` into cockpit UI
5. **Expand Universe**: Add more symbols to `HUNTER_STOCK_SYMBOLS` / `HUNTER_CRYPTO_SYMBOLS`


### Performance Considerations

- Current universe: 4 stocks + 6 crypto = **10 total symbols**- Each prediction takes ~2-5 seconds (depending on price provider latency)
- Full hunter run:**~20-50 seconds**(sequential execution)
- Scheduled 3x/day (8am, 12pm, 4pm) =**~150 seconds total daily runtime**### Scaling Path

- If universe expands beyond 20 symbols, consider:
  - Parallel async prediction generation (use `asyncio.gather()`)
  - Stagger predictions across multiple scheduler windows
  - Cache price data to reduce provider calls
  - Add rate limiting per provider (already exists in quorum logic)


---

## API Documentation Summary

### New Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/hunter/snapshot` | GET | None | Compact multi-symbol prediction view for UI |

### Modified Endpoints

| Endpoint | Changes |
|----------|---------|
| `/api/cockpit/snapshot` | Now populates both `predictions.stocks` and `predictions.crypto` arrays |
| `/api/cockpit` | Continues to expose all predictions in flat `.predictions` hash |

### Internal Functions

| Function | Purpose |
|----------|---------|
| `_classify_symbol_category(symbol)` | Classify symbol into "stocks", "crypto", or "vip" |
| `_generate_multi_symbol_predictions()` | Loop through hunter universe, generate predictions |
| `_send_multi_symbol_telegram_alert()` | Send Telegram summary of latest predictions |

---

## Troubleshooting

### Issue: No predictions in hunter/snapshot**Check**

1. Run manual prediction: `POST /api/predict/run {"symbol":"WOLF"}`
2. Check debug endpoint: `GET /api/debug/predictions`
3. Verify store has data: `jq '.count'` should be > 0


### Issue: Timestamp is null

**Check**:

1. Verify `_LATEST_PREDICTIONS` has data
2. Check if `run_at` field exists in prediction records
3. Ensure predictions were created after hunter wiring (not legacy predictions)


### Issue: Symbols classified incorrectly

**Check**:

1. Verify `_classify_symbol_category()` logic (line ~1332)
2. Check if symbol is in `VIP_COINS`, `CRYPTO_SYMBOLS`, or `HUNTER_CRYPTO_SYMBOLS`
3. Add debug logging to classification function if needed


### Issue: Scheduled predictions not running

**Check**:

1. Verify scheduler started: Check logs for "[MULTI-PREDICTION SCHEDULER] Loop started"
2. Confirm market day (Mon-Fri): Scheduler skips weekends
3. Check time windows: 8:00 AM, 12:00 PM, 4:00 PM ET (±2.5 min trigger window)
4. Verify `_generate_multi_symbol_predictions` wired to scheduler (line ~3895)


---

## Success Criteria

### ✅ Phase 1-4 Complete

- [x] `_LATEST_PREDICTIONS` structure supports multi-symbol
- [x] Classification helper function implemented
- [x] Hunter universe defined (4 stocks, 6 crypto)
- [x] Multi-symbol prediction generator implemented
- [x] Scheduler wired to hunter generator


### ✅ Phase 5: Execution Safety

- [x] No trading/execution code modified
- [x] No order placement changes
- [x] No SL/TP monitor changes
- [x] No risk guard changes
- [x] No auto-trading toggle changes


### ✅ Phase 6: Documentation

- [x] Developer notes complete
- [x] Test plan with curl commands
- [x] API documentation updated
- [x] Troubleshooting guide included


---

## Commit Message

```text

feat: Ghost Hunter V1 - Multi-symbol prediction layer

Extends single-symbol WOLF predictions to hunter universe (4 stocks + 6 crypto).
Adds /api/hunter/snapshot endpoint for compact UI view.
Updates /api/cockpit/snapshot to classify predictions (stocks vs crypto).
Wires scheduler to generate multi-symbol predictions 3x/day (8am, 12pm, 4pm ET).

EXECUTION OFF: No trading, order placement, or risk changes.

Changes:

- Add HUNTER_STOCK_SYMBOLS: ["WOLF", "AAPL", "MSFT", "NVDA"]
- Add HUNTER_CRYPTO_SYMBOLS: ["WEPE", "LILPEPE", "DORKL", "SLOTH", "APC", "BTC"]
- Add _classify_symbol_category() helper (stocks/crypto/vip)
- Add _generate_multi_symbol_predictions() for scheduler
- Add _send_multi_symbol_telegram_alert() for notifications
- Add GET /api/hunter/snapshot (compact view)
- Update GET /api/cockpit/snapshot (classify predictions)
- Keep _LATEST_PREDICTIONS flat structure with classification


Files modified:

- wolf_app.py: +221 lines, -4 lines (predictions only, no execution changes)


Test with:
curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/hunter/snapshot>>>>> | jq

```text

---

**Status**: ✅ Ready for review and commit
**Next Action**: User reviews changes, approves, then commits to Railway
