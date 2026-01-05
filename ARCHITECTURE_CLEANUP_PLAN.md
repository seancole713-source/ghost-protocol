# Ghost Protocol Architecture Cleanup Plan

**Date:** January 5, 2026  
**Status:** Active Refactoring

## Current Problems

### 1. Monolithic `wolf_app.py` (37,000+ lines)
- Everything in one file - API routes, business logic, schedulers, configs
- Impossible to navigate or maintain
- Import cycles force awkward code placement

### 2. Dual Database Systems (Not Coordinated)
| System | Storage | Tables | Reconciler Access |
|--------|---------|--------|-------------------|
| Stock Predictions | PostgreSQL | `predictions`, `prediction_points` | ✅ Works |
| Crypto Predictions | SQLite `ai_memory.db` | `crypto_predictions`, `crypto_forecast_points` | ❌ Was broken |
| FeedbackLoop | SQLite `feedback_loop.db` | `outcomes`, `signal_performance` | N/A |

**Fixed:** Crypto predictions now dual-write to PostgreSQL (commit `92c4c07`)

### 3. Duplicate Reconcilers
| File | Status | Purpose |
|------|--------|---------|
| `services/outcome_reconciler.py` | **DISABLED** | Uses prediction_points table (broken) |
| `services/outcome_reconciler_v2.py` | Active | Uses price_at_prediction directly |

### 4. Duplicate Direction Logic
- **Stocks:** RSI mean-reversion in `wolf_app.py` (RSI > 70 = DOWN)
- **Crypto:** Trend-following in `crypto_predictor.py` (momentum = direction)
- Inconsistent approaches lead to confusion

### 5. Multiple Feature Calculation Systems
- `core/features/technical_indicators.py` - 50+ indicators
- Inline calculations in `wolf_app.py`
- Inline calculations in `crypto_predictor.py`

---

## Immediate Fixes Completed (Jan 5, 2026)

### ✅ Fix 1: Crypto predictions write to PostgreSQL
```python
# crypto_predictor.py now also saves to PostgreSQL
pg_prediction_id = store.save_prediction(
    symbol=symbol,
    forecast_points=forecast_tuples,
    method="ghost-crypto-v1",
    features={"current_price": current_price, ...},
    ...
)
```

### ✅ Fix 2: Disabled broken old reconciler
```python
# wolf_app.py line 4054
# Old reconciler DISABLED - using V2 at line 3973
```

### ✅ Fix 3: FeedbackLoop integrated into crypto predictor
```python
# crypto_predictor.py lines 140-190
# Now applies accuracy penalties for consistently wrong symbols
```

---

## Phase 1: Quick Wins (Week 1)

### 1.1 Consolidate Prediction Storage
- [ ] All predictions go through `PredictionStore` abstraction
- [ ] Remove direct SQLite writes in `crypto_predictor.py` 
- [ ] Single source of truth in PostgreSQL

### 1.2 Delete Dead Code
```bash
# Files to review for deletion:
services/outcome_reconciler.py  # Replaced by v2
wolf_app.py.backup*             # Backup files
```

### 1.3 Extract API Routes
```python
# Move from wolf_app.py to:
api/crypto_endpoints.py      # /api/v3/crypto/*
api/predictions_endpoints.py # /api/v3/predictions/*
api/accuracy_endpoints.py    # /api/v3/accuracy/*
```

---

## Phase 2: Unify Prediction Logic (Week 2-3)

### 2.1 Create Unified Predictor
```python
# core/unified_predictor.py
class UnifiedPredictor:
    """Single entry point for all prediction generation"""
    
    def generate_prediction(self, symbol: str, asset_type: str) -> Prediction:
        if asset_type == "crypto":
            return self._crypto_prediction(symbol)
        else:
            return self._stock_prediction(symbol)
    
    def _apply_feedback_adjustments(self, metrics: dict, symbol: str):
        """Common feedback loop integration"""
        feedback = get_feedback_loop()
        return feedback.get_adjusted_features(metrics)
```

### 2.2 Standardize Direction Logic
```python
# core/direction_analyzer.py
class DirectionAnalyzer:
    """Consistent direction determination"""
    
    def analyze(self, indicators: dict, asset_type: str) -> Tuple[str, float]:
        # Use trend-following for strong trends
        if abs(indicators['trend_strength']) > 0.5:
            return self._trend_following(indicators)
        
        # Use mean-reversion for ranging markets
        return self._mean_reversion(indicators)
```

---

## Phase 3: Split `wolf_app.py` (Week 4+)

### Target Structure
```
ghost-protocol/
├── app.py                    # FastAPI app setup, middleware
├── api/
│   ├── __init__.py
│   ├── crypto_endpoints.py   # /api/v3/crypto/*
│   ├── stock_endpoints.py    # /api/v3/stocks/*
│   ├── predictions.py        # /api/v3/predictions/*
│   ├── accuracy.py           # /api/v3/accuracy/*
│   ├── watchlist.py          # /api/v3/watchlist/*
│   └── admin.py              # /api/v3/admin/*
├── core/
│   ├── unified_predictor.py  # Single prediction entry point
│   ├── direction_analyzer.py # Consistent direction logic
│   ├── feedback_loop.py      # ✅ Already exists
│   └── learning_loop.py      # ✅ Already exists
├── services/
│   ├── outcome_reconciler_v2.py  # ✅ Active reconciler
│   ├── price_providers.py        # Consolidated price fetching
│   └── scheduler.py              # Background tasks
└── wolf_app.py              # Shrink to ~2000 lines (routes + glue)
```

---

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| `wolf_app.py` lines | 37,195 | < 3,000 |
| Prediction storage systems | 2 (SQLite + Postgres) | 1 (Postgres) |
| Active reconcilers | 1 (v2) | 1 |
| BTC prediction accuracy | 21.4% | > 55% |

---

## Risk Mitigation

1. **Feature flags** - Each refactor behind a flag
2. **Dual writes** - Maintain backward compatibility during migration
3. **Integration tests** - Add tests before major changes
4. **Gradual rollout** - One module at a time

---

## Commands for Quick Checks

```bash
# Check wolf_app.py line count
wc -l wolf_app.py

# Find all prediction-related files
find . -name "*.py" -exec grep -l "create_prediction\|save_prediction" {} \;

# Check which reconciler is active
grep -n "reconcile_outcomes" wolf_app.py

# Verify crypto dual-write
grep -n "store.save_prediction" core/crypto/crypto_predictor.py
```
