# GHOST Pass/Fail Evaluation Table

**Date**: October 4, 2025\
**Auditor**: GitHub Copilot\
**Evaluation Criteria**: Master Directive compliance axes

______________________________________________________________________

## Evaluation Summary

| Criterion | Status | Score | Notes | |-----------|--------|-------|-------| |
**Live-only data** | ⚠️ **PARTIAL PASS** | 70/100 | Quorum logic exists, but backoff
sticky + no degraded mode | | **Correct math** | ✅ **PASS** | 95/100 | PnL, NAV,
forecast accuracy all correct; minor rounding edge cases | | **Persistence** | ⚠️
**PARTIAL PASS** | 80/100 | Portfolio persists, but $0 risk exists if mode=none (default
for some setups) | | **Prediction-vs-Reality** | ⚠️ **PARTIAL PASS** | 75/100 | MAP/RMSE
computed but not exposed to UI; no historical comparison API | | **Runtime config** | ✅
**PASS** | 85/100 | 100+ env vars, runtime `/api/runtime/config` endpoint, BUT lacks
centralized docs | | **Transparency** | ⚠️ **PARTIAL PASS** | 70/100 | Logs/metrics
extensive, but no user-facing "why" for anomaly/pause states | | **Zero randomness** | ✅
**PASS** | 100/100 | No RNG in production paths; all variance is deterministic
(volatility estimates) |

**Overall Pass Rate**: **82/100** (B grade)\
**Production Readiness**: ⚠️ **Conditional** (requires P0+P1 remediation)

______________________________________________________________________

## Detailed Evaluation

### 1. Live-only data

**Requirement**: "No mock/placeholder/simulation data in production paths. All prices
from real APIs."

| Check | Status | Evidence | |-------|--------|----------| | No hardcoded mock prices |
✅ PASS | Grep for `mock`, `placeholder`, `simulation` → zero production hits | |
Yahoo/Polygon/AlphaVantage integration | ✅ PASS | Lines 2580-2750:
`_get_live_price_quorum()` calls real APIs | | Circuit breaker prevents infinite retries
| ✅ PASS | Lines 2530-2580: Exponential backoff implemented | | **ISSUE**: Backoff never
resets on success | ❌ FAIL | **GH-AUD-005**: `backoff_factor` sticky → permanent
degradation | | **ISSUE**: Reuters DNS fail → empty news | ❌ FAIL | **GH-AUD-006**: No
degraded mode → blank UI | | Quorum logic (3 providers → pick median) | ✅ PASS | Line
2680: `PRICE_MAX_DEVIATION` enforces consensus | | Prev_close fallback when all
providers fail | ✅ PASS | Line 2730: Falls back to cached `prev_close` |

**Score Rationale**: Core data pipeline is real, but resilience gaps (sticky backoff, no
degraded mode) → 70/100

______________________________________________________________________

### 2. Correct math

**Requirement**: "PnL, NAV, position calculations must be algebraically correct and
match brokerage statements."

| Check | Status | Evidence | |-------|--------|----------| | Position value = qty ×
price | ✅ PASS | Line 6850: `val = pos["quantity"] * price` | | Unrealized P&L = (price
\- avg_cost) × qty | ✅ PASS | Line 6855:
`pnl = (price - pos["avg_cost"]) * pos["quantity"]` | | Total NAV = cash + sum(position
values) | ✅ PASS | Line 6880: `nav = STATE["cash"] + sum(vals)` | | Weighted avg cost on
add | ✅ PASS | Line 6620:
`new_avg = ((old_qty * old_avg) + (add_qty * add_price)) / (old_qty + add_qty)` | |
Forecast MAP/RMSE formulas | ✅ PASS | Line 791: `map = sum(apes)/len(apes)`, Line 792:
`rmse = sqrt(sum(e²)/n)` | | **Minor**: Floating-point precision edge cases | ⚠️ NOTE |
Python float (64-bit) → ~15 decimal digits; acceptable for $USD |

**Score Rationale**: All core financial math correct; no rounding errors observed in
tests → 95/100

______________________________________________________________________

### 3. Persistence

**Requirement**: "Portfolio shows correct balances on restart; no $0 at boot if state
exists."

| Check | Status | Evidence | |-------|--------|----------| | Persistence mode
configurable | ✅ PASS | Line 1159: `WOLF_PERSIST_MODE` (none/file/redis/sqlite/auto) | |
Startup calls `_persist_load()` | ✅ PASS | Line 1377: Loads state on boot | | Autosave
thread available | ✅ PASS | Line 3550: `_autosave_loop()` if `WOLF_AUTOSAVE_S > 0` | |
Manual `/control/save` endpoint | ✅ PASS | Line 3582: Explicit save trigger | |
**ISSUE**: Default mode is `none` | ⚠️ FAIL | Line 1159:
`os.getenv("WOLF_PERSIST_MODE", "none")` → **no persistence by default** | | **ISSUE**:
Portfolio layer not always enabled | ⚠️ FAIL | Line 53: `PORTFOLIO_PERSISTENCE_ENABLED`
depends on import success | | Graceful fallback hierarchy (redis→sqlite→file) | ✅ PASS |
Line 3346: `mode=="auto"` tries all 3 in sequence |

**Score Rationale**: Persistence works when enabled, but **defaults to off** → user must
explicitly configure → 80/100

**Fix**: Change default to `"auto"` or `"sqlite"` (not `"none"`)

______________________________________________________________________

### 4. Prediction-vs-Reality

**Requirement**: "Forecast overlay must show Ghost predictions vs. actual prices, with
MAP/RMSE accuracy chips."

| Check | Status | Evidence | |-------|--------|----------| | Two-line overlay function
exists | ✅ PASS | Line 801: `_build_two_line_forecast()` computes Ghost vs Live | |
Accuracy metrics computed | ✅ PASS | Line 755: `_compute_forecast_accuracy()` returns
MAP/RMSE/bias | | **ISSUE**: Metrics not in `/api/cockpit` response | ❌ FAIL |
**GH-AUD-010**: Accuracy object not included in snapshot | | **ISSUE**: No historical
comparison API | ❌ FAIL | No `/api/forecast/history` endpoint to compare past forecasts
vs actuals | | Forecast pause on anomaly | ✅ PASS | Line 431:
`FORECAST_PAUSE_ON_ANOMALY=1` prevents bad predictions | | UI contract mentions "two
lines" | ⚠️ UNKNOWN | Need to check `UI_BASELINE_CONTRACT.md` |

**Score Rationale**: Backend computes accuracy but doesn't expose it → UI cannot display
chips → 75/100

**Fix**: Add `"forecast_accuracy": {"map": X, "rmse": Y}` to `/api/cockpit` response

______________________________________________________________________

### 5. Runtime config

**Requirement**: "All behavior configurable via env vars or runtime API; no hardcoded
magic numbers."

| Check | Status | Evidence | |-------|--------|----------| | 100+ environment variables
| ✅ PASS | Audit identified 100+ vars across 10 categories | | Runtime config endpoint
exists | ✅ PASS | Line 5430: `POST /api/runtime/config` updates toggles | | Feature
flags for AI, alerts, overlay | ✅ PASS | `AI_ON`, `OVERLAY_ENABLED`, `ALERT_MODE` all
configurable | | No hardcoded credentials | ✅ PASS | All secrets from env (after P0 fix)
| | **ISSUE**: No centralized env var docs | ❌ FAIL | **GH-AUD-009**: 100+ vars
undocumented → developers grep source | | `/api/config` exposes current settings | ✅
PASS | Line 4329: Returns full config snapshot (sanitized) |

**Score Rationale**: Excellent configurability, but documentation gap → 85/100

**Fix**: Create `ENV_VARS_REFERENCE.md` with all 100+ vars cataloged

______________________________________________________________________

### 6. Transparency

**Requirement**: "Logs/metrics expose decision rationale; users understand why Ghost
took action."

| Check | Status | Evidence | |-------|--------|----------| | Structured JSON logging |
✅ PASS | Line 190: `LOG_JSON=1` mode available | | Prometheus metrics (50+ series) | ✅
PASS | Lines 1840-1980: Counters, gauges, histograms | | `/diagnostics/summary` endpoint
| ✅ PASS | Line 4150: System health + recent events | | `/logs/recent` endpoint | ✅ PASS
| Line 4300: Last N log entries | | **ISSUE**: No user-facing "why" for anomaly pause |
⚠️ FAIL | Forecast pauses but UI doesn't explain reason | | **ISSUE**: No alert delivery
history | ⚠️ FAIL | Telegram sends alerts but no audit trail in UI | | AI decision
reasoning stored | ✅ PASS | Line 4839: AIMemory stores full reasoning text |

**Score Rationale**: Excellent observability for ops, but user-facing transparency gaps
→ 70/100

**Fix**: Add `"pause_reason": "anomaly_detected"` to forecast object,
`/api/alerts/history` endpoint

______________________________________________________________________

### 7. Zero randomness

**Requirement**: "No RNG in production; all variance from deterministic volatility/drift
models."

| Check | Status | Evidence | |-------|--------|----------| | No `random.random()` in
price logic | ✅ PASS | Grep: `random` only in test fixtures | | Forecast uses
deterministic cone | ✅ PASS | Line 507: `drift_daily = 0.3*chg_pct + 0.01*news`
(deterministic) | | No Monte Carlo in production | ✅ PASS | Grep: `monte`, `carlo` →
zero hits | | Volatility from historical data | ✅ PASS | Line 476: `PRED_SIGMA_DAILY` is
config constant, not random | | Circuit breaker backoff is deterministic | ✅ PASS | Line
2570: `backoff = BACKOFF_S * (2^bf)` (no jitter yet, but deterministic) |

**Score Rationale**: Perfect compliance; all "randomness" is configurable variance →
100/100

______________________________________________________________________

## Recommendations

### To Achieve 100% Pass Rate

1. **Live-only data → 100%**: Fix GH-AUD-005 (backoff reset) + GH-AUD-006 (Reuters
   degraded mode)
2. **Persistence → 95%**: Change default `WOLF_PERSIST_MODE` from `"none"` to `"auto"`
3. **Prediction-vs-Reality → 95%**: Add accuracy metrics to `/api/cockpit`, create
   `/api/forecast/history`
4. **Runtime config → 95%**: Generate `ENV_VARS_REFERENCE.md`
5. **Transparency → 90%**: Add pause_reason field, alert history endpoint

### Blockers for Production

- **P0**: Secrets rotation (GH-AUD-001)
- **P1**: Backoff reset (GH-AUD-005), Reuters degraded mode (GH-AUD-006)
- **P1**: SSE client tracking (GH-AUD-004) to prevent memory leaks

**Estimated Effort to 100% Pass**: 8-10 developer-days

______________________________________________________________________

## Testing Evidence

| Criterion | Test Coverage | Gap | |-----------|---------------|-----| | Live-only data
| `tests/test_price_providers.py` | No backoff reset test | | Correct math |
`tests/test_pnl_math.py`, `tests/test_forecast_accuracy.py` | Good coverage | |
Persistence | `tests/test_state_persistence.py`, `test_portfolio_persistence.py` |
Missing default=none test | | Prediction-vs-Reality | `tests/test_forecast_overlay.py` |
Exists but not checked | | Runtime config | `tests/test_api_config.py` | Good coverage |
| Transparency | Manual only (curl /logs/recent) | No automated test | | Zero randomness
| Implicit (no RNG imported) | Could add explicit assertion |

______________________________________________________________________

**Pass/Fail Table Generated By**: GitHub Copilot\
**Based On**: Master Directive evaluation criteria + repo audit findings\
**Next Steps**: Review UPGRADE_PLAN.md for remediation sequence
