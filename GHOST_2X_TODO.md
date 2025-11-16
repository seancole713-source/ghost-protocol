# Ghost 2.x Completion Checklist

**Status**: 🚧 In Progress  
**Last Updated**: 2025-11-16  
**Agent**: Ghost 2.x Completion Agent

---

## 🎯 Top-Level Summary

### Backend Core Status
- **Endpoints**: ⚠️ Partially Complete
- **Prediction Engine**: ❌ Blocked by Environment
- **Health Counters**: ✅ Working

### Cockpit UI Status
- **Auth Wiring**: ✅ Complete
- **Ghost 2.x Panel**: ✅ Complete (Basic)
- **Data Display**: ❌ All panels show NO_DATA/empty

### Provider/Environment Status
- **API Keys**: ❌ User-only (Missing in Railway)
- **Configuration**: ⚠️ Needs Verification

---

## 📋 Backend & Predict Engine

### Endpoints
- [x] `/api/status` - returns mode, active, version
- [x] `/ui/health` - simple JSON health check
- [x] `/api/health/predictions` - returns Ghost 2.x health data
- [x] `/api/cockpit` - public snapshot with ghost_2x block
- [x] `/api/predictions/multi/run` - generates multi-symbol predictions
- [x] All public endpoints in `public_paths` list

### Prediction Engine
- [x] `_generate_multi_symbol_predictions()` updates health counters
- [x] Health counters (`_LAST_MULTI_PREDICTION_TIME`, `_LAST_MULTI_PREDICTION_COUNTS`) wired correctly
- [ ] **BLOCKER**: Multi-run returns 0 predictions in production
  - **Root Cause**: Price provider failures (all 21 symbols fail to get prices)
  - **Issue**: Missing API keys or provider configuration in Railway
  - **Symbols Affected**: All stocks (WOLF, AAPL, etc.), all crypto (BTC, ETH, etc.), VIP coins (WEPE, DORKL)

### Price Provider Dependencies
- [ ] **POLYGON_API_KEY** (User-only) - Required for stock prices including WOLF
- [ ] **ALPHAVANTAGE_API_KEY** (User-only) - Backup for stock prices
- [ ] **Crypto providers** - CoinGecko, Binance, Coinbase for crypto/VIP prices
- [ ] Verify `PRICE_STRICT_LIVE`, `PREDICT_REQUIRE_PRICE_QUORUM` settings

---

## 🎨 Cockpit UI & Ghost Score

### Auth & Token Injection
- [x] `ghostAuthHeaders()` helper in `static/js/predict.js`
- [x] All `/api/predict/*` endpoints use auth headers
- [x] `window.GHOST_API_TOKEN` injected in cockpit.html

### Ghost 2.x Health Panel
- [x] HTML structure with IDs: `ghost2-score`, `ghost2-grade`, `ghost2-status`, `ghost2-vip-summary`
- [x] `renderGhostScore()` function implemented
- [x] `renderVipHealth()` function implemented
- [x] `refreshGhost2xPanel()` wired to DOMContentLoaded
- [x] Auto-refresh every 30 seconds via setInterval

### Current UI State Issues (From User Report)
- [ ] **CRITICAL**: Cockpit shows "DELISTED MODE PROVIDER UNAUTHORIZED"
- [ ] **CRITICAL**: All panels show `—` or NO_DATA instead of live data
- [ ] Ghost-AI v1/v2 Monitor: empty (no decisions, tool success)
- [ ] World Context & Market Mood: empty (SPY, VIX, news all missing)
- [ ] Daily Accuracy Ledger: 0/0/0/0 with empty table
- [ ] Market Regime & Risk: all fields empty
- [ ] Portfolio Optimization: "No allocation calculated yet"
- [ ] Smart Execution: shows "TRADING ACTIVE" but all metrics are `—`
- [ ] Personal Portfolio: empty table, no positions
- [ ] Watchlist: empty textareas, not wired to backend
- [ ] Diagnostics: claims "last 50 events" but shows none

### Missing Baseline UI Modules
- [ ] **Goals Panel**: daily/weekly/monthly/yearly targets with % progress
- [ ] **Ghost Score (GPS)**: live 0-10 score in real-time
- [ ] **VIP Coins Panel**: WEPE, LILPEPE, DORKL, SLOTH, APC with prices/status
- [ ] **XRP Tracker**: "bullish eye" indicator with live signals
- [ ] **Presale/Strike Prep**: LILPEPE-style sniper feed

---

## 🌍 Environment & Providers

### Required Environment Variables (User-only)

#### Stock Price Providers
- [ ] `POLYGON_API_KEY` - Primary stock price provider (User-only)
- [ ] `ALPHAVANTAGE_API_KEY` - Backup stock provider (User-only)
- [ ] `STOCK_PRICE_SOURCE` - Should be "polygon" (default)
- [ ] `PRICE_MIN_PROVIDERS` - Minimum providers for quorum (default: 1)

#### Crypto Price Providers
- [ ] `CRYPTO_PRICE_SOURCE` - Default: "coingecko"
- [ ] `CRYPTO_QUORUM` - Quorum size for crypto prices
- [ ] Verify CoinGecko, Binance, Coinbase APIs accessible

#### Price Quorum Settings
- [ ] `PRICE_REQUIRE_QUORUM` - Default: 0 (off)
- [ ] `PREDICT_REQUIRE_PRICE_QUORUM` - Default: 0 (off)
- [ ] `PRICE_STRICT_LIVE` - Default: 0 (allow cached)
- [ ] `PRICE_YAHOO_FIRST` - Yahoo Finance priority

#### Mode & Safety
- [ ] `SIM_MODE=0` - Must be 0 for production (live only)
- [ ] Verify no simulation/placeholder paths active

### Provider Behavior Matrix

| Symbol Type | Provider | API Key Required | Fallback |
|------------|----------|------------------|----------|
| WOLF (stock) | Polygon/AlphaVantage | ✅ Yes | Previous close |
| Other Stocks | Polygon/AlphaVantage + Yahoo | ✅ Yes | Price quorum |
| Crypto (BTC, ETH) | CoinGecko/Binance/Coinbase | ⚠️ Maybe | Price quorum |
| VIP (WEPE, DORKL) | VIP Provider | ⚠️ Maybe | Explicit NO_DATA |
| VIP (LILPEPE, SLOTH, APC) | VIP Provider | N/A | Returns NO_DATA |

---

## 🔔 Alerts & Agents

### Scheduled Jobs
- [ ] Verify `core/scheduled_predictions.py` runs multi-symbol predictions on schedule
- [ ] Confirm Telegram alerts wire to updated health state
- [ ] Check alert timing: 8 AM, 12 PM, 4 PM ET

### Agent Configuration
- [ ] `AGENTS_ENABLED` - Agent system toggle
- [ ] `AGENTKIT_ENABLED` - AgentKit integration
- [ ] `AGENT_ROLE` - Agent behavior mode
- [ ] Verify agents use Ghost 2.x endpoints (not legacy routes)

---

## 🧪 Tests & Verification

### Test Coverage
- [x] `tests/test_cockpit_endpoint.py` - 4/4 passing
  - ✅ test_cockpit_endpoint_200
  - ✅ test_cockpit_system_fields
  - ✅ test_cockpit_ghost2x_fields
  - ✅ test_cockpit_no_auth_required

- [x] `tests/test_multi_predictions.py` - 2/2 passing
  - ✅ test_multi_predictions_endpoint_ok_field_present
  - ✅ test_multi_predictions_endpoint_shape_basic

### Missing Test Coverage
- [ ] `/api/status` endpoint test
- [ ] `/ui/health` endpoint test
- [ ] `/api/health/predictions` comprehensive test
- [ ] Auth-protected `/api/predict/*` endpoints with Bearer token
- [ ] Multi-run with mocked price providers (to verify logic without API keys)

### Production Verification Checklist (User-only)
- [ ] `curl $PROD_BASE/api/status` → 200 with mode/active/version
- [ ] `curl $PROD_BASE/ui/health` → 200 with status ok
- [ ] `curl $PROD_BASE/api/health/predictions` → 200 with ghost_score_v2
- [ ] `curl $PROD_BASE/api/cockpit` → 200 with ghost_2x block
- [ ] `curl $PROD_BASE/api/predictions/multi/run` → total > 0 (not 0)
- [ ] Cockpit UI loads without 401 errors
- [ ] Ghost 2.x panel shows numeric score (not NO_DATA)

---

## 🚨 Critical Issues Identified

### 1. Cockpit Shows "DELISTED MODE PROVIDER UNAUTHORIZED"
- **Status**: 🔍 Investigating
- **Impact**: All functionality appears broken to user
- **Potential Causes**:
  - WOLF symbol is marked as delisted
  - Provider auth failures cascade to UI state
  - State enum inconsistency (shows "TRADING ACTIVE" when stopped)

### 2. Zero Predictions in Production
- **Status**: ✅ Root Cause Identified
- **Cause**: Missing API keys (POLYGON_API_KEY, ALPHAVANTAGE_API_KEY)
- **Impact**: Multi-run returns `{"stocks": 0, "crypto": 0, "vip": 0}`
- **Fix**: User must set API keys in Railway environment

### 3. All Cockpit Panels Empty
- **Status**: 🔍 Requires Investigation
- **Symptoms**: Every panel shows `—`, NO_DATA, or empty arrays
- **Likely Causes**:
  - Backend endpoints returning null/empty data
  - No SSE/streaming updates being emitted
  - Frontend not handling failed API calls properly
  - No error surfacing (failures render as empty instead of error messages)

### 4. Admin Toggles Exposed with No Validation
- **Status**: ⚠️ Security Risk
- **Issue**: Raw config editing with no:
  - Current value display
  - Type validation
  - Safe ranges
  - Change confirmation
- **Risk**: User can set invalid values and break system

---

## 📝 Code Changes Log

### Commit: `fix: rename duplicate /api/cockpit to /api/cockpit/snapshot for legacy endpoint`
- **File**: `wolf_app.py`, `templates/cockpit.html`
- **Change**: Renamed legacy cockpit endpoint to `/api/cockpit/snapshot` to avoid conflict
- **Impact**: New Ghost 2.x `/api/cockpit` now properly served

### Commit: `fix: remove duplicate stub /api/predictions/multi/run endpoint`
- **File**: `wolf_app.py`
- **Change**: Removed stub endpoint at line 14082 that returned empty predictions
- **Impact**: Real implementation at line 18017 now properly used
- **Result**: Health counters now update correctly on every multi-run call

### Commit: `feat(cockpit): wire /api/cockpit and Ghost 2.x health panel`
- **Files**: `wolf_app.py`, `templates/cockpit.html`, `tests/test_cockpit_endpoint.py`
- **Changes**:
  - Implemented `/api/cockpit` endpoint with Ghost 2.x health data
  - Added Ghost 2.x panel HTML to cockpit.html
  - Added JS functions: `renderGhostScore()`, `renderVipHealth()`, `refreshGhost2xPanel()`
  - Wired auto-refresh every 30 seconds
  - Created comprehensive test suite

### Commit: `fix: cockpit auth + add Ghost 2.x health panel`
- **Files**: `static/js/predict.js`, `templates/cockpit.html`
- **Changes**:
  - Added `ghostAuthHeaders()` helper function
  - Applied auth headers to all `/api/predict/*` endpoints
  - Injected `window.GHOST_API_TOKEN` via template script

---

## 🎯 Next Steps Priority

### High Priority (Blocking Production)
1. **Investigate "DELISTED MODE PROVIDER UNAUTHORIZED"** - Must fix to restore UI functionality
2. **Audit all cockpit data endpoints** - Find why everything returns empty/null
3. **Implement error surfacing** - Show actual errors instead of empty data
4. **User: Set API keys in Railway** - Unblocks prediction engine

### Medium Priority (Core Functionality)
5. **Wire Ghost-AI Monitor** - Connect to decision/agent logs
6. **Wire World Context** - Pull SPY, VIX, news feeds
7. **Wire Accuracy Ledger** - Connect to predictions DB
8. **Wire Regime & Risk** - Implement regime detection + risk metrics
9. **Wire Portfolio Panel** - Connect to position DB
10. **Implement Admin config validation** - Safe ranges, current values

### Low Priority (Baseline Features)
11. **Add Goals Panel** - Daily/weekly/monthly targets
12. **Add Ghost Score (GPS) Panel** - Live 0-10 scoring
13. **Add VIP Coins Panel** - WEPE, LILPEPE, DORKL, SLOTH, APC
14. **Add XRP Tracker** - Bullish eye indicator
15. **Add Presale/Strike Prep Panel** - Sniper feed

---

## 📊 Test Results

**Last Run**: 2025-11-16  
**Command**: `pytest tests/test_cockpit_endpoint.py tests/test_multi_predictions.py -v`  
**Result**: ✅ **6/6 passing**

```
tests/test_cockpit_endpoint.py::test_cockpit_endpoint_200 PASSED
tests/test_cockpit_endpoint.py::test_cockpit_system_fields PASSED
tests/test_cockpit_endpoint.py::test_cockpit_ghost2x_fields PASSED
tests/test_cockpit_endpoint.py::test_cockpit_no_auth_required PASSED
tests/test_multi_predictions.py::test_multi_predictions_endpoint_ok_field_present PASSED
tests/test_multi_predictions.py::test_multi_predictions_endpoint_shape_basic PASSED
```

**Coverage**: Backend endpoints verified, prediction engine structure validated  
**Gaps**: Need tests for data population, error handling, auth-protected endpoints

---

## ⚠️ Known Limitations

1. **VIP NO_DATA Behavior**: LILPEPE, SLOTH, APC intentionally return NO_DATA (no price sources)
2. **Price Provider Dependence**: All predictions require at least one working price provider
3. **Market Hours**: Some providers may fail outside market hours without cached data
4. **Error Visibility**: Current UI silently fails (shows empty) instead of surfacing errors

---

*This file is the single source of truth for Ghost 2.x completion status.*
