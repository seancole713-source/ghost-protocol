# Ghost 2.x Cockpit Full Diagnosis & Remediation Plan

**Date**: 2025-11-16
**Agent**: Ghost 2.x Completion Agent
**Status**: 🚨 CRITICAL - Cockpit Appears Completely Offline

---

## 🔍 Executive Summary

The Ghost Intelligence Cockpit is displaying "DELISTED MODE PROVIDER UNAUTHORIZED" with virtually all panels showing
empty data (`—`, `NO_DATA`, or empty arrays). This comprehensive audit identifies:

1. **Root Causes**of the "broken" appearance

2.**Missing Data Wiring**preventing panels from populating
3.**Implementation Plan**to restore full functionality**Critical Finding**: The system is NOT broken - it's **under-configured and under-wired**.
The backend endpoints exist but return empty/null data because:

- Missing API keys (POLYGON_API_KEY, ALPHAVANTAGE_API_KEY)
- No data sources wired for most panels
- UI error handling renders failures as empty data instead of explicit errors

---

## 🚨 Critical UI State Issues

### Issue #1: "DELISTED MODE PROVIDER UNAUTHORIZED"

**What User Sees**:

```text
DELISTED MODE PROVIDER UNAUTHORIZED | stopped | tick: n/a
Focus: — DELISTED MODE PROVIDER UNAUTHORIZED

```text

**Root Cause Analysis**:

1. **DELISTED MODE Badge**(line 176 in cockpit.html):
   - Shown when `/api/price/diagnostics` returns `delisted_hint: true`
   - WOLF is in DELISTED_SYMBOLS registry (line 1331, wolf_app.py) marked as "restructured" with 120:1 reverse split
   - Badge appears even though `untradable: False` (can still trade post-restructuring)


   -**Issue**: Badge is too aggressive - should only show for truly untradable symbols

1. **PROVIDER UNAUTHORIZED Badge**(line 177 in cockpit.html):
   - Shown when SSE stream data includes degraded_reasons containing "price:provider-unauthorized"
   - This happens when price provider API calls return 401/403 (missing/invalid API keys)


   -**Issue**: Missing POLYGON_API_KEY and ALPHAVANTAGE_API_KEY in Railway environment

1. **Focus Display**(line 1897 in cockpit.html):
   - Shows "Focus: WOLF" when FOCUS_WOLF_ONLY=1 (enabled by default per line 1273 wolf_app.py)


   -**Issue**: Locking focus to WOLF (a delisted/restructured symbol) makes entire UI appear broken


**Why This Looks Catastrophic**:

- User sees two red error badges combined with "stopped" and "tick: n/a"
- Creates impression that entire system is offline
- In reality: backend is running, just missing API keys for price providers


---

### Issue #2: All Panels Show Empty Data

**Symptoms by Panel**:

| Panel | Current State | Expected State | Root Cause |
|-------|---------------|----------------|------------|
| Ghost-AI v1 Decision Preview | Empty | Recent decision text | Not wired to decision log |
| Ghost-AI v2 Agent Monitor | All fields `—` | Confidence, decisions, tools | Not wired to agent run logs |
| World Context & Market Mood | SPY/VIX/News all `—` | Live index prices + news | Not wired to market data feeds |
| Daily Accuracy Ledger | 0/0/0/0, empty table | Prediction results | Not wired to predictions DB |
| Market Regime & Risk | All fields empty | Regime, drawdown, VaR | No regime detection implemented |
| Portfolio Optimization | "No allocation calculated" | Real allocation/hedge | No optimizer wired |
| Smart Execution | All `—` but says "TRADING ACTIVE" | Fill rate, latency, orders | Not wired to execution logs |
| Personal Portfolio | Empty table | User positions | DB table exists but UI may not load |
| Watchlist | Empty textareas | Symbol lists | No persistence wired |
| Diagnostics | "Showing last 50 events" but none visible | Event log | Frontend filter issue |
| Ghost Prediction | Empty predictions table | Symbol forecasts | Works but needs symbols= param |

**Core Problem**: UI polling endpoints that return `null`, `[]`, or `{}` instead of populated data structures.

---

### Issue #3: Misleading State Labels

**Contradictions**:

1. Top banner says "stopped" but Smart Execution says "RISK STATUS: TRADING ACTIVE"
2. Multiple "Refresh" buttons with no visual feedback when API calls fail
3. No error messages when endpoints return 404/500 - just renders as empty data
4. Admin Toggles show blank inputs (no current values displayed)


**Impact**: User cannot distinguish between:

- "System is working but no activity yet"
- "System is broken / offline"
- "API call failed / endpoint not implemented"


---

## 📋 Backend Endpoint Audit

### Fully Implemented & Working ✅

| Endpoint | Status | Returns | Used By Panel |
|----------|--------|---------|---------------|
| `/api/status` | ✅ Working | `{mode, active, version}` | Top badge |
| `/ui/health` | ✅ Working | `{status: "ok"}` | Internal checks |
| `/api/health/predictions` | ✅ Working | Ghost 2.x health data | Ghost 2.x Health panel |
| `/api/cockpit` | ✅ Working | Ghost 2.x + legacy snapshot | Ghost 2.x panel (partial) |
| `/api/predictions/multi/run` | ✅ Working | Multi-symbol predictions (0 due to API keys) | Multi-run jobs |
| `/api/cockpit/stream` | ✅ Working | SSE stream with snapshots | Real-time updates |

### Partially Implemented ⚠️

| Endpoint | Status | Issue |
|----------|--------|-------|
| `/api/predict/*` | ⚠️ Partial | Requires auth; works for single symbols but multi-run returns 0 |
| `/api/price/diagnostics` | ⚠️ Partial | Works but requires `symbol=` param; UI may call without it |
| `/api/runtime/config` | ⚠️ Partial | Admin Toggles - doesn't return current values |

### Not Implemented / Missing ❌

| Expected Endpoint | Current State | Required For Panel |
|-------------------|---------------|-------------------|
| `/api/agents/decisions` or similar | ❌ Missing | Ghost-AI v1/v2 Monitor |
| `/api/market/context` or `/api/indexes` | ❌ Missing | World Context (SPY, VIX) |
| `/api/news/context` | ❌ Missing | World Context (news feed) |
| `/api/predictions/accuracy` or `/api/map` | ❌ Missing | Daily Accuracy Ledger |
| `/api/regime` | ❌ Missing | Market Regime panel |
| `/api/risk/metrics` | ❌ Missing | Risk metrics (VaR, drawdown) |
| `/api/portfolio/optimization` | ❌ Missing | Portfolio Optimization panel |
| `/api/execution/stats` | ❌ Missing | Smart Execution panel |
| `/api/portfolio/positions` | ⚠️ Unclear | Personal Portfolio table |
| `/api/watchlist` | ❌ Missing | Watchlist persistence |
| `/api/events` or `/logs/recent` | ⚠️ Exists? | Diagnostics panel |

---

## 🔧 Environment Configuration Issues

### Missing API Keys (User-Only Tasks)

**Critical Blockers**- Must be set in Railway:

```bash

POLYGON_API_KEY=(obtain from polygon.io)
ALPHAVANTAGE_API_KEY=(obtain from alphavantage.co)

```text**Impact**: Without these:

- All stock price fetches fail
- Multi-run predictions return 0 for all 21 symbols
- Price diagnostics show "provider: unavailable"
- UI displays "PROVIDER UNAUTHORIZED" badge


**How to Fix**:

1. Log into Railway dashboard
2. Navigate to ghost-protocol project → Variables tab
3. Add both keys with valid values
4. Trigger redeploy or wait for auto-restart


### Configuration Validation Findings

**From wolf_app.py startup validation (lines 3475-3525)**:

Current enforcement:

```python

# These MUST be 0 or unset for production

SIM_MODE=0                      # ✅ Required
DELISTED_MODE=0                 # ✅ Required
ALLOW_SAFE_PRICE=0              # ✅ Required
PRICE_FALLBACK_PREVCLOSE=0      # ✅ Required

# These MUST be present

POLYGON_API_KEY=(not empty)     # ❌ MISSING IN PRODUCTION
ALPHAVANTAGE_API_KEY=(not empty) # ❌ MISSING IN PRODUCTION

```text

**Current Production State**(inferred from behavior):

- `SIM_MODE`: Likely 0 (good)
- `DELISTED_MODE`: Unknown (should verify it's 0)
- `FOCUS_WOLF_ONLY`: 1 (default) -**PROBLEM**: locks UI to delisted symbol
- `POLYGON_API_KEY`: Missing/invalid
- `ALPHAVANTAGE_API_KEY`: Missing/invalid


**Recommended Changes**:

```bash

# Railway environment - suggested settings

SIM_MODE=0
DELISTED_MODE=0
FOCUS_WOLF_ONLY=0              # ← CHANGE: Allow any symbol focus
ALLOW_SAFE_PRICE=0
PRICE_FALLBACK_PREVCLOSE=0
POLYGON_API_KEY=(valid key)
ALPHAVANTAGE_API_KEY=(valid key)
STOCK_PRICE_SOURCE=polygon
CRYPTO_PRICE_SOURCE=coingecko
PRICE_MIN_PROVIDERS=1
PRICE_REQUIRE_QUORUM=0
PREDICT_REQUIRE_PRICE_QUORUM=0

```text

---

## 🎯 Implementation Roadmap

### Phase 1: Fix Critical UI State (1-2 hours)

**Goal**: Remove misleading "DELISTED MODE PROVIDER UNAUTHORIZED" appearance

**Tasks**:

1. **Soften Delisted Badge Logic**(cockpit.html line 2410):
   - Don't show "DELISTED MODE" for symbols with `untradable: False`
   - Only show for truly blocked symbols
   - Change badge text to "Corporate Action: 120:1 Split" for WOLF


1.**Add Error Surfacing**(cockpit.html, multiple locations):

   - When fetch fails, display explicit error message instead of empty data
   - Add visual feedback for failed "Refresh" clicks
   - Show timestamps for last successful data refresh per panel


1.**Fix State Label Consistency**:

   - Don't show "TRADING ACTIVE" if engine is stopped or degraded
   - Sync activeBadge with actual trading loop state
   - Add "Degraded (missing API keys)" state between "active" and "stopped"

1. **Improve Focus Handling**:
   - Allow user to override FOCUS_WOLF_ONLY via UI toggle
   - Show "Focus: WOLF (restricted)" instead of "Focus: — DELISTED MODE..."
   - Add button to switch focus to AAPL or BTC (non-controversial symbols)


**Code Changes**:

```javascript

// cockpit.html - Soften delisted badge (line ~2410)
async function poll(){
    try {
        const r = await fetch('/api/price/diagnostics?symbol=WOLF');
        if(!r.ok) {
            // Show explicit error instead of silent fail
            if(r.status === 404) {
                console.warn('Price diagnostics endpoint not found');
            }
            return;
        }
        const d = await r.json();

        // Only show delisted badge if truly untradable
        if(d && d.delisted_hint && delistedBadge){
            const corpAction = await fetch('/api/corporate_actions').then(r=>r.json()).catch(()=>({}));
            const wolfAction = corpAction?.actions?.WOLF;
            if(wolfAction && !wolfAction.untradable) {
                // Restructured but tradable - show softer message
                delistedBadge.textContent = 'Corporate Action: 120:1 Split';
                delistedBadge.style.background = '#2a2a16';  // Less alarming color
            }
            delistedBadge.style.display='inline-flex';
        }
        renderRateDiag(d||{});
    } catch(e){
        console.error('Poll error:', e);
        // Show error to user
    }
}

```text

### Phase 2: Wire Data Sources to Backend (4-6 hours)

**Goal**: Implement missing backend endpoints so panels show real data

#### 2.1 Ghost-AI Monitor

**New Endpoint**: `/api/agents/monitor`

**Returns**:

```json

{
  "confidence": 0.87,
  "decisions_24h": 12,
  "tool_success_rate": 0.94,
  "last_decision": {
    "timestamp": "2025-11-16T07:53:24Z",
    "action": "BUY_WEPE",
    "confidence": 0.89
  },
  "recent_decisions": [
    {"time": "07:53", "action": "BUY_WEPE", "outcome": "success"},
    // ... last 10
  ],
  "tool_performance": [
    {"tool": "price_fetch", "calls_24h": 156, "success_rate": 0.98},
    // ... top 5 tools
  ]
}

```text

**Implementation**:

- Query agent decision log (likely stored in EVENTS or separate DB)
- Calculate success rate from outcomes
- Track tool calls per agent framework


#### 2.2 World Context & Market Mood

**New Endpoint**: `/api/market/context`

**Returns**:

```json

{
  "indexes": {
    "SPY": {"price": 450.23, "change_pct": 0.45, "updated": 1700000000},
    "VIX": {"price": 14.56, "change_pct": -2.1, "updated": 1700000000}
  },
  "confidence": 0.78,
  "news_summary": {
    "trending": ["Fed rate decision", "Tech earnings"],
    "top_headlines": [
      {"title": "...", "source": "Reuters", "sentiment": 0.6},
      // ... top 5
    ],
    "articles_24h": 47
  }
}

```text

**Implementation**:

- Use existing price providers to fetch SPY, VIX
- Wire to news connector (check if `NEWS_CACHE` is populated)
- Calculate market confidence from volatility + news sentiment


#### 2.3 Daily Accuracy Ledger

**New Endpoint**: `/api/predictions/accuracy`

**Returns**:

```json

{
  "summary": {
    "correct": 34,
    "warning": 12,
    "wrong": 8,
    "pending": 15
  },
  "map_table": [
    {
      "symbol": "WOLF",
      "predictions": 5,
      "hit_rate": 0.60,
      "avg_confidence": 0.72
    },
    // ... all predicted symbols
  ],
  "auto_tuning": {
    "last_tune": "2025-11-15T12:00:00Z",
    "tune_count": 23,
    "current_config": "ensemble_v2"
  }
}

```text

**Implementation**:

- Query predictions DB or FORECAST_STORE (line 4011 in wolf_app.py)
- Compare forecasts to FORECAST_ACTUALS
- Implement classification: correct (within 5%), warning (5-15%), wrong (>15%)


#### 2.4 Market Regime & Risk

**New Endpoint**: `/api/regime`

**Returns**:

```json

{
  "current_regime": {
    "type": "bull_trend",
    "confidence": 0.81,
    "since": "2025-11-10T09:30:00Z"
  },
  "regime_factors": {
    "trend": "up",
    "volatility": "low",
    "liquidity": "high"
  },
  "risk_metrics": {
    "current_drawdown_pct": -2.3,
    "var_95": 4567.89,
    "status": "normal",
    "max_position_size": 50000
  },
  "position_status": {
    "risk_level": "moderate",
    "position_pct": 65,
    "stop_loss": 0.05
  }
}

```text

**Implementation**:

- Implement regime detection: analyze SPY trend (SMA crossover), VIX level, volume
- Calculate VaR from portfolio positions + symbol volatilities
- Use existing ALERT_STATE for trailing stops


#### 2.5 Portfolio & Execution

**New Endpoint**: `/api/portfolio/positions`

**Returns**:

```json

{
  "positions": [
    {
      "symbol": "WOLF",
      "type": "stock",
      "qty": 1000,
      "entry": 2.45,
      "current": 2.67,
      "pnl_abs": 220.00,
      "pnl_pct": 8.98,
      "gps": 7.2
    },
    // ... all positions
  ]
}

```text

**New Endpoint**: `/api/execution/stats`

**Returns**:

```json

{
  "stats_24h": {
    "success_rate": 0.98,
    "avg_latency_ms": 156,
    "fill_rate": 0.95,
    "total_orders": 34
  },
  "risk_status": "active" | "stopped" | "degraded",
  "execution_quality": "excellent" | "good" | "poor"
}

```text

**Implementation**:

- Portfolio: Query STATE variable + any DB tables with positions
- Execution: Track order log (may need to instrument execution code)


#### 2.6 Portfolio Optimization

**New Endpoint**: `/api/portfolio/optimization`

**Returns**:

```json

{
  "allocation": {
    "symbols": ["WOLF", "BTC", "ETH"],
    "weights": [0.4, 0.35, 0.25],
    "expected_return": 0.18,
    "volatility": 0.22,
    "sharpe": 0.82
  },
  "hedge": {
    "instrument": "SPY_PUT",
    "size": 0.15,
    "cost_pct": 2.3
  },
  "backtest": {
    "period": "90d",
    "return_pct": 12.4,
    "max_drawdown_pct": -8.2,
    "sharpe": 1.45
  }
}

```text

**Implementation**:

- Implement mean-variance optimization (scipy.optimize)
- Use historical returns from price data
- Calculate hedging instruments based on portfolio beta


### Phase 3: Add Baseline UI Modules (3-4 hours)

**Goal**: Implement missing Ghost baseline features

#### 3.1 Goals Panel

**HTML**(add to cockpit.html after Ghost 2.x Health):

```html

<section class="card" style="grid-column: span 6;">
    <div class="section-title">
        <h2>🎯 Goals & Targets</h2>
        <button id="btnGoalsRefresh">Refresh</button>
    </div>

    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
        <div class="goal-card">
            <div class="goal-label">Daily</div>
            <div class="goal-progress" id="goal-daily-pnl">$0</div>
            <div class="goal-target">Target: $<span id="goal-daily-target">500</span></div>
            <div class="goal-bar">
                <div id="goal-daily-bar" style="width: 0%;"></div>
            </div>
        </div>

        <div class="goal-card">
            <div class="goal-label">Weekly</div>
            <div class="goal-progress" id="goal-weekly-pnl">$0</div>
            <div class="goal-target">Target: $<span id="goal-weekly-target">2500</span></div>
            <div class="goal-bar">
                <div id="goal-weekly-bar" style="width: 0%;"></div>
            </div>
        </div>

        <div class="goal-card">
            <div class="goal-label">Monthly</div>
            <div class="goal-progress" id="goal-monthly-pnl">$0</div>
            <div class="goal-target">Target: $<span id="goal-monthly-target">10000</span></div>
            <div class="goal-bar">
                <div id="goal-monthly-bar" style="width: 0%;"></div>
            </div>
        </div>

        <div class="goal-card">
            <div class="goal-label">Yearly</div>
            <div class="goal-progress" id="goal-yearly-pnl">$0</div>
            <div class="goal-target">Target: $<span id="goal-yearly-target">120000</span></div>
            <div class="goal-bar">
                <div id="goal-yearly-bar" style="width: 0%;"></div>
            </div>
        </div>
    </div>
</section>

```text**Backend**: `/api/goals` endpoint returning PnL by time period

#### 3.2 Ghost Score (GPS) Panel

**HTML**:

```html

<section class="card ghost-score-card" style="grid-column: span 3;">
    <h2>📡 Ghost Score (GPS)</h2>
    <div class="gps-display">
        <div class="gps-value" id="gps-value">0.0</div>
        <div class="gps-gauge" id="gps-gauge"></div>
    </div>
    <div class="gps-components">
        <div>Risk: <span id="gps-risk">—</span></div>
        <div>Hit Rate: <span id="gps-hit">—</span></div>
        <div>Regime: <span id="gps-regime">—</span></div>
    </div>
</section>

```text

**Backend**: Implement GPS algorithm (0-10 scale based on risk, hit rate, regime alignment)

#### 3.3 VIP Coins Panel

**HTML**:

```html

<section class="card" style="grid-column: span 9;">
    <div class="section-title">
        <h2>💎 VIP Coins Tracker</h2>
        <button id="btnVipRefresh">Refresh</button>
    </div>

    <table class="vip-table">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Price</th>
                <th>24h %</th>
                <th>GPS</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody id="vip-coins-body">
            <tr><td colspan="5">Loading...</td></tr>
        </tbody>
    </table>
</section>

```text

**Backend**: `/api/vip/status` returning WEPE, LILPEPE, DORKL, SLOTH, APC data

#### 3.4 XRP Tracker

**HTML**:

```html

<section class="card" style="grid-column: span 3;">
    <h2>👁️ XRP Bullish Eye</h2>
    <div class="xrp-price" id="xrp-price">$0.00</div>
    <div class="xrp-indicator" id="xrp-indicator">🔴 Bearish</div>
    <div class="xrp-metrics">
        <div>Trend: <span id="xrp-trend">—</span></div>
        <div>Volume: <span id="xrp-volume">—</span></div>
        <div>Signal: <span id="xrp-signal">—</span></div>
    </div>
</section>

```text

**Backend**: Use existing crypto price provider + add trend analysis

#### 3.5 Presale / Strike Prep Panel

**HTML**:

```html

<section class="card" style="grid-column: span 12;">
    <div class="section-title">
        <h2>🎯 Presale Sniper Feed</h2>
        <button id="btnPresaleRefresh">Refresh</button>
    </div>

    <table class="presale-table">
        <thead>
            <tr>
                <th>Token</th>
                <th>Launch</th>
                <th>Status</th>
                <th>GPS</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody id="presale-feed-body">
            <tr><td colspan="5">No presales tracked</td></tr>
        </tbody>
    </table>
</section>

```text

**Backend**: `/api/presales` endpoint (may need external presale tracker API)

### Phase 4: Admin Config Hardening (1 hour)

**Goal**: Make `/api/runtime/config` safe and useful

**Changes Needed**:

1. Return current values for all config fields (not empty strings)
2. Add validation: reject invalid types/ranges
3. Add confirmation dialog before applying changes
4. Log all config changes to audit trail
5. Remove dead/unused flags


**Implementation**:

```python

@APP.get("/api/runtime/config")
async def api_runtime_config_get():
    """Return current runtime configuration."""
    return {
        "price_ttl_s": PRICE_TTL_S,
        "price_ttl_open_s": PRICE_TTL_OPEN_S,
        "news_ttl_s": NEWS_TTL_S,
        "price_max_deviation_open": PRICE_MAX_DEVIATION_OPEN,
        "diag_ring_size": EVENTS.maxlen,
        "overlay_dt_minutes": OVERLAY_DT_MINUTES,
        "band_widen_factor": BAND_WIDEN_FACTOR,
        "overlay_enabled": OVERLAY_ENABLED,
        "learning_enabled": LEARNING_ENABLED,
        "reuters_feeds_on": REUTERS_FEEDS_ON,
        "diag_collapse_dupes": DIAG_COLLAPSE_DUPES,
        "yahoo_first": PRICE_YAHOO_FIRST
    }

@APP.post("/api/runtime/config")
async def api_runtime_config_set(
    config: dict[str, Any],
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """Update runtime configuration with validation."""
    _require_bearer(...)

    # Validate types and ranges

    if "price_ttl_s" in config:
        val = int(config["price_ttl_s"])
        if not (10 <= val <= 3600):
            raise HTTPException(422, "price_ttl_s must be 10-3600")
        globals()["PRICE_TTL_S"] = val

    # ... validate other fields

    # Log change

    _add_event("config.update", "Runtime config updated", config)

    return {"ok": True, "updated": list(config.keys())}

```text

### Phase 5: Testing & Validation (2 hours)

**Goal**: Verify all changes work end-to-end

**Test Plan**:

1. Set POLYGON_API_KEY and ALPHAVANTAGE_API_KEY in Railway
2. Confirm multi-run returns non-zero predictions
3. Check each cockpit panel loads real data (not empty)
4. Verify error messages appear when API calls fail
5. Test admin config changes with validation
6. Run pytest suite (should remain at 6/6 passing)


**Validation Script** (create `scripts/verify_cockpit.py`):

```python

import requests
import sys

PROD_BASE = "<<<<<https://ghost-protocol-production.up.railway.app">>>>>

def test_endpoint(path, expected_keys):
    """Test endpoint returns expected structure."""
    try:
        r = requests.get(f"{PROD_BASE}{path}", timeout=10)
        if r.status_code != 200:
            print(f"❌ {path} returned {r.status_code}")
            return False
        data = r.json()
        missing = [k for k in expected_keys if k not in data]
        if missing:
            print(f"⚠️  {path} missing keys: {missing}")
            return False
        print(f"✅ {path}")
        return True
    except Exception as e:
        print(f"❌ {path} failed: {e}")
        return False

# Test all critical endpoints

tests = [
    ("/api/status", ["mode", "active", "version"]),
    ("/ui/health", ["status"]),
    ("/api/health/predictions", ["ghost_score_v2", "symbol_counts"]),
    ("/api/cockpit", ["ghost_2x"]),
    ("/api/predictions/multi/run", ["ok", "counts"]),
]

results = [test_endpoint(path, keys) for path, keys in tests]
success_rate = sum(results) / len(results)
print(f"\n📊 Success Rate: {success_rate*100:.0f}% ({sum(results)}/{len(results)})")
sys.exit(0 if success_rate == 1.0 else 1)

```text

---

## 📈 Success Metrics

### Before Fixes

- ❌ Cockpit shows "DELISTED MODE PROVIDER UNAUTHORIZED"
- ❌ All panels empty (`—`, `NO_DATA`, or empty arrays)
- ❌ Multi-run returns 0 predictions
- ❌ No error visibility (silent failures)
- ❌ Admin config shows blank fields
- ❌ Missing baseline modules (Goals, GPS, VIP, XRP, Presales)


### After Phase 1 (Critical Fixes)

- ✅ Softer "Corporate Action" message (not "DELISTED MODE")
- ✅ "Provider Unauthorized" only shows when API keys actually missing
- ✅ Focus switching enabled (user can select AAPL/BTC)
- ✅ Error messages visible when API calls fail
- ✅ State labels consistent (no "TRADING ACTIVE" when stopped)


### After Phase 2 (Data Wiring)

- ✅ Ghost-AI Monitor shows confidence, decisions, tool success
- ✅ World Context shows SPY, VIX, news articles
- ✅ Accuracy Ledger shows prediction results table
- ✅ Market Regime shows current regime + risk metrics
- ✅ Portfolio shows positions with PnL
- ✅ Execution shows fill rate, latency, order count
- ✅ Diagnostics shows event log


### After Phase 3 (Baseline Modules)

- ✅ Goals panel shows daily/weekly/monthly/yearly progress
- ✅ Ghost Score (GPS) displays 0-10 live score
- ✅ VIP Coins panel tracks WEPE, LILPEPE, DORKL, SLOTH, APC
- ✅ XRP tracker shows bullish eye indicator
- ✅ Presale feed lists upcoming sniper targets


### After Phase 4 (Admin Hardening)

- ✅ Admin config displays current values
- ✅ Type validation on all inputs
- ✅ Confirmation dialog before applying changes
- ✅ Audit log for config changes


### After Phase 5 (Testing)

- ✅ All tests passing (6/6 minimum)
- ✅ Multi-run returns >0 predictions with API keys
- ✅ End-to-end cockpit verification script passes


---

## 🚀 Execution Priority

**Critical Path**(fixes user perception of "broken system"):

1. Set POLYGON_API_KEY + ALPHAVANTAGE_API_KEY in Railway
2. Phase 1: Fix UI state (remove misleading error badges)
3. Phase 2.1: Wire Ghost-AI Monitor (high visibility panel)
4. Phase 2.2: Wire World Context (SPY/VIX/news)**High Value**(core functionality):

1. Phase 2.3: Accuracy Ledger (proves predictions work)
2. Phase 2.4: Market Regime (trading intelligence)
3. Phase 3.2: Ghost Score (GPS) panel (brand identity)**Nice to Have**(complete the baseline):

1. Phase 3: Remaining baseline modules
2. Phase 4: Admin config hardening

1. Phase 5: Comprehensive testing


---

## 📝 User Action Items (Cannot Be Automated)

### URGENT - Railway Environment

```bash

# Log into Railway dashboard and add

POLYGON_API_KEY=(get from polygon.io)
ALPHAVANTAGE_API_KEY=(get from alphavantage.co)

# Verify/set these as well

FOCUS_WOLF_ONLY=0  # Allow flexible symbol selection
SIM_MODE=0         # Live mode only
DELISTED_MODE=0    # No delisted mode

```text

### Verification Steps

1. Wait 2-3 minutes for Railway restart
2. Visit <<<<<https://ghost-protocol-production.up.railway.app/cockpit>>>>>
3. Check if "PROVIDER UNAUTHORIZED" badge disappears
4. Run: `curl <<<<<https://ghost-protocol-production.up.railway.app/api/predictions/multi/run>>>>> | jq '.counts'`
5. Confirm: `{"stocks": X, "crypto": Y, "vip": Z}` where X+Y+Z > 0


### External API Accounts Needed

- Polygon.io account (stock prices)
- AlphaVantage account (backup stock prices)
- CoinGecko API (crypto prices, may be free tier)
- Reuters or news API (if not already configured)


---

## 🔬 Technical Notes

### Why Multi-Run Returns 0**Code Flow**

1. `/api/predictions/multi/run` → `_generate_multi_symbol_predictions()` (line 17871)
2. For each of 21 symbols: call `_generate_48h_forecast(symbol)` (line 2975)
3. _generate_48h_forecast() immediately fails at line 3009:


   ```python

   if price is None or price <= 0:
       return {"ok": False, "error": "live price unavailable"}

   ```text

1. `get_wolf_price()`, `get_price_quorum()`, `get_vip_price()` all return `None`
2. Root cause: Missing API keys → HTTP 401 → price provider failures


**Proof**: Health counters update correctly even when 0 predictions (lines 17963-17968)

### Why UI Shows Empty Data

**Root Causes**:

1. **Backend returns null/empty**: Many endpoints not implemented (market context, accuracy, regime, etc.)
2. **Frontend renders null as "—"**: No error handling in JS (cockpit.html)
3. **SSE stream doesn't emit updates**: Real-time panels never refresh
4. **No timestamps**: User can't tell if data is stale or missing


**Example**(World Context panel):

```javascript

// Current code (line ~XXX in cockpit.html):
const spyPrice = data.spy_price;  // undefined
document.getElementById('spy-price').textContent = spyPrice || '—';

// Should be:
try {
    const spyPrice = data.spy_price;
    if (spyPrice === undefined || spyPrice === null) {
        throw new Error('SPY price not available');
    }
    document.getElementById('spy-price').textContent = `$${spyPrice.toFixed(2)}`;
    document.getElementById('spy-updated').textContent = formatTime(data.spy_updated);
} catch(e) {
    document.getElementById('spy-price').textContent = '❌ Error';
    document.getElementById('spy-error').textContent = e.message;
    document.getElementById('spy-error').style.display = 'block';
}

```text

### Ghost Score V2 vs GPS**Clarification**

- **Ghost Score V2**: Existing implementation in backend (wolf_app.py) for health monitoring
  - Data quality (40%): symbol coverage, provider redundancy
  - Prediction coverage (35%): success rate
  - Risk behavior (25%): position limits, drawdown compliance
  - Range: 0-100 with letter grade (A-F)

- **Ghost Score (GPS)**: **NEW**live 0-10 indicator for cockpit
  - Simplified real-time score
  - Components: risk level, hit rate, regime alignment
  - Purpose: At-a-glance "is Ghost performing well right now?"
  - Should be computed every tick (SSE stream)**Recommendation**: Implement GPS as separate from Ghost Score V2:


```python

def calculate_gps() -> float:
    """Calculate Ghost Performance Score (GPS) 0-10."""
    risk_score = 1.0 - (current_drawdown / max_drawdown_threshold)  # 0-1
    hit_rate = recent_predictions_correct / recent_predictions_total  # 0-1
    regime_score = 1.0 if current_regime == "favorable" else 0.5  # 0-1

    raw_gps = (risk_score *0.4) + (hit_rate*0.4) + (regime_score* 0.2)
    return round(raw_gps * 10, 1)  # Scale to 0-10

```text

---

## 📚 References

### Key Code Locations

| Component | File | Line | Description |
|-----------|------|------|-------------|
| STATE global | wolf_app.py | 3989 | Main engine state dict |
| DELISTED_SYMBOLS | wolf_app.py | 1331 | Corporate actions registry |
| FOCUS_WOLF_ONLY | wolf_app.py | 1273 | Symbol focus restriction |
| Environment validation | wolf_app.py | 3475 | API key checks |
| SSE stream | wolf_app.py | 12318 | Cockpit real-time updates |
| Multi-run | wolf_app.py | 17871 | Prediction engine |
| Delisted badge | cockpit.html | 176 | UI banner element |
| Provider unauth badge | cockpit.html | 177 | API key error indicator |
| Focus badge | cockpit.html | 175 | Symbol focus display |
| Price diagnostics poll | cockpit.html | 2400 | 10s polling for delisted check |

### Related Documents

- GHOST_2X_TODO.md - Master checklist (created by this agent)
- tests/test_cockpit_endpoint.py - Endpoint validation
- tests/test_multi_predictions.py - Multi-run tests


---

**End of Diagnosis Report**
