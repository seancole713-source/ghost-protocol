# 🚀 Ghost Cockpit UI Rebuild Plan

**Date**: November 20, 2025
**Goal**: Transform WOLF-centric cockpit into multi-asset Hunter dashboard
**Current File**: `templates/cockpit.html` (3015 lines)

---

## 📋 Current Panels (Discovered via grep search)

### ✅ PANELS TO KEEP

1. **Ghost‑AI v1 — Decision Preview**(line 212)


2.**🤖 Ghost‑AI v2 — Agent Monitor**(line 221)
3.**🌍 World Context & Market Mood**(line 283)
4.**📊 Daily Accuracy Ledger**(line 331)
5.**🎯 Market Regime & Risk**(line 404)
6.**🎲 Portfolio Optimization**(line 449)
7.**🔎 Research Snapshot**(line 561)
8.**Ghost Prediction**(line 570)
9.**Admin Toggles**(line 636)


### 🔄 PANELS TO REBUILD

1.**Ghost Score Heatmap (GPS 0–10)**(line 542) - Single WOLF heatmap → Multi-asset
2.**Top Movers**(needs expansion to 4000+ stocks)
3.**Personal Portfolio**(needs multi-asset support)
4.**News Feed**(currently disabled - reuters_feeds_off)

### ❌ PANELS TO REMOVE

1.**⚡ Smart Execution**(line 495) - Always shows zeros, not active
2.**Diagnostics**(line 551) - Merge into Health panel
3.**⏱ Provider Backoff**(standalone) - Merge into Provider Health Matrix

---

## 🎯 New Layout Structure

### ROW 0: Header / Environment

```text
┌─────────────────────────────────────────────────────────────────┐
│ 👻 GHOST | [LIVE/SIM] | Last Update: XX:XX:XX                  │
│                                                                  │
│ Ghost 2.x Health: [A] Excellent   Data: OK   AI: Active   Risk: OK │
└─────────────────────────────────────────────────────────────────┘

```text

### ROW 1: Goals + Hunter Feed + VIP

```text

┌──────────────┬─────────────────────────────────┬──────────────┐
│ GOALS +      │ HUNTER FEED                     │ VIP COINS    │
│ GHOST SCORE  │ Top 20 Opportunities            │ ------------ │
│ ------------ │ ┌───────┬─────┬────┬────┬─────┐ │ WEPE: $X.XX  │
│ Score: 87/100│ │Symbol │Mkt  │Prc │Δ%  │GPS  │ │ LILPEPE: $X  │
│ Grade: B+    │ │AAPL   │STK  │$150│+2% │92   │ │ DORKL: $X    │
│              │ │BTC    │CRYPT│$99k│+5% │88   │ │ SLOTH: $X    │
│ Daily: 65%   │ │SPY    │ETF  │$580│+1% │85   │ │ APC: $X      │
│ Weekly: 45%  │ └───────┴─────┴────┴────┴─────┘ │              │
│ Monthly: 32% │                                  │ XRP TRACKER  │
│ Yearly: 18%  │ [Multi-asset, clickable rows]   │ Price: $X.XX │
└──────────────┴──────────────────────────────────┴──────────────┘

```text

### ROW 2: Macro + Risk + Portfolio

```text

┌──────────────┬───────────────────────────┬──────────────┐
│ WORLD CONTEXT│ RISK ENGINE               │ PORTFOLIO    │
│ ------------ │ ------------------------- │ ------------ │
│ SPY: $580    │ NAV: $10,525              │ Top 5 Pos    │
│ QQQ: $485    │ Open Risk: 15%            │ AAPL  +5.2%  │
│ VIX: 14.2    │ Max Pos: 25% (actual 18%) │ BTC   +12.1% │
│ BTC: $99,500 │ VaR(95%): $850            │ SPY   +2.1%  │
│ DXY: 103.5   │ Drawdown: -2.3%           │ WEPE  +45.8% │
│              │ Status: 🟢 Green          │ WOLF  -93.2% │
│ Regime: BULL │                           │              │
│ Conf: 78%    │                           │ [View Full]  │
└──────────────┴───────────────────────────┴──────────────┘

```text

### ROW 3: Predictions + AI Brain + Accuracy

```text

┌──────────────┬───────────────────────────┬──────────────┐
│ PREDICTIONS  │ GHOST AI ACTIVITY         │ ACCURACY     │
│ ------------ │ ------------------------- │ ------------ │
│ Symbol: [▼]  │ Decisions (24h): 42       │ Daily: 68%   │
│ AAPL ▼       │ Tool Calls: 156 (98% ✓)   │ Weekly: 72%  │
│              │                           │ Monthly: 69% │
│ Last:        │ Last 3 Actions:           │              │
│ BUY          │ • AAPL → BUY (12:45)      │ Correct: 45  │
│ Conf: 78%    │ • BTC → HOLD (11:30)      │ Warning: 8   │
│ Horizon: 24h │ • SPY → WATCH (10:15)     │ Wrong: 12    │
│              │                           │ Pending: 3   │
│ [Run New]    │ Status: 🟢 Active         │              │
└──────────────┴───────────────────────────┴──────────────┘

```text

### ROW 4: Logs, Providers, Settings

```text

┌──────────────────────────────────────────────────────────┐
│ PROVIDER HEALTH MATRIX                                   │
│ Polygon: 🟢 OK (45ms) │ Yahoo: 🟢 OK (120ms) │ Binance: 🟢 │
│ AlphaVantage: 🟡 Degraded │ Reuters: 🔴 Down          │
└──────────────────────────────────────────────────────────┘
│ SYSTEM LOGS (Last 10 warnings)                           │
│ • 14:32 - Price staleness detected for WOLF              │
│ • 13:15 - Reuters feed timeout (retry succeeded)         │
└──────────────────────────────────────────────────────────┘
│ RUNTIME CONFIG                                           │
│ [Admin config panel - hide blank fields]                 │
└──────────────────────────────────────────────────────────┘

```text

---

## 🛠️ Implementation Steps

### Step 1: Backup & Create New Template ✅

- Backup current `cockpit.html` → `cockpit_v1_backup.html`
- Create new `cockpit_v2.html` from scratch
- Start with minimal layout, progressively add panels


### Step 2: ROW 0 - Header Implementation**Files**: `cockpit_v2.html`

**API**: `/api/health`, `/api/cockpit/snapshot`
**Fields Needed**:

- `mode` (LIVE/SIM)
- `last_update_time`
- `health.ghost_score`
- `health.data_status`
- `health.ai_status`
- `health.risk_status`


### Step 3: ROW 1 Column 1 - Goals + Ghost Score

**API**: `/api/cockpit/snapshot`
**Fields Needed**:

- `ghost_score.score` (0-100)
- `ghost_score.grade` (F-A)
- `goals.daily`, `goals.weekly`, `goals.monthly`, `goals.yearly`
- If missing: show "—" placeholder


### Step 4: ROW 1 Column 2 - Hunter Feed

**API**: `/api/hunter/feed` (preferred) or `/api/cockpit/snapshot → top_movers`
**Fields Needed**:

- Array of objects: `{ symbol, market_type, price, change_pct, volume_ratio, gps_score }`
- Fallback: If only single symbol available, show that + placeholder rows


### Step 5: ROW 1 Column 3 - VIP + XRP + Presales

**API**: `/api/cockpit/snapshot → vip_health`, `/api/crypto/price/{symbol}`
**Symbols**: WEPE, LILPEPE, DORKL, SLOTH, APC, XRP
**Fields Needed**:

- For each VIP: `price`, `change_24h`, `status` (Calm/Heating/Exploding)
- XRP: `price`, `trend`, `signals`


### Step 6: ROW 2 - Macro + Risk + Portfolio

**API**: `/api/world/context`, `/api/cockpit/snapshot`
**Fields Needed**:

- Macro: `spy`, `qqq`, `vix`, `btc`, `dxy` (price + change_pct)
- Risk: `nav`, `open_risk_pct`, `max_position_pct`, `var_95`, `drawdown`, `status`
- Portfolio: Top 5 positions with `symbol`, `qty`, `avg_cost`, `current_price`, `pnl_pct`


### Step 7: ROW 3 - Predictions + AI + Accuracy

**API**: `/api/predict/latest`, `/api/cockpit/snapshot → ai_telemetry`, `/api/accuracy/ledger`
**Fields Needed**:

- Predictions: `direction`, `confidence`, `horizon`, `timestamp`
- AI: `decisions_24h`, `tool_calls`, `success_rate`, `recent_actions[]`
- Accuracy: `daily_pct`, `weekly_pct`, `monthly_pct`, `correct`, `warning`, `wrong`, `pending`


### Step 8: ROW 4 - Providers + Logs + Config

**API**: `/api/providers/health`, `/api/logs/recent`, `/api/runtime/config`
**Fields Needed**:

- Providers: Array of `{ name, status, latency_ms }`
- Logs: Array of `{ timestamp, level, message }`
- Config: Runtime config object (filter out empty/null fields)


### Step 9: Remove Obsolete Panels

**Delete**:

- Smart Execution panel (line 495)
- Standalone Diagnostics (line 551) - merge into ROW 0 header
- Provider Backoff (standalone) - merge into ROW 4


### Step 10: Styling & Responsiveness

**File**: `static/cockpit_v2.css` (create new, clean CSS)
**Requirements**:

- Dark theme (existing color scheme)
- Grid layout (12-column system)
- Responsive breakpoints (stack columns on mobile)
- No inline styles
- Card/panel component reusable classes


### Step 11: JavaScript Refactor

**Approach**:

- Extract JS from HTML into `static/cockpit_v2.js`
- Modular functions per ROW/panel
- Unified snapshot fetcher
- Graceful degradation (missing fields → "—")
- No fake data generation


### Step 12: Testing Checklist

- [ ] Load `/cockpit_v2` route
- [ ] Verify no JS console errors
- [ ] Confirm all API calls succeed
- [ ] Check responsive layout (resize browser)
- [ ] Verify mandatory elements visible:
  - [ ] Goals panel
  - [ ] Ghost Score
  - [ ] VIP coins (WEPE, LILPEPE, DORKL, SLOTH, APC)
  - [ ] XRP tracker
  - [ ] Presale/microcap watch


---

## 📊 API Endpoint Requirements

### Existing Endpoints (Use As-Is)

- ✅ `/api/health` - System health
- ✅ `/api/cockpit` - Legacy cockpit data
- ✅ `/api/cockpit/snapshot` - Main data source
- ✅ `/api/world/context` - Macro data
- ✅ `/api/runtime/config` - Runtime config
- ✅ `/api/predict/run` - Run prediction
- ✅ `/api/predict/latest` - Get latest prediction


### New Endpoints Needed (Create If Missing)

- ⚠️ `/api/hunter/feed` - Multi-asset opportunity feed (20+ symbols)
- ⚠️ `/api/providers/health` - Provider health matrix
- ⚠️ `/api/logs/recent?limit=10` - Recent system logs
- ⚠️ `/api/accuracy/ledger` - Prediction accuracy stats
- ⚠️ `/api/vip/health` - VIP coin health data


**Fallback Strategy**: If new endpoints don't exist, extract data from `/api/cockpit/snapshot` and display available fields only.

---

## 🚨 Safety Rails (MUST PRESERVE)

### 1. No Auto-Trading Changes

- ❌ Do NOT enable `AUTO_TRADE`
- ❌ Do NOT change `SIM_MODE`
- ❌ Do NOT modify order execution logic
- ✅ UI changes ONLY


### 2. API Contract Preservation

- ❌ Do NOT break existing endpoint semantics
- ❌ Do NOT change response structures
- ✅ Only ADD new optional fields
- ✅ Gracefully handle missing fields


### 3. No Inline CSS

- ❌ No `style="..."` attributes
- ✅ All styling in `static/cockpit_v2.css`


### 4. No Hardcoded WOLF

- ❌ Remove all WOLF-specific hardcoding
- ✅ Use runtime config or API data for symbols
- ✅ Default to SPY if no symbol specified


### 5. Database Schema Safety

- ❌ Do NOT modify existing database schemas
- ✅ Create NEW tables if needed
- ✅ Use migrations for schema changes


---

## 📁 File Changes

### New Files

1. `templates/cockpit_v2.html` - New cockpit layout
2. `static/cockpit_v2.css` - New CSS (no inline styles)
3. `static/cockpit_v2.js` - New JavaScript (extracted from HTML)
4. `GHOST_COCKPIT_LAYOUT_V2.md` - Documentation


### Modified Files

1. `wolf_app.py` - Add route for `/cockpit_v2` (keeps old `/cockpit` intact)
2. `templates/cockpit.html` - Backup to `cockpit_v1_backup.html`


### Deleted Sections (within cockpit.html)

- Smart Execution panel (lines ~495-540)
- Standalone Diagnostics (lines ~551-560)
- Provider Backoff standalone panel


---

## 🎯 Success Criteria

### Must Have (Mandatory)

- [x] All mandatory elements visible:
  - Daily/weekly/monthly/yearly goals
  - Real-time Ghost Score
  - VIP coins: WEPE, LILPEPE, DORKL, SLOTH, APC
  - XRP tracker
  - Presale/microcap watch
- [ ] No WOLF hardcoding
- [ ] No JavaScript console errors
- [ ] All API calls succeed (with graceful fallbacks)
- [ ] Responsive layout (mobile + desktop)


### Should Have (Important)

- [ ] Multi-asset Hunter Feed (20+ symbols)
- [ ] Provider Health Matrix
- [ ] Clean separation: HTML/CSS/JS
- [ ] Performance: <2s initial load


### Nice to Have (Enhancements)

- [ ] Real-time updates (WebSocket)
- [ ] Chart integration (TradingView)
- [ ] Export/download data
- [ ] Dark/light theme toggle


---

## 📝 Next Actions

1. ✅ Create this plan document
2. ⏳ Backup current cockpit → `cockpit_v1_backup.html`
3. ⏳ Create `cockpit_v2.html` skeleton
4. ⏳ Implement ROW 0 (Header)
5. ⏳ Implement ROW 1 (Goals + Hunter + VIP)
6. ⏳ Implement ROW 2 (Macro + Risk + Portfolio)
7. ⏳ Implement ROW 3 (Predictions + AI + Accuracy)
8. ⏳ Implement ROW 4 (Providers + Logs + Config)
9. ⏳ Create `cockpit_v2.css`

1. ⏳ Extract JS → `cockpit_v2.js`
2. ⏳ Add route in `wolf_app.py`
3. ⏳ Test & validate


---

**Status**: 📋 Plan Complete - Ready for Implementation
**Estimated Time**: 4-6 hours (systematic rebuild)
**Risk Level**: 🟢 LOW (additive, non-breaking)
