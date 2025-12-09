# Ghost Cockpit V3 - Quick Fix Guide

**Priority Order for Implementation**

---

## 🔴 CRITICAL FIX #1: VIP Panel (30 min)

### Problem

Shows BTC/ETH/SOL/BNB/XRP (all offline) instead of VIP sniper coins + XRP tracker

### Files to Change

- `templates/cockpit_v3.html` (lines 60-67)
- `static/cockpit_v3.js` (lines 282-342)

### Steps

1. Add 3 new div containers to HTML: `xrp-tracker`, `vip-sniper-list`, `vip-majors-list`
2. Replace `loadVIPCoins()` function to call 3 endpoints:
   - `/api/xrp/tracker` → XRP bullish eye
   - `/api/presale/watch` → WEPE/LILPEPE/etc
   - `/api/v3/vip/snapshot` → BTC/ETH reference
3. Add render functions: `renderXRPTracker()`, `renderVIPSniperCoins()`, `renderMajorCaps()`

### Code Snippet

```javascript
async function loadVIPCoins() {
    // XRP Tracker
    const xrpResp = await fetch('/api/xrp/tracker');
    if (xrpResp.ok) renderXRPTracker(await xrpResp.json());

    // VIP Sniper
    const presaleResp = await fetch('/api/presale/watch');
    if (presaleResp.ok) {
        const data = await presaleResp.json();
        renderVIPSniperCoins(data.presales || []);
    } else {
        renderVIPSniperPlaceholder();  // Show WEPE/LILPEPE placeholder
    }

    // Majors (BTC/ETH)
    const majorsResp = await fetch('/api/v3/vip/snapshot');
    if (majorsResp.ok) renderMajorCaps(await majorsResp.json().vip_coins);
}

```text

---

## 🟠 HIGH FIX #2: Status Indicator (10 min)

### Problem

Status dot hidden, no "RUNNING"/"STOPPED" text

### File to Change

- `static/cockpit_v3.js` (lines 25, 117-128)


### Steps

1. Add `loadCockpitStatus()` call in `initializeApp()`
2. Make status dot visible in `updateStatusIndicator()`
3. Change text to "RUNNING" / "STOPPED" (not "LIVE")


### Code Snippet

```javascript

// In initializeApp():
loadCockpitStatus();
setInterval(() => loadCockpitStatus(), 30000);

// New function:
async function loadCockpitStatus() {
    const resp = await fetch('/api/v3/cockpit/status');
    if (resp.ok) {
        const data = await resp.json();
        updateStatusIndicator(data.active !== false);
    }
}

// Update updateStatusIndicator:
function updateStatusIndicator(isActive) {
    document.getElementById('status-indicator').style.display = 'inline-block';
    document.getElementById('status-text').textContent = isActive ? 'RUNNING' : 'STOPPED';
    // ... colors
}

```text

---

## 🟠 HIGH FIX #3: Health Metrics (45 min)

### Problem

Data Health, AI Activity, Accuracy hard-coded (85/75/70)

### Files to Change

- `wolf_app.py` (add endpoint after line 7310)
- `static/cockpit_v3.js` (lines 702-730, 740-770)


### Steps

1. Create `/api/v3/health/metrics` endpoint in backend
2. Calculate real values:
   - Data Health: provider uptime test
   - AI Activity: predictions per hour
   - Accuracy: win rate from prediction store
1. Update `loadHealthScore()` to fetch metrics
2. Update `renderHealthMetrics()` to use passed values (remove hard-codes)


### Backend Code

```python

@APP.get("/api/v3/health/metrics")
async def api_v3_health_metrics():

    # Test provider

    data_health = 100.0 if test_provider_btc() else 50.0

    # Count predictions/hour

    recent_preds = get_prediction_store().get_recent_predictions(hours=1)
    ai_activity = min(100.0, len(recent_preds) * 5.0)

    # Get win rate

    stats = get_prediction_store().get_accuracy_stats(days=7)
    accuracy = stats.get("win_rate", 0.7) * 100

    return {
        "ok": True,
        "data_health": round(data_health, 1),
        "ai_activity": round(ai_activity, 1),
        "accuracy": round(accuracy, 1)
    }

```text

### Frontend Code

```javascript

async function loadHealthScore() {
    const metricsResp = await fetch('/api/v3/health/metrics');
    const metricsData = await metricsResp.json();

    renderHealthMetrics({
        daily: goalsData.daily_goal_pct,
        weekly: goalsData.weekly_goal_pct,
        monthly: goalsData.monthly_goal_pct,
        data_health: metricsData.data_health,  // REAL
        ai_activity: metricsData.ai_activity,  // REAL
        accuracy: metricsData.accuracy         // REAL
    });
}

```text

---

## 🟡 MEDIUM FIX #4: Forecast Label (5 min)

### Problem

No symbol label, no loading/error states

### Files to Change

- `templates/cockpit_v3.html` (line 72)
- `static/cockpit_v3.js` (line 348)


### Steps

1. Add `<span id="forecast-symbol-label">` after input
2. Update `loadForecast()` to set label text:
   - Before fetch: "Loading BTC..."
   - On success: "Forecast for BTC"
   - On error: "❌ BTC unavailable"


### Code Snippet

```javascript

async function loadForecast() {
    const label = document.getElementById('forecast-symbol-label');
    label.textContent = `Loading ${currentForecastSymbol}...`;
    label.style.color = 'var(--accent-orange)';

    try {
        const resp = await fetch(`/api/v3/predictions/latest?symbol=${currentForecastSymbol}`);
        // ... update cards
        label.textContent = `Forecast for ${currentForecastSymbol}`;
        label.style.color = 'var(--accent-green)';
    } catch (error) {
        label.textContent = `❌ ${currentForecastSymbol} unavailable`;
        label.style.color = 'var(--accent-red)';
    }
}

```text

---

## 🟡 MEDIUM FIX #5: Watchlist Types (TBD)

### Problem

All entries show "STOCK --" even for crypto

### Investigation Required

```bash

# Test API response

curl "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/enriched">>>>> | jq '.items[0]'

# Check if API returns correct fields

# - type: "crypto" (not "stock")

# - price: 92468.00 (not 0 or null)

# - ghost_confidence: 46.0

# - predicted_direction: "UP"

```text

### If API Wrong

Fix backend endpoint `/api/v3/watchlist/enriched` in `wolf_app.py`

### If Frontend Wrong

Update `renderWatchlist()` field mappings in `static/cockpit_v3.js` (lines 668-700)

---

## USER TESTS REQUIRED

Before deploying, user MUST test:

1. **START/STOP/RESET**→ Does status change?


2.**LIVE/FIXED toggle**→ Does anything change?
3.**Top Movers tabs**→ Do ticker lists filter?
4.**Forecast input BTC**→ Do values update?
5.**News refresh ↻**→ Do entries update?
6.**Watchlist: Add/Own/Remove**→ Does it persist?
7.**Goals: Save $123/$456**→ Do percentages update?


---

## DEPLOYMENT CHECKLIST

- [ ] Backup `cockpit_v3.js` and `cockpit_v3.html`
- [ ] Apply all fixes
- [ ] Test locally (no console errors)
- [ ] Deploy to Railway
- [ ] Hard refresh browser (Cmd+Shift+R)
- [ ] Verify XRP tracker loads
- [ ] Verify status dot visible
- [ ] Verify health metrics change over time
- [ ] Run all 7 user tests above


---

## ESTIMATED TIME

- Critical Fix #1 (VIP):**30 min**- High Fix #2 (Status):**10 min**- High Fix #3 (Health):**45 min**- Medium Fix #4 (Forecast):**5 min**- Testing:**20 min**


**Total: ~2 hours**for critical + high priority fixes

---

## SUCCESS CRITERIA**Before:**- VIP panel: All offline

- Status: Dot hidden
- Health: 85/75/70 static
- Forecast: No label**After:**- VIP panel: XRP tracker + sniper coins + majors
- Status: Green dot + "RUNNING"
- Health: Real values from APIs
- Forecast: "Forecast for BTC" label**Score:** 4/10 → 8/10


---

For detailed code patches, see: `COCKPIT_DIAGNOSTIC_COMPLETE.md`
