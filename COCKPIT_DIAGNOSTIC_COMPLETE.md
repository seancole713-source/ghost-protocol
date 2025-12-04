# Ghost Protocol Cockpit V3 - Diagnostic & Fix Plan
**Generated:** December 4, 2025

---

## EXECUTIVE SUMMARY

**Stack:** Jinja2 (`templates/cockpit_v3.html`) + Vanilla JS (`static/cockpit_v3.js`, `static/personal_watchlist_ui.js`)

**Scores:**
- Layout: ✅ 8/10 (all panels present)
- Live Wiring: ⚠️ 4/10 (only movers/news/forecast partially work)
- Baseline Compliance: ❌ 2/10 (missing VIP sniper coins, XRP tracker, presale radar)

**Critical Issues:**
1. ❌ VIP panel shows BTC/ETH/SOL/BNB/XRP (all offline) - should show WEPE/LILPEPE/DORKL/SLOTH/APC
2. ❌ XRP tracker API exists but not exposed in UI
3. ❌ Presale awareness completely missing
4. ❌ Watchlist shows "STOCK --" for everything (wrong types, no data)
5. ⚠️ Health metrics hard-coded (85/75/70 static values)
6. ⚠️ START/STOP buttons work but no visible status

---

## DETAILED FINDINGS

### 1. GLOBAL CONTROLS

**Files:** `templates/cockpit_v3.html` (lines 10-38), `static/cockpit_v3.js` (lines 39-62)

**What Works:**
- ✅ System clock updates every 1s
- ✅ START/STOP/RESET call `/api/cockpit/{action}` correctly
- ✅ Backend endpoints update `STATE["active"]`

**Issues:**
- ❌ Status dot hidden (no initial `updateStatusIndicator()` call)
- ❌ Text never shows "RUNNING" / "STOPPED" explicitly
- ❌ LIVE/FIXED toggle has no effect (just console.log)

**Fix:**
```javascript
// In initializeApp() - ADD:
loadCockpitStatus();  // Load status immediately
setInterval(() => loadCockpitStatus(), 30000);

// NEW function:
async function loadCockpitStatus() {
    const response = await fetch('/api/v3/cockpit/status');
    if (response.ok) {
        const data = await response.json();
        updateStatusIndicator(data.active !== false);
    }
}

// UPDATE updateStatusIndicator:
function updateStatusIndicator(isActive) {
    const dot = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');
    dot.style.display = 'inline-block';  // Make visible
    if (isActive) {
        dot.style.background = 'var(--accent-green)';
        text.textContent = 'RUNNING';  // Explicit
        text.style.color = 'var(--accent-green)';
    } else {
        dot.style.background = 'var(--accent-red)';
        text.textContent = 'STOPPED';
        text.style.color = 'var(--accent-red)';
    }
}
```

---

### 2. TOP MOVERS

**Files:** `wolf_app.py` (7363-7450), `static/cockpit_v3.js` (219-280)

**What Works:**
- ✅ Loads from `/api/v3/hunter/feed`
- ✅ Shows symbols with % change and Ghost confidence
- ✅ Tabs render (Stocks/Crypto/All)

**Unknown (Need User Test):**
- ❓ Do tabs actually filter? (click Stocks → Crypto → All and confirm ticker lists change)
- ❓ Why all confidences 58-59%? (may be real, but suspicious)

---

### 3. VIP COINS ⚠️ CRITICAL

**Current:**
```python
# wolf_app.py line 1312
VIP_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP"]  # Wrong!
# Comment says: "presale coins unavailable on exchanges"
```

**Problem:**
- Shows BTC/ETH/SOL/BNB/XRP (all return `price: 0`, `status: offline`)
- Should show: WEPE, LILPEPE, DORKL, SLOTH, APC
- XRP should have dedicated "bullish eye" tracker (backend exists: `/api/xrp/tracker`)

**Backend APIs Available:**
- ✅ `/api/xrp/tracker` - XRP bullish eye (exists in `wolf_app.py` line 24553)
- ✅ `/api/presale/watch` - Presale coins (exists in `api/cockpit_v2_endpoints.py` line 140)
- ✅ `core/xrp_tracker.py` - Full XRP analysis module
- ✅ `core/vip_scanner.py` - VIP coin scanner (tracks WEPE/LILPEPE/etc)

**Fix Plan:**

**Step 1:** Restructure VIP panel HTML
```html
<!-- templates/cockpit_v3.html - REPLACE lines 60-67 -->
<section class="panel" id="panel-vip">
    <div class="panel-header"><h2>🌟 VIP Watch</h2></div>
    <div class="panel-body">
        <!-- XRP Bullish Eye (Always Visible) -->
        <div id="xrp-tracker" style="margin-bottom: 20px;"></div>
        
        <!-- VIP Sniper Coins -->
        <div style="margin-bottom: 20px;">
            <h3 style="font-size: 14px; color: #f39c12;">🎯 VIP Sniper Coins</h3>
            <div id="vip-sniper-list"></div>
        </div>
        
        <!-- Major Caps (Reference) -->
        <div style="border-top: 1px solid var(--border); padding-top: 15px;">
            <h3 style="font-size: 14px; color: var(--text-secondary);">📊 Major Caps</h3>
            <div id="vip-majors-list" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;"></div>
        </div>
    </div>
</section>
```

**Step 2:** Wire to APIs
```javascript
// static/cockpit_v3.js - REPLACE loadVIPCoins (lines 282-342)
async function loadVIPCoins() {
    // Load XRP Tracker
    try {
        const xrpResponse = await fetch('/api/xrp/tracker');
        if (xrpResponse.ok) {
            renderXRPTracker(await xrpResponse.json());
        }
    } catch (error) {
        renderXRPTrackerError(error.message);
    }
    
    // Load VIP Sniper Coins
    try {
        const presaleResponse = await fetch('/api/presale/watch');
        if (presaleResponse.ok) {
            const data = await presaleResponse.json();
            renderVIPSniperCoins(data.presales || []);
        } else {
            renderVIPSniperPlaceholder();
        }
    } catch (error) {
        renderVIPSniperPlaceholder();
    }
    
    // Load Major Caps (BTC/ETH for reference)
    try {
        const majorsResponse = await fetch('/api/v3/vip/snapshot');
        if (majorsResponse.ok) {
            const data = await majorsResponse.json();
            renderMajorCaps(data.vip_coins || []);
        }
    } catch (error) {
        console.error('[VIP] Majors error:', error);
    }
}

function renderXRPTracker(data) {
    const container = document.getElementById('xrp-tracker');
    const eye = data.bullish_eye || '🟡';
    const signal = data.signal || 'WAIT';
    const price = data.price ? `$${data.price.toFixed(4)}` : 'Loading...';
    const change = data.change_24h_pct;
    const signalColor = signal === 'BUY' ? 'var(--accent-green)' : 
                       signal === 'SELL' ? 'var(--accent-red)' : 'var(--accent-orange)';
    
    container.innerHTML = `
        <div style="background: linear-gradient(135deg, #1a1a1a, #2a2a2a); 
                    border: 2px solid #f39c12; padding: 20px; border-radius: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 28px;">${eye} XRP Bullish Eye</div>
                    <div style="font-size: 16px; color: var(--text-secondary); margin-top: 5px;">
                        ${price}
                        ${change !== null ? `<span style="color: ${change >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}"> (${change >= 0 ? '+' : ''}${change.toFixed(2)}%)</span>` : ''}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px; font-weight: 700; color: ${signalColor};">${signal}</div>
                    <div style="font-size: 13px; margin-top: 5px; color: var(--text-secondary);">
                        ${((data.confidence || 0) * 100).toFixed(0)}% confidence
                    </div>
                </div>
            </div>
            ${data.factors && data.factors.length > 0 ? `
                <div style="font-size: 12px; color: var(--text-secondary); border-top: 1px solid #333; padding-top: 10px; margin-top: 10px;">
                    ${data.factors.map(f => `<div>• ${f}</div>`).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

function renderVIPSniperCoins(coins) {
    const container = document.getElementById('vip-sniper-list');
    if (!coins || coins.length === 0) {
        renderVIPSniperPlaceholder();
        return;
    }
    container.innerHTML = coins.map(coin => `
        <div style="background: #1e1e1e; border: 1px solid #444; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <div style="font-weight: 600;">${coin.name}</div>
                    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 5px;">
                        ${coin.description || 'Presale coin'}
                    </div>
                </div>
                <div style="padding: 4px 12px; border-radius: 4px; 
                            background: ${coin.status === 'Active' ? 'var(--accent-green)' : 'var(--accent-orange)'}; 
                            color: white; height: fit-content;">
                    ${coin.status}
                </div>
            </div>
        </div>
    `).join('');
}

function renderVIPSniperPlaceholder() {
    document.getElementById('vip-sniper-list').innerHTML = `
        <div style="background: #1e1e1e; border: 2px dashed #555; padding: 20px; border-radius: 8px; text-align: center;">
            <div style="font-size: 16px; margin-bottom: 10px; color: #f39c12;">🎯 VIP Sniper Watchlist</div>
            <div style="color: var(--text-secondary); margin-bottom: 10px;">WEPE • LILPEPE • DORKL • SLOTH • APC</div>
            <div style="font-size: 12px; color: #888;">⚠️ Presale data unavailable - monitoring placeholder</div>
        </div>
    `;
}

function renderMajorCaps(coins) {
    const majors = coins.filter(c => ['BTC', 'ETH'].includes(c.symbol));
    document.getElementById('vip-majors-list').innerHTML = majors.map(coin => {
        const offline = coin.price === 0;
        return `
            <div style="background: #1a1a1a; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="font-weight: 600;">${coin.symbol}</div>
                <div style="font-size: 12px; color: var(--text-secondary);">
                    ${offline ? '--' : `$${coin.price.toLocaleString()}`}
                </div>
                <div style="font-size: 11px; color: ${coin.change_pct >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'};">
                    ${offline ? '--' : `${coin.change_pct >= 0 ? '+' : ''}${coin.change_pct.toFixed(2)}%`}
                </div>
            </div>
        `;
    }).join('');
}
```

---

### 4. GHOST FORECAST

**Files:** `static/cockpit_v3.js` (344-438)

**What Works:**
- ✅ Input triggers `loadForecast()` on change
- ✅ Backend returns real predictions
- ✅ Three time buckets with decay multipliers (1.0/0.7/0.5)

**Issues:**
- ❌ No symbol label shown (user can't see what forecast is for)
- ❌ No loading state
- ❌ No error handling for invalid symbols

**Fix:**
```html
<!-- templates/cockpit_v3.html line 72 - ADD after input -->
<span id="forecast-symbol-label" style="margin-left: 10px; font-weight: 600; color: var(--accent-green);"></span>
```

```javascript
// static/cockpit_v3.js - UPDATE loadForecast (line 348)
async function loadForecast() {
    const labelEl = document.getElementById('forecast-symbol-label');
    if (labelEl) {
        labelEl.textContent = `Loading ${currentForecastSymbol}...`;
        labelEl.style.color = 'var(--accent-orange)';
    }
    
    try {
        const response = await fetch(`/api/v3/predictions/latest?symbol=${currentForecastSymbol}`);
        if (!response.ok) throw new Error('API failed');
        
        const data = await response.json();
        const predictions = data.predictions || [];
        const pred = predictions[0] || {};
        
        // Update cards...
        updateForecastCard(0, pred, '☀️', '24h', 1.0);
        updateForecastCard(1, pred, '⛅', '2-5d', 0.7);
        updateForecastCard(2, pred, '🌤️', '7-14d', 0.5);
        
        // Success label
        if (labelEl) {
            labelEl.textContent = `Forecast for ${currentForecastSymbol}`;
            labelEl.style.color = 'var(--accent-green)';
        }
    } catch (error) {
        console.error('[FORECAST] Error:', error);
        
        // Error label
        if (labelEl) {
            labelEl.textContent = `❌ ${currentForecastSymbol} unavailable`;
            labelEl.style.color = 'var(--accent-red)';
        }
        
        // Show empty state
        for (let i = 0; i < 3; i++) {
            updateForecastCard(i, {direction: 'FLAT', confidence: 0, expected_move: 0}, 
                             ['☀️', '⛅', '🌤️'][i], ['24h', '2-5d', '7-14d'][i], 1.0);
        }
    }
}
```

---

### 5. WATCHLIST ⚠️ CRITICAL

**Files:** `static/cockpit_v3.js` (600-700), `static/personal_watchlist_ui.js` (1-572)

**Problem:**
- All entries show "STOCK --"
- Even crypto (XRP, BTC, DOGE) labeled "STOCK"
- No prices (all "--")
- All signals "FLAT"

**Code Analysis:**
```javascript
// static/cockpit_v3.js lines 668-700 - Field mappings CORRECT:
const priceDisplay = item.price ? `$${item.price.toFixed(2)}` : '--';
const scoreDisplay = item.ghost_confidence > 0 ? `${item.ghost_confidence.toFixed(0)}%` : '--';
const direction = item.predicted_direction || 'FLAT';
```

**Root Cause:** API returning empty/null data OR wrong field names

**Investigation Required:**
```bash
# Test API response:
curl "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/enriched" | jq '.items[0]'

# Expected:
{
  "symbol": "BTC",
  "type": "crypto",  # NOT "stock"
  "price": 92468.00,
  "ghost_confidence": 46.0,
  "predicted_direction": "UP"
}
```

**If API wrong:** Fix backend endpoint `@APP.get("/api/v3/watchlist/enriched")`  
**If frontend wrong:** Update field mappings in `renderWatchlist()`

**User Test:** Add symbol, mark owned, reload → confirm persistence

---

### 6. HEALTH SCORE ⚠️ STATIC VALUES

**Files:** `static/cockpit_v3.js` (702-780)

**Current:**
- Ghost Health Score: 100 (Grade A)
- Data Health: **85** (hard-coded line 750)
- AI Activity: **75** (hard-coded line 751)
- Accuracy: **70** (hard-coded line 752)

**Fix:** Create real metrics endpoint

```python
# wolf_app.py - ADD after line 7310:
@APP.get("/api/v3/health/metrics")
async def api_v3_health_metrics():
    """Real Ghost health metrics (not static)"""
    try:
        # 1. Data Health: Provider availability
        data_health = 75.0
        try:
            from core.crypto.crypto_providers import get_crypto_price_quorum
            btc_test = await get_crypto_price_quorum("BTC", use_cache=True)
            data_health = 100.0 if btc_test and btc_test.get("price") else 50.0
        except Exception:
            data_health = 25.0
        
        # 2. AI Activity: Predictions per hour
        ai_activity = 0.0
        try:
            from core.prediction_store import get_prediction_store
            store = get_prediction_store()
            recent = len(store.get_recent_predictions(hours=1))
            ai_activity = min(100.0, recent * 5.0)  # 20/hr = 100%
        except Exception:
            pass
        
        # 3. Accuracy: Win rate
        accuracy = 70.0
        try:
            from core.prediction_store import get_prediction_store
            store = get_prediction_store()
            stats = store.get_accuracy_stats(days=7)
            if stats and "win_rate" in stats:
                accuracy = round(stats["win_rate"] * 100, 1)
        except Exception:
            pass
        
        return {
            "ok": True,
            "data_health": round(data_health, 1),
            "ai_activity": round(ai_activity, 1),
            "accuracy": round(accuracy, 1),
            "timestamp": time.time()
        }
    except Exception as e:
        LOGGER.error(f"Health metrics failed: {e}")
        return {"ok": False, "data_health": 0, "ai_activity": 0, "accuracy": 0, "error": str(e)}
```

```javascript
// static/cockpit_v3.js - UPDATE loadHealthScore (lines 702-730)
async function loadHealthScore() {
    try {
        // Load goals
        const goalsResp = await fetch('/api/v3/goals/snapshot');
        const goalsData = await goalsResp.json();
        
        // Load REAL metrics
        const metricsResp = await fetch('/api/v3/health/metrics');
        let metricsData = { data_health: 75, ai_activity: 75, accuracy: 70 };  // Fallback
        if (metricsResp.ok) {
            metricsData = await metricsResp.json();
            console.log('[HEALTH] Real metrics:', metricsData);
        }
        
        // Update score
        const score = goalsData.ghost_score || 0;
        document.getElementById('health-score-value').textContent = score.toFixed(0);
        document.getElementById('health-grade').textContent = calculateGrade(score);
        
        // Pass REAL values to renderHealthMetrics
        renderHealthMetrics({
            daily: goalsData.daily_goal_pct || 0,
            weekly: goalsData.weekly_goal_pct || 0,
            monthly: goalsData.monthly_goal_pct || 0,
            data_health: metricsData.data_health || 75,
            ai_activity: metricsData.ai_activity || 75,
            accuracy: metricsData.accuracy || 70
        });
    } catch (error) {
        console.error('[HEALTH] Error:', error);
    }
}

// UPDATE renderHealthMetrics (lines 740-770) - remove hard-coded values:
function renderHealthMetrics(metrics) {
    const metricsList = [
        { name: 'Daily Goal', value: metrics.daily },
        { name: 'Weekly Goal', value: metrics.weekly },
        { name: 'Monthly Goal', value: metrics.monthly },
        { name: 'Data Health', value: metrics.data_health },  // REAL
        { name: 'AI Activity', value: metrics.ai_activity },  // REAL
        { name: 'Accuracy', value: metrics.accuracy }         // REAL
    ];
    // ... render bars
}
```

---

## BASELINE ALIGNMENT GAPS

| Feature | Required | Current | Priority |
|---------|----------|---------|----------|
| **VIP Sniper Coins** | WEPE/LILPEPE/DORKL/SLOTH/APC visible | BTC/ETH/SOL/BNB/XRP (all offline) | 🔴 CRITICAL |
| **XRP Bullish Eye** | Dedicated tracker | In generic list, no widget | 🔴 CRITICAL |
| **Presale Awareness** | "Strike Prep" surface | Missing | 🔴 CRITICAL |
| **Goals ↔ P&L** | Real progress | Static percentages | 🟠 HIGH |
| **Real Ghost Score** | Accuracy + quality | Hard-coded 100 | 🟠 HIGH |
| **Status Visibility** | "Running"/"Stopped" | Dot hidden | 🟡 MEDIUM |

---

## USER ACTION ITEMS

**Test These NOW:**

1. START/STOP/RESET → Confirm status text changes
2. LIVE/FIXED toggle → Confirm ANY UI change
3. Top Movers tabs → Confirm ticker lists change
4. Forecast input "BTC" → Confirm values update
5. News refresh ↻ → Confirm new entries appear
6. Watchlist: Add "TEST", mark owned, reload → Confirm persists
7. Goals: Set $123/$456, save, reload → Confirm percentages change

**For EACH test report:**
- ✅ WORKS: What changed
- ❌ BROKEN: What didn't work
- ❓ UNKNOWN: Couldn't tell

---

## FILES TO MODIFY

### Phase 1: Critical UI (2-4 hours)
1. `templates/cockpit_v3.html` - VIP panel structure
2. `static/cockpit_v3.js` - VIP functions, status, forecast label

### Phase 2: Backend Metrics (3-5 hours)
3. `wolf_app.py` - `/api/v3/health/metrics` endpoint

### Phase 3: Presale Surface (2-3 hours)
4. `templates/cockpit_v3.html` - Presale panel (optional)
5. `wolf_app.py` - Enhance `/api/presale/watch`

---

## EXPECTED OUTCOMES

**After Fixes:**

✅ **VIP Panel:**
- XRP tracker visible with 🟢🟡🔴 eye
- XRP shows price, change, BUY/HOLD/SELL signal
- VIP Sniper section shows WEPE/LILPEPE/etc (even if placeholder)
- Major caps (BTC/ETH) below for reference

✅ **Status:**
- Dot visible, colored green/red
- Text shows "RUNNING" / "STOPPED"

✅ **Health:**
- Data Health = provider uptime (not 85)
- AI Activity = predictions/hour (not 75)
- Accuracy = real win rate (not 70)

✅ **Forecast:**
- Label shows "Forecast for BTC"
- Loading/error states work

**Score Improvement:** 4/10 → 8/10

---

**Next Review:** After Phase 1 implementation  
**Questions:** Direct to Ghost Commander
