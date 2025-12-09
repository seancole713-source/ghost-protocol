# Cockpit V3 Remaining Patches (8/12)

**Status:**4/12 CRITICAL patches applied ✅**Commit:**4ff519b**Date:**December 4, 2025

---

## ✅ COMPLETED (4/12)

1. ✅**Watchlist ghost_direction field**- Fixed (was using wrong field name)
2. ✅**VIP sniper coins (5 total)**- Fixed (added DORKL, SLOTH, APC)
3. ✅**Forecast input sync**- Fixed (now shows default symbol)
4. ✅**Market watchlist endpoint**- Implemented (`/api/v3/watchlist/market`)

---

## 🟡 HIGH PRIORITY (4/8 remaining)

### Patch 5: Wire XRP Confidence to Prediction Engine**Priority:**🔴 HIGH**File:**`core/xrp_tracker.py`**Issue:**XRP tracker returns `confidence: 0.0` always (not connected to predictions)**Implementation:**

```python

# In get_xrp_tracker_data() function (around line 85)

def get_xrp_tracker_data():

    # ... existing bullish_eye calculation

    # NEW: Get XRP prediction confidence

    from wolf_app import _LATEST_PREDICTIONS
    xrp_pred = _LATEST_PREDICTIONS.get("XRP", {})

    # Convert confidence from 0-1 scale to 0-100 percentage

    confidence = xrp_pred.get("confidence", 0)
    if 0 < confidence <= 1:
        confidence = confidence * 100

    # Calculate 24h price change (add logic here)

    change_24h = calculate_24h_change(xrp_price)  # Implement this

    # Return numeric eye score alongside emoji

    bullish_eye_numeric = calculate_numeric_eye_score(...)  # Implement this

    return {
        "ok": True,
        "price": xrp_price,
        "change_24h_pct": change_24h,  # Real 24h delta
        "bullish_eye_score": bullish_eye_numeric,  # Numeric 0-100
        "bullish_eye": bullish_eye_emoji,  # Keep emoji
        "signal": signal,
        "confidence": confidence,  # Real confidence from predictions
        "factors": factors,
        "timestamp": time.time()
    }

```text

**Test:**```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/xrp/tracker>>>>> | jq '.confidence'

# Should return: >0 (not 0.0)

```text

---

### Patch 6: Implement LIVE/FIXED Mode Toggle**Priority:**🔴 HIGH**Files:**`wolf_app.py` + `static/cockpit_v3.js`**Issue:**Mode selector is cosmetic only (no backend effect)**Backend Implementation:**```python

# Add after /api/cockpit/reset endpoint (line ~7428)

@APP.post("/api/cockpit/mode")
async def api_cockpit_mode(request: Request):
    """Toggle between LIVE (real-time) and FIXED (snapshot) mode"""
    data = await request.json()
    mode = data.get("mode", "live")  # "live" or "fixed"

    STATE["cockpit_mode"] = mode
    _add_event("control", f"Mode changed to {mode}", {"mode": mode})

    return {"ok": True, "mode": mode}

```text**Frontend Implementation:**```javascript

// Modify handleModeChange() in cockpit_v3.js (line ~126)

async function handleModeChange(e) {
    const mode = e.target.value;
    console.log('Mode changed to:', mode);

    // POST to backend
    try {
        const response = await fetch('/api/cockpit/mode', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({mode})
        });
        if (response.ok) {
            // Reload all panels with new mode
            loadAllPanels();
        }
    } catch (error) {
        console.error('Mode change failed:', error);
    }
}

```text**Test:**1. Toggle LIVE → FIXED in UI

1. Verify console shows POST request
2. Verify panels reload


---

### Patch 7: Debug Top Movers Empty Display**Priority:**🔴 HIGH**Files:**`static/cockpit_v3.css` (investigate), `static/cockpit_v3.js` (add logging)**Issue:**Data loads correctly (`/api/v3/hunter/feed` returns movers) but DOM shows empty container**Investigation Steps:**```javascript

// Add to loadTopMovers() function (line ~250)
console.log('[TOP MOVERS DEBUG] Fetched:', movers.length, 'movers');
console.log('[TOP MOVERS DEBUG] Filtered:', filtered.length, 'movers');
console.log('[TOP MOVERS DEBUG] Container innerHTML length:', container.innerHTML.length);
console.log('[TOP MOVERS DEBUG] Container visible?', window.getComputedStyle(container).display);

```text**Check CSS:**

```css

/*In cockpit_v3.css - look for these rules*/
.mover-card {
    display: none; /*❌ If this exists, remove it*/
}

.movers-grid {
    height: 0; /*❌ If this exists, fix it*/
    overflow: hidden; /*❌ Check if cutting off content*/
}

.panel-body {
    max-height: 300px; /*⚠️  Might be too small*/
    overflow-y: auto; /*✅ Should be present*/
}

```text

**Likely Fix:**- Remove `display: none` from `.mover-card`

- Increase `.panel-body` min-height
- Ensure `.movers-grid` has proper grid layout


---

### Patch 8: Fix News Feed Sentiment Display**Priority:**🟡 MEDIUM**File:**`static/cockpit_v3.js`**Issue:**All news items show "Neutral" even though API returns "bullish"/"bearish"**Investigation:**```javascript

// In loadNews() function (line ~545-587)
// Check if renderNewsItem() uses item.sentiment field

function renderNewsItem(item) {
    // WRONG (if this exists):
    const sentiment = 'Neutral';  // Hard-coded

    // CORRECT:
    const sentiment = item.sentiment || 'Neutral';
    const sentimentClass = sentiment === 'bullish' ? 'positive' :
                           sentiment === 'bearish' ? 'negative' : 'neutral';

    // Use sentimentClass in rendering
}

```text

---

## 🟢 MEDIUM PRIORITY (4/8 remaining)

### Patch 9: Show Ghost Signals on Major Caps**Enhancement:**Overlay BUY/SELL/WAIT badges on BTC/ETH cards**Implementation:**

```javascript

// Modify renderMajorCaps() in cockpit_v3.js (line ~410-437)

function renderMajorCaps(coins) {
    container.innerHTML = coins.map(coin => {
        // NEW: Fetch prediction for this symbol
        const pred = _LATEST_PREDICTIONS.get(coin.symbol, {});
        const signal = pred.direction || 'HOLD';
        const confidence = (pred.confidence || 0) * 100;

        return `
            <div style="...">
                <div>...</div>
                <!-- NEW: Add Ghost signal badge -->
                <div style="font-size: 11px; color: var(--accent-green);">
                    Ghost: ${signal} (${confidence.toFixed(0)}%)
                </div>
            </div>
        `;
    }).join('');
}

```text

---

### Patch 10: Presale Radar Block (New Feature)

**Enhancement:**Dedicated presale awareness surface**Template Addition (cockpit_v3.html):**```html

<!-- After VIP Sniper Coins section -->
<div style="border-top: 1px solid var(--border); padding-top: 15px;">
    <h3>🎯 Presale Radar</h3>
    <div id="presale-radar-list"></div>
</div>

```text**New Render Function:**```javascript

function renderPresaleRadar(coins) {
    // Show countdown, hard cap progress, Ghost risk score, strike window
    // Requires enhanced presale API with metadata
}

```text

---

### Patch 11: Prediction Accuracy Placeholder**Enhancement:**Show message when no data available**Implementation:**```javascript

// In renderAccuracyChart() when no data (line ~615)

if (!accuracyData || !accuracyData.daily_accuracy_pct) {
    ctx.fillStyle = 'var(--text-secondary)';
    ctx.font = '14px var(--font-mono)';
    ctx.textAlign = 'center';
    ctx.fillText('⏳ Accuracy tracking begins after first 48h prediction window',
                 rect.width / 2, rect.height / 2 - 10);
    ctx.fillText('Check back soon!', rect.width / 2, rect.height / 2 + 10);
    return;
}

```text

---

### Patch 12: Watchlist Action Buttons**Enhancement:**Wire up ➕ (mark owned), 📊 (history), ✖ (remove)**Implementation:**```javascript

// Add event listeners in setupEventListeners()
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('watchlist-btn-own')) {
        const symbol = e.target.dataset.symbol;
        markAsOwned(symbol);
    } else if (e.target.classList.contains('watchlist-btn-history')) {
        const symbol = e.target.dataset.symbol;
        showPredictionHistory(symbol);
    } else if (e.target.classList.contains('watchlist-btn-remove')) {
        const symbol = e.target.dataset.symbol;
        removeFromWatchlist(symbol);
    }
});

```text

---

## Deployment Sequence

### Phase 1 (Immediate - Critical Fixes)

1. Deploy Patch 5 (XRP confidence)
2. Deploy Patch 6 (LIVE/FIXED mode)
3. Deploy Patch 7 (Top movers render)
4. Test all 4 completed patches in production


### Phase 2 (Next Sprint - Enhancements)

1. Deploy Patch 8 (News sentiment)
2. Deploy Patch 9 (Major caps signals)
3. Deploy Patch 10 (Presale radar)


### Phase 3 (Future - Polish)

1. Deploy Patch 11 (Accuracy placeholder)
2. Deploy Patch 12 (Watchlist actions)


---

## Regression Test Suite

After each deployment:

```bash

# 1. Watchlist shows correct data

curl /api/v3/watchlist/user | jq '.items[0] | {symbol, ghost_direction, type}'

# Expected: {symbol: "DOT", ghost_direction: "UP", type: "crypto"}

# 2. Market watchlist exists

curl /api/v3/watchlist/market | jq '.ok'

# Expected: true

# 3. VIP sniper has 5 coins

curl /api/presale/watch | jq '.presales | length'

# Expected: 5

# 4. XRP confidence is real

curl /api/xrp/tracker | jq '.confidence'

# Expected: >0 (after Patch 5)

# 5. Forecast input is synced

# Manual: Open cockpit, check if input shows "BTC"

# 6. Top movers visible

# Manual: Open cockpit, verify rows appear in Top Movers panel

# 7. Mode toggle works

# Manual: Toggle LIVE/FIXED, verify POST in network tab

# 8. News sentiment varies

# Manual: Check news feed, verify not all "Neutral"

```text

---**Next Action:** Implement Patches 5-8 (High Priority batch)
