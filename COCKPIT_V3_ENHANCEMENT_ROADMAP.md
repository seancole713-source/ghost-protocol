# Cockpit v3 Enhancement Roadmap

**Status**: Post-QA Pass - Optional Improvements
**Date**: December 5, 2025
**Priority**: Non-blocking (all enhancements are polish/optimization)

---

## Quick Wins (1-2 hours each)

### 1. Wire Watchlist Metadata Columns ⭐

**Current State**: Watchlist displays symbol and price only
**Available Data**: Backend returns `change_pct`, `ghost_confidence`, `ghost_direction`
**Gap**: UI not displaying these fields despite data availability

**Implementation**:

File: `static/js/cockpit_v3.js` (approx line 740-760)

```javascript
// BEFORE (current)
const row = `
  <tr>
    <td>${item.symbol}</td>
    <td>$${item.price.toFixed(2)}</td>
  </tr>
`;

// AFTER (enhanced)
const row = `
  <tr>
    <td class="symbol-cell">${item.symbol}</td>
    <td class="price-cell">$${item.price.toFixed(2)}</td>
    <td class="change-cell ${item.change_pct >= 0 ? 'positive' : 'negative'}">
      ${item.change_pct >= 0 ? '+' : ''}${item.change_pct.toFixed(2)}%
    </td>
    <td class="confidence-cell">${item.ghost_confidence}%</td>
    <td class="direction-cell ghost-dir-${item.ghost_direction.toLowerCase()}">
      <span class="direction-arrow">${item.ghost_direction === 'UP' ? '↑' : '↓'}</span>
      ${item.ghost_direction}
    </td>
  </tr>
`;

```text

**CSS Addition**(`static/css/cockpit_v3.css`):

```css

.change-cell.positive { color: #00ff88; font-weight: 600; }
.change-cell.negative { color: #ff4444; font-weight: 600; }
.confidence-cell { color: #ffd700; font-weight: 500; }
.ghost-dir-up { color: #00ff88; }
.ghost-dir-down { color: #ff4444; }
.direction-arrow { font-size: 1.2em; margin-right: 4px; }

```text**Table Header Update**(`templates/cockpit_v3.html`):

```html

<!-- BEFORE -->
<thead>
  <tr>
    <th>Symbol</th>
    <th>Price</th>
  </tr>
</thead>

<!-- AFTER -->
<thead>
  <tr>
    <th>Symbol</th>
    <th>Price</th>
    <th>24h Change</th>
    <th>Ghost Confidence</th>
    <th>Direction</th>
  </tr>
</thead>

```text**Effort**: 30-45 minutes

**Impact**: High (shows full trading context per symbol)
**Priority**: ⭐⭐⭐ (Recommended next)

---

### 2. Major Caps Display (BTC/ETH) ⭐

**Current State**: Top-right block shows "-- / --" for BTC and ETH
**Available Data**: Market data exists in predictions and market endpoints
**Gap**: Not wired to data source

**Implementation**:

File: `static/js/cockpit_v3.js` (add to polling loop)

```javascript

// Add new function
async function updateMajorCaps() {
  try {
    // Fetch BTC data
    const btcResp = await fetch('/api/v3/predictions/latest?symbol=BTC');
    const btcData = await btcResp.json();

    // Fetch ETH data
    const ethResp = await fetch('/api/v3/predictions/latest?symbol=ETH');
    const ethData = await ethResp.json();

    // Update BTC display
    if (btcData && btcData.predictions && btcData.predictions.length > 0) {
      const btc = btcData.predictions[0];
      const btcPrice = btc.current_price || btc.price || 0;
      const btcChange = btc.change_pct || 0;

      document.getElementById('btc-price').textContent = `$${btcPrice.toLocaleString()}`;
      document.getElementById('btc-change').textContent = `${btcChange >= 0 ? '+' : ''}${btcChange.toFixed(2)}%`;
      document.getElementById('btc-change').className = btcChange >= 0 ? 'positive' : 'negative';
    }

    // Update ETH display
    if (ethData && ethData.predictions && ethData.predictions.length > 0) {
      const eth = ethData.predictions[0];
      const ethPrice = eth.current_price || eth.price || 0;
      const ethChange = eth.change_pct || 0;

      document.getElementById('eth-price').textContent = `$${ethPrice.toLocaleString()}`;
      document.getElementById('eth-change').textContent = `${ethChange >= 0 ? '+' : ''}${ethChange.toFixed(2)}%`;
      document.getElementById('eth-change').className = ethChange >= 0 ? 'positive' : 'negative';
    }

  } catch (error) {
    console.error('[MAJOR CAPS] Update failed:', error);
  }
}

// Add to initialization
setInterval(updateMajorCaps, 5000); // Update every 5 seconds
updateMajorCaps(); // Initial load

```text

**HTML Update**(`templates/cockpit_v3.html`):

```html

<!-- BEFORE -->
<div class="major-cap-item">
  <span class="symbol">BTC</span>
  <span class="price">-- / --</span>
</div>

<!-- AFTER -->
<div class="major-cap-item">
  <span class="symbol">BTC</span>
  <span class="price" id="btc-price">--</span>
  <span class="change" id="btc-change">--</span>
</div>
<div class="major-cap-item">
  <span class="symbol">ETH</span>
  <span class="price" id="eth-price">--</span>
  <span class="change" id="eth-change">--</span>
</div>

```text**Effort**: 45 minutes

**Impact**: Medium (informational context)
**Priority**: ⭐⭐ (Nice to have)

---

## Performance Optimization (2-3 hours)

### 3. Tiered Polling Strategy 🚀

**Current State**: All endpoints polled every 300-500ms (very aggressive)
**Issue**: 50-100+ requests per minute creates noise and load
**Opportunity**: Tier polling by data criticality

**Proposed Tiers**:

| Tier | Interval | Endpoints | Rationale |
|------|----------|-----------|-----------|
| **Critical**| 500ms | `/api/v3/ghost/status`<br>`/api/v3/ghost/health` | Engine state must be real-time |
|**High**| 2000ms | `/api/v3/predictions/latest`<br>`/api/v3/watchlist/user` | Trading data needs fast updates |
|**Medium**| 5000ms | `/api/v3/ghost/feed`<br>`/api/v3/ghost/snapshot`<br>`/api/v3/ghost/enriched` | News/context less
time-sensitive |
|**Low**| 10000ms | `/api/v3/ghost/metrics`<br>`/api/v3/tracker` | Historical/aggregate data |**Implementation**:

File: `static/js/cockpit_v3.js`

```javascript

// BEFORE (current - single aggressive interval)
setInterval(updateAllData, 400); // Polls everything every 400ms

// AFTER (tiered polling)
// Critical tier - Engine status and health
const criticalPoller = setInterval(async () => {
  await updateEngineStatus();
  await updateGhostHealth();
}, 500);

// High tier - Trading data
const highPoller = setInterval(async () => {
  await updateForecast();
  await updateWatchlist();
}, 2000);

// Medium tier - Context data
const mediumPoller = setInterval(async () => {
  await updateFeed();
  await updateSnapshot();
  await updateEnriched();
}, 5000);

// Low tier - Metrics and analytics
const lowPoller = setInterval(async () => {
  await updateMetrics();
  await updateTracker();
}, 10000);

// Initial load (fire immediately)
Promise.all([
  updateEngineStatus(),
  updateGhostHealth(),
  updateForecast(),
  updateWatchlist(),
  updateFeed(),
  updateSnapshot(),
  updateMetrics(),
  updateTracker()
]).then(() => {
  console.log('[COCKPIT] All data loaded');
});

```text

**Benefits**:

- Reduce backend requests by ~60% (from 100/min to 40/min)
- Lower network noise for debugging
- Maintain responsiveness for critical data
- Easier to profile performance issues


**Metrics Before/After**:
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Requests/min | 100-120 | 35-45 | -65% |
| Critical data lag | 400ms | 500ms | +100ms |
| Context data lag | 400ms | 5000ms | +4600ms |
| Load on backend | High | Low | Significant |

**Effort**: 2 hours
**Impact**: High (performance, stability)
**Priority**: ⭐⭐⭐ (Recommended for production optimization)

---

## UX Enhancements (1 hour each)

### 4. Goals Persistence 💾

**Current State**: Trading goals modal fields empty on reopen
**User Pain**: Must re-enter goals every session
**Solution**: Persist to localStorage (or DB for multi-device)

**Implementation**:

File: `static/js/cockpit_v3.js`

```javascript

// Save goals on modal submit
function saveGoals() {
  const goals = {
    daily: document.getElementById('goal-daily').value,
    weekly: document.getElementById('goal-weekly').value,
    monthly: document.getElementById('goal-monthly').value,
    yearly: document.getElementById('goal-yearly').value,
    updated: new Date().toISOString()
  };

  localStorage.setItem('ghost_trading_goals', JSON.stringify(goals));
  console.log('[GOALS] Saved:', goals);
}

// Load goals on modal open
function loadGoals() {
  const saved = localStorage.getItem('ghost_trading_goals');
  if (!saved) return;

  try {
    const goals = JSON.parse(saved);
    document.getElementById('goal-daily').value = goals.daily || '';
    document.getElementById('goal-weekly').value = goals.weekly || '';
    document.getElementById('goal-monthly').value = goals.monthly || '';
    document.getElementById('goal-yearly').value = goals.yearly || '';

    console.log('[GOALS] Loaded:', goals);
  } catch (e) {
    console.error('[GOALS] Load failed:', e);
  }
}

// Wire to modal events
document.getElementById('goals-modal-open-btn').addEventListener('click', loadGoals);
document.getElementById('goals-submit-btn').addEventListener('click', saveGoals);

```text

**Alternative**: Server-side persistence via `/api/v3/user/goals` endpoint (requires backend changes)

**Effort**: 1 hour (localStorage) / 3 hours (server-side)
**Impact**: Medium (UX quality of life)
**Priority**: ⭐ (Low priority)

---

### 5. Symbol Switcher for Forecast 🔄

**Current State**: Forecast always shows BTC
**User Request**: "Flip the symbol from BTC to XRP"
**Enhancement**: Add dropdown to change forecast symbol

**Implementation**:

File: `templates/cockpit_v3.html`

```html

<!-- Add dropdown above forecast panel -->
<div class="forecast-controls">
  <label>Symbol:</label>
  <select id="forecast-symbol-select">
    <option value="BTC" selected>BTC</option>
    <option value="ETH">ETH</option>
    <option value="XRP">XRP</option>
    <option value="SOL">SOL</option>
    <option value="WOLF">WOLF</option>
    <option value="SPY">SPY</option>
  </select>
</div>

```text

File: `static/js/cockpit_v3.js`

```javascript

// Update forecast function to use selected symbol
async function updateForecast() {
  const symbol = document.getElementById('forecast-symbol-select').value;
  const url = `/api/v3/predictions/latest?symbol=${symbol}`;

  const resp = await fetch(url);
  const data = await resp.json();

  console.log(`[FORECAST] Loaded for ${symbol}:`, data);

  // ... rest of forecast rendering logic
}

// Listen for symbol changes
document.getElementById('forecast-symbol-select').addEventListener('change', () => {
  updateForecast(); // Reload forecast immediately
});

```text

**Effort**: 1 hour
**Impact**: Medium (user flexibility)
**Priority**: ⭐⭐ (Quality of life)

---

## Implementation Priority

**Recommended Order**:

1. **⭐⭐⭐ Watchlist Metadata Columns**(30 min) - Highest impact, quickest win


2.**⭐⭐⭐ Tiered Polling**(2 hrs) - Best performance ROI
3.**⭐⭐ Major Caps Display**(45 min) - Contextual value
4.**⭐⭐ Symbol Switcher**(1 hr) - User flexibility
5.**⭐ Goals Persistence**(1 hr) - Nice to have**Total Effort**: 5-6 hours for all enhancements

---

## Testing Checklist (Post-Enhancement)

After implementing enhancements, verify:

- [ ] Watchlist table shows 5 columns (symbol, price, change, confidence, direction)
- [ ] BTC/ETH prices update in top-right block
- [ ] Console shows reduced request frequency (40-50 req/min vs 100+)
- [ ] Changing forecast symbol triggers new console log: `[FORECAST] Loaded for XRP`
- [ ] Trading goals persist across page reloads
- [ ] No new console errors introduced
- [ ] Response times remain <1s for all tiers


---

## Long-Term Improvements (Future Consideration)

### Advanced Features (Not Prioritized)

1. **WebSocket Streaming**(4-6 hours)
   - Replace polling with WebSocket for real-time updates
   - Reduce latency from 500ms to <50ms
   - Requires backend WebSocket server


1.**Chart Integration**(8-12 hours)

   - Add TradingView or Plotly charts for price history
   - Visual forecast overlay on price action
   - Requires historical data pipeline


1.**Alert System**(4-6 hours)

   - Browser notifications for confidence threshold hits
   - Email/SMS alerts for high-conviction trades
   - Requires notification backend


1.**Performance Dashboard**(6-8 hours)

   - Track prediction accuracy over time
   - P&L visualization
   - Win rate by symbol/timeframe
   - Requires accuracy tracking DB (already built in previous work)


---

## Conclusion

Cockpit v3 UI is**production-ready**as-is. All enhancements in this roadmap are optional polish and performance
optimizations.**Next Action**: Select 1-2 quick wins from the priority list and implement as time permits.

**Status**: ✅ No blockers, proceed to monitoring phase

---

**END OF ROADMAP**

Date: December 5, 2025
Author: Ghost Protocol Enhancement Team
Phase: Post-QA Pass Optimization
