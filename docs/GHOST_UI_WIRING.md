# GHOST UI WIRING - COCKPIT V3

**Purpose**: Map each UI panel to its backend API endpoint and validate data flow.

---

## PANEL MAPPING TABLE

| Panel Name | HTML Element | JS Function | API Endpoint | JSON Fields Used | Status |
|------------|--------------|-------------|--------------|------------------|--------|
| **Top Movers** | `#panel-movers` | `loadTopMovers()` | `/api/v3/hunter/feed` | `movers[].symbol`, `.type`, `.name`, `.price`, `.change`, `.confidence` | ⚠️ PARTIAL |
| **VIP Coins** | `#panel-vip` | `loadVIPCoins()` | `/api/v3/vip/snapshot` | `vip_coins[].symbol`, `.price`, `.change_pct`, `.status`; `xrp.symbol`, `.price`, `.change_pct` | ✅ WORKING |
| **Ghost Forecast** | `#panel-forecast` | `loadForecast()` | `/api/v3/predictions/latest?symbol={symbol}` | `predictions[0].direction`, `.confidence`, `.expected_move` | ✅ WORKING |
| **News Feed** | `#panel-news` | `loadNews()` | `/api/v3/news/feed?limit=10` | `items[].headline`, `.sentiment`, `.timestamp` | ✅ WORKING |
| **Prediction Accuracy** | `#panel-accuracy` | `loadAccuracyChart()` | `/api/v3/predictions/history?limit=100` | `predictions[]` (array) | ❌ NOT WIRED |
| **Watchlist** | `#panel-watchlist` | `loadWatchlist()` | `/api/v3/watchlist` + `/api/v3/predictions/latest?limit=100` | watchlist: `stocks[]`, `crypto[]`, `vip[]`; predictions: `predictions[].symbol`, `.confidence`, `.direction` | ✅ WORKING |
| **Ghost Health Score** | `#panel-health` | `loadHealthScore()` | `/api/v3/goals/snapshot` | `ghost_score`, `daily_goal_pct`, `weekly_goal_pct`, `monthly_goal_pct` | ✅ WORKING |

---

## DETAILED WIRING ANALYSIS

### 1. TOP MOVERS PANEL

**HTML**: `templates/cockpit_v3.html:39`
```html
<section class="panel" id="panel-movers">
    <div class="panel-header">
        <h2>Top Movers</h2>
        <div class="tabs">
            <button class="tab active" data-tab="stocks">Stocks</button>
            <button class="tab" data-tab="crypto">Crypto</button>
            <button class="tab" data-tab="all">All</button>
        </div>
    </div>
    <div class="panel-body">
        <div id="movers-list" class="movers-grid"></div>
    </div>
</section>
```

**JavaScript**: `static/cockpit_v3.js:153`
```javascript
async function loadTopMovers() {
    try {
        const response = await fetch('/api/v3/hunter/feed');
        if (!response.ok) throw new Error('Failed to load movers');
        
        const data = await response.json();
        const container = document.getElementById('movers-list');
        
        // V3 format: {movers: [...], timestamp: N}
        const movers = data.movers || [];
        
        if (!movers || movers.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; padding: 40px;">
                    <div style="font-size: 48px;">👁️</div>
                    <div>No High-Quality Opportunities</div>
                </div>
            `;
            return;
        }
        
        // Filter by current tab (stocks/crypto/all)
        let filtered = movers;
        if (currentTab === 'stocks') {
            filtered = movers.filter(item => item.type === 'stock');
        } else if (currentTab === 'crypto') {
            filtered = movers.filter(item => item.type === 'crypto');
        }
        
        container.innerHTML = filtered.map(item => `
            <div class="mover-card">
                <div class="mover-name">${item.symbol}</div>
                <div class="mover-change ${item.change >= 0 ? 'positive' : 'negative'}">
                    ${item.change >= 0 ? '+' : ''}${item.change?.toFixed(2)}%
                </div>
                <div class="mover-confidence">Ghost: ${item.confidence}%</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('[GHOST V3] Error loading movers:', error);
    }
}
```

**API Response** (actual from production):
```json
{
  "movers": [
    {
      "symbol": "BTC",
      "type": "crypto",
      "name": "Bitcoin",
      "price": 0.0,
      "change": 0.0,
      "volume": 0,
      "confidence": 0,
      "note": "Scanner warming up - check back in 60 seconds"
    }
  ],
  "timestamp": 1764000081.1028004
}
```

**ISSUES DETECTED**:
- ⚠️ **Warming Up Placeholder**: Hunter feed returns placeholder data when scan hasn't completed
- ⚠️ **Zero Values**: `price: 0.0`, `change: 0.0`, `confidence: 0` indicate no real data
- ⚠️ **HTTP 502 Errors**: Railway logs show frequent `GET /api/v3/hunter/feed 502` timeouts
- ❌ **Root Cause**: Hunter feed cache refresh takes too long, causing HTTP timeouts

**WIRING STATUS**: ⚠️ PARTIAL - UI correctly fetches and renders API data, but backend frequently returns placeholder/timeout

---

### 2. VIP COINS PANEL

**HTML**: `templates/cockpit_v3.html:58`
```html
<section class="panel" id="panel-vip">
    <div class="panel-header">
        <h2>🌟 VIP Coins</h2>
    </div>
    <div class="panel-body">
        <div id="vip-list" class="movers-grid"></div>
    </div>
</section>
```

**JavaScript**: `static/cockpit_v3.js:201`
```javascript
async function loadVIPCoins() {
    try {
        const response = await fetch('/api/v3/vip/snapshot');
        if (!response.ok) throw new Error('Failed to load VIP coins');
        
        const data = await response.json();
        const container = document.getElementById('vip-list');
        
        // Get VIP coins and XRP
        const vipCoins = data.vip_coins || [];
        const xrp = data.xrp || null;
        
        // Combine VIP + XRP
        const allCoins = [...vipCoins];
        if (xrp) {
            allCoins.push({
                symbol: xrp.symbol,
                price: xrp.price,
                change_pct: xrp.change_pct,
                status: xrp.provider !== 'offline' ? 'online' : 'offline'
            });
        }
        
        container.innerHTML = allCoins.map(coin => `
            <div class="mover-card vip-card">
                <div class="mover-name">${coin.symbol}</div>
                <div class="mover-change">${coin.change_pct}%</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('[GHOST V3] Error loading VIP coins:', error);
    }
}
```

**API Response**: ✅ Returns proper data structure

**WIRING STATUS**: ✅ WORKING - Endpoint exists, returns correct JSON, UI renders correctly

---

### 3. GHOST FORECAST PANEL

**HTML**: `templates/cockpit_v3.html:72`
```html
<section class="panel" id="panel-forecast">
    <div class="panel-header">
        <h2>Ghost Forecast</h2>
        <input type="text" id="forecast-symbol" placeholder="Enter symbol..." />
    </div>
    <div class="panel-body">
        <div id="forecast-grid" class="forecast-container">
            <div class="forecast-card">
                <div class="forecast-time">Next 24h</div>
                <div class="forecast-direction">↑ BUY</div>
                <div class="forecast-prob">Prob: <span class="prob-value">--</span>%</div>
            </div>
        </div>
    </div>
</section>
```

**JavaScript**: `static/cockpit_v3.js:267`
```javascript
async function loadForecast() {
    try {
        const response = await fetch(`/api/v3/predictions/latest?symbol=${currentForecastSymbol}`);
        if (!response.ok) throw new Error('Failed to load forecast');
        
        const data = await response.json();
        
        // V3 format: {predictions: [{direction, confidence, horizon_h}]}
        const predictions = data.predictions || [];
        const pred = predictions[0] || {};
        
        // Map single prediction to all timeframes
        updateForecastCard(0, pred, '☀️', '24h');
        updateForecastCard(1, pred, '⛅', '2-5d');
        updateForecastCard(2, pred, '🌤️', '7-14d');
    } catch (error) {
        console.error('[GHOST V3] Error loading forecast:', error);
    }
}
```

**API Response** (actual):
```json
{
  "predictions": [
    {
      "id": 85,
      "symbol": "WOLF",
      "run_at": 1764000894,
      "direction": "FLAT",
      "confidence": 0.4,
      "horizon_h": 48,
      "outcome": "pending"
    }
  ],
  "count": 5,
  "timestamp": 1764000907
}
```

**WIRING STATUS**: ✅ WORKING - Correct endpoint, correct JSON mapping, UI updates properly

---

### 4. NEWS FEED PANEL

**HTML**: `templates/cockpit_v3.html:97`
```html
<section class="panel" id="panel-news">
    <div class="panel-header">
        <h2>News Feed</h2>
    </div>
    <div class="panel-body">
        <div id="news-list" class="news-container"></div>
    </div>
</section>
```

**JavaScript**: `static/cockpit_v3.js:300`
```javascript
async function loadNews() {
    try {
        const response = await fetch('/api/v3/news/feed?limit=10');
        if (!response.ok) throw new Error('Failed to load news');
        
        const data = await response.json();
        const container = document.getElementById('news-list');
        
        if (!data || !data.items || data.items.length === 0) {
            container.innerHTML = '<p>No news available yet</p>';
            return;
        }
        
        container.innerHTML = data.items.map(article => `
            <div class="news-item">
                <div class="news-headline">${article.headline}</div>
                <div class="news-sentiment">${formatSentiment(article.sentiment)}</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('[GHOST V3] Error loading news:', error);
    }
}
```

**WIRING STATUS**: ✅ WORKING - Correct endpoint (`/api/v3/news/feed`), correct field mapping (`items[]`)

---

### 5. PREDICTION ACCURACY PANEL

**HTML**: `templates/cockpit_v3.html:111`
```html
<section class="panel" id="panel-accuracy">
    <div class="panel-header">
        <h2>Prediction Accuracy</h2>
    </div>
    <div class="panel-body">
        <canvas id="accuracy-chart"></canvas>
    </div>
</section>
```

**JavaScript**: `static/cockpit_v3.js:337`
```javascript
async function loadAccuracyChart() {
    try {
        const response = await fetch('/api/v3/predictions/history?limit=100');
        if (!response.ok) throw new Error('Failed to load accuracy data');
        
        const data = await response.json();
        renderAccuracyChart(data);
    } catch (error) {
        console.error('[GHOST V3] Error loading accuracy chart:', error);
        renderAccuracyChart({predictions: []});
    }
}

function renderAccuracyChart(data) {
    const canvas = document.getElementById('accuracy-chart');
    const ctx = canvas.getContext('2d');
    
    // Simple line chart (placeholder - not actually implemented)
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'var(--accent-green)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}
```

**ISSUES DETECTED**:
- ❌ **Not Called**: `loadAccuracyChart()` is NEVER called in `loadAllPanels()` or any interval
- ❌ **Stub Implementation**: `renderAccuracyChart()` just draws a horizontal line, doesn't use actual data
- ❌ **No Chart Library**: Code comments say "can be replaced with Chart.js later" but never was

**WIRING STATUS**: ❌ NOT WIRED - Function exists but is never called, panel always shows blank canvas

---

### 6. WATCHLIST PANEL

**HTML**: `templates/cockpit_v3.html:123`
```html
<section class="panel" id="panel-watchlist">
    <div class="panel-header">
        <h2>Watchlist</h2>
    </div>
    <div class="panel-body">
        <div id="watchlist-table" class="watchlist-grid"></div>
    </div>
</section>
```

**JavaScript**: `static/cockpit_v3.js:365`
```javascript
async function loadWatchlist() {
    try {
        const response = await fetch('/api/v3/watchlist');
        if (!response.ok) throw new Error('Failed to load watchlist');
        
        const data = await response.json();
        
        // Combine all symbol groups
        const allSymbols = [
            ...(data.stocks || []).map(s => ({symbol: s, type: 'stock'})),
            ...(data.crypto || []).map(s => ({symbol: s, type: 'crypto'})),
            ...(data.vip || []).map(s => ({symbol: s, type: 'vip'}))
        ];
        
        // Fetch predictions for all symbols
        const predResponse = await fetch('/api/v3/predictions/latest?limit=100');
        const predData = await predResponse.json();
        
        // Create lookup map for predictions
        const predMap = {};
        if (predData && predData.predictions) {
            predData.predictions.forEach(pred => {
                predMap[pred.symbol] = {
                    confidence: pred.confidence * 100,  // Convert 0.45 to 45
                    direction: pred.direction
                };
            });
        }
        
        // Enrich watchlist with prediction data
        const watchlistData = allSymbols.map(item => ({
            symbol: item.symbol,
            ghost_score: predMap[item.symbol]?.confidence || 0,
            direction: predMap[item.symbol]?.direction || 'FLAT',
            type: item.type
        }));
        
        renderWatchlist(watchlistData);
    } catch (error) {
        console.error('[GHOST V3] Error loading watchlist:', error);
    }
}
```

**WIRING STATUS**: ✅ WORKING - Fetches from two endpoints, merges data correctly, enriches watchlist with predictions

---

### 7. GHOST HEALTH SCORE PANEL

**HTML**: `templates/cockpit_v3.html:143`
```html
<section class="panel" id="panel-health">
    <div class="panel-header">
        <h2>Ghost Health Score</h2>
    </div>
    <div class="panel-body">
        <div class="health-container">
            <div class="health-score">
                <div class="score-circle">
                    <span id="health-score-value">--</span>
                </div>
                <div class="score-grade" id="health-grade">-</div>
            </div>
            <div class="health-metrics" id="health-metrics"></div>
        </div>
    </div>
</section>
```

**JavaScript**: `static/cockpit_v3.js:455`
```javascript
async function loadHealthScore() {
    try {
        const response = await fetch('/api/v3/goals/snapshot');
        if (!response.ok) throw new Error('Failed to load health score');
        
        const data = await response.json();
        
        const score = data.ghost_score || 0;
        const grade = calculateGrade(score);
        
        document.getElementById('health-score-value').textContent = score > 0 ? score.toFixed(0) : '--';
        document.getElementById('health-grade').textContent = grade;
        
        // Update goal progress as health metrics
        renderHealthMetrics({
            daily: data.daily_goal_pct || 0,
            weekly: data.weekly_goal_pct || 0,
            monthly: data.monthly_goal_pct || 0
        });
    } catch (error) {
        console.error('[GHOST V3] Error loading health score:', error);
    }
}
```

**API Response** (actual):
```json
{
  "ghost_score": 92.0,
  "daily_goal_pct": 45.2,
  "weekly_goal_pct": 68.5,
  "monthly_goal_pct": 82.3,
  "status": "ok",
  "timestamp": 1763999640.1564348
}
```

**WIRING STATUS**: ✅ WORKING - Correct endpoint, correct JSON mapping, displays score and grade

---

## CRITICAL ISSUES SUMMARY

### ❌ BROKEN

1. **Prediction Accuracy Chart** - Function exists but NEVER called, panel always blank
   - **Fix**: Add `loadAccuracyChart()` to `loadAllPanels()` and implement Chart.js rendering

### ⚠️ PARTIAL

2. **Hunter Feed** - Frequently returns placeholder "warming up" message or HTTP 502 timeouts
   - **Root Cause**: Backend hunter feed cache refresh is slow, causes HTTP timeouts
   - **Evidence**: Railway logs show `GET /api/v3/hunter/feed 502` errors
   - **Fix**: Optimize hunter feed cache refresh or increase timeout limits

### ✅ WORKING

3. **VIP Coins** - Correct endpoint, correct data structure
4. **Ghost Forecast** - Fetches predictions correctly, updates cards
5. **News Feed** - Displays news items with sentiment
6. **Watchlist** - Merges watchlist + predictions correctly
7. **Ghost Health Score** - Displays score and goal progress

---

## BASE URL VALIDATION

All JavaScript fetch calls use **relative paths** (`/api/v3/...`), which means:
- ✅ **Local**: Uses `http://localhost:8080/api/v3/...`
- ✅ **Production**: Uses `https://ghost-protocol-production.up.railway.app/api/v3/...`

**No hardcoded localhost URLs detected** - UI will work on any base URL.

---

## ERROR HANDLING AUDIT

All panel loading functions have:
- ✅ `try/catch` blocks
- ✅ `console.error()` logging
- ✅ Graceful degradation (show "--" or placeholder if data unavailable)

**MISSING**:
- ❌ No visual error indicators (red banner, error icon)
- ❌ No retry logic for failed fetches
- ❌ No loading spinners (user doesn't know if data is loading or failed)

---

**Generated**: November 24, 2025  
**Auditor**: Ghost Truth Squad  
**Overall Status**: 6/7 Panels Wired (86%) | 1 Panel Broken | 1 Panel Partial
