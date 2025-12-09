# COCKPIT PRODUCTION VERIFICATION CHECKLIST

**Date:**December 3, 2025**Commits:**- 8502279: UI fixes (timer, status, timeouts)

- 33cd320: Background worker (499 fix)
- c143509: Health score, goals modal, counters**Production URL:**<<<<<https://ghost-protocol-production.up.railway.app/cockpit>>>>>

---

## I. BEFORE YOU START

### 1. Open Browser DevTools (F12)

- Navigate to: <<<<<https://ghost-protocol-production.up.railway.app/cockpit>>>>>
- Open DevTools:**F12**or**Right-click → Inspect**- Switch to**Console**tab
- Keep it open during all tests

### 2. Check for JavaScript Errors

Look for**RED text**in console. Common issues:

- `Uncaught ReferenceError: initializeApp is not defined`
- `Failed to load resource: /static/cockpit_v3.js 404`
- `Uncaught TypeError: Cannot read property...`

If you see errors,**copy the entire error message and report it**.

---

## II. TIMER TEST (Issue #1 from original diagnostic)

### Expected Behavior

✅ Timer should animate: 00:00:01, 00:00:02, 00:00:03...

### Current Status in Browser

- [ ] Timer stuck at 00:00:00
- [ ] Timer animating correctly

### If Stuck, Run This in Console

```javascript
// Check if function exists
typeof updateSystemTime

// Should return: "function"
// If it returns "undefined", JS file didn't load

// Force timer update
setInterval(() => {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, '0');
    const m = String(now.getMinutes()).padStart(2, '0');
    const s = String(now.getSeconds()).padStart(2, '0');
    document.getElementById('system-time').textContent = `${h}:${m}:${s}`;
}, 1000);

// Timer should start animating now

```text

---

## III. STATUS INDICATOR TEST

### Expected Behavior

✅ Status should show "LIVE" with **green dot**### Current Status in Browser

- [ ] Shows "LIVE" with green dot
- [ ] Shows "STOPPED" with red dot
- [ ] Shows "LIVE" but no green dot
- [ ] Static "LIVE" text only


---

## IV. START/STOP/RESET BUTTONS (Issue from diagnostic)

### Test START Button

1. Click**START**button
2. Open DevTools →**Network**tab
3. Look for a request to `/api/cockpit/start`


### Expected

✅ POST request appears in Network tab
✅ Status changes to LIVE (if wasn't already)

### Current Status

- [ ] Button works (Network request sent)
- [ ] Button does nothing (no Network activity)
- [ ] Console error when clicked


### If Nothing Happens, Run This

```javascript

// Check if event listener attached
document.getElementById('btn-start').onclick

// Should return: function or null
// If null, event listeners not attached

// Force attach
document.getElementById('btn-start').addEventListener('click', async () => {
    const r = await fetch('/api/cockpit/start', {method: 'POST'});
    console.log('START response:', await r.json());
});

```text

---

## V. TOP MOVERS PANEL (Issue #2 from original diagnostic)

### Expected Behavior

✅ Shows list of crypto/stocks with:

- Ticker symbols (BTC, ETH, etc.)
- Price change percentages (+1.5%, -0.3%)
- Ghost confidence scores (46%, 50%)


### Current Status in Browser

- [ ] Shows movers list with data
- [ ] Empty panel (no tickers)
- [ ] Shows "No High-Quality Opportunities" message
- [ ] Loading spinner stuck


### If Empty, Check in Console

```javascript

// Test API endpoint
fetch('/api/v3/hunter/feed?limit=5')
    .then(r => r.json())
    .then(data => console.log('Movers data:', data))

// Should show: {movers: [...], timestamp: ...}

// Force load movers
if (typeof loadTopMovers === 'function') {
    loadTopMovers();
}

```text

---

## VI. VIP COINS PANEL (Issue #3 from original diagnostic)

### Expected Behavior

✅ Shows 5 VIP coins:

- WEPE
- LILPEPE
- DORKL
- SLOTH
- APC


### Known Issue

⚠️**VIP Coins may timeout**(external APIs take 4+ minutes)

### Current Status in Browser

- [ ] Shows VIP coins with prices
- [ ] Shows "VIP data loading..."
- [ ] Empty (only heading visible)
- [ ] Timeout message


### If Empty, Check

```javascript

// Test VIP endpoint (may take 10s+)
fetch('/api/v3/vip/snapshot')
    .then(r => r.json())
    .then(data => console.log('VIP data:', data))
    .catch(err => console.log('VIP timeout:', err))

```text**Note:**This is a known issue (external API slow). Skip if it times out.

---

## VII. GHOST FORECAST CARDS (Issue #4 from original diagnostic)

### Expected Behavior

✅ Three cards show:

-**Next 24h:**Prob: 46%, Move: +2.5%
-**2-5 Days:**Prob: 50%, Move: +1.0%
-**7-14 Days:**Prob: 55%, Move: +3.0%


### Current Status in Browser

- [ ] Shows real probabilities and moves
- [ ] All show "--%" (no data)
- [ ] Cards exist but empty


### If Empty, Check

```javascript

// Test predictions endpoint
fetch('/api/v3/predictions/latest?symbol=BTC')
    .then(r => r.json())
    .then(data => console.log('Predictions:', data))

// Force load forecast
if (typeof loadForecast === 'function') {
    currentForecastSymbol = 'BTC';
    loadForecast();
}

```text

---

## VIII. NEWS FEED (Issue #5 from original diagnostic)

### Expected Behavior

✅ Shows 5-10 news articles with:

- Headlines
- Timestamps
- Sentiment indicators


### Current Status in Browser

- [ ] Shows news articles
- [ ] Empty (no headlines)
- [ ] Shows "News feed temporarily unavailable"


### If Empty, Check

```javascript

// Test news endpoint
fetch('/api/v3/news/feed?limit=5')
    .then(r => r.json())
    .then(data => console.log('News:', data))

// Force load news
if (typeof loadNews === 'function') {
    loadNews();
}

```text

---

## IX. WATCHLIST (Issue #6 from original diagnostic)

### Expected Behavior

✅ Shows symbols with:

- Ticker names
- Prices
- % changes
- Ghost predictions


### Current Status in Browser

- [ ] Shows watchlist with symbols
- [ ] Empty under all tabs
- [ ] Shows "Watchlist empty" message


### If Empty, Check

```javascript

// Test watchlist endpoint
fetch('/api/v3/watchlist/enriched')
    .then(r => r.json())
    .then(data => console.log('Watchlist:', data))

// Force load watchlist
if (typeof loadWatchlistByMode === 'function') {
    loadWatchlistByMode();
}

```text

---

## X. GHOST HEALTH SCORE (Issue #7 - FIXED in c143509)

### Expected Behavior

✅ Shows numeric value: 50-100 (based on recent predictions)

### Current Status in Browser

- [ ] Shows numeric value (50, 75, 100, etc.)
- [ ] Still shows "--"


### If Still Shows "--", Check

```javascript

// Test health endpoint
fetch('/api/v3/cockpit/status')
    .then(r => r.json())
    .then(data => console.log('Health score:', data.ghost_health_score))

// Should show number, not 0

// Force load health
if (typeof loadHealthScore === 'function') {
    loadHealthScore();
}

```text**Fix Applied:**- Now calculates from actual DB predictions (last 24 hours)

- 10 points per prediction, max 100
- If shows 0, predictions may not have run yet (wait 60 minutes)


---

## XI. GOALS MODAL (Issue #8 - FIXED in c143509)

### Expected Behavior

✅ Modal prepopulates with saved values:

- Daily: $500
- Weekly: $2500
- Monthly: $10000
- Yearly: $120000


### Test Steps

1. Click**⚙️**(settings) button in header
2. Goals modal opens
3. Check if input fields have values


### Current Status in Browser

- [ ] Modal opens with values ($500, $2500, etc.)
- [ ] Modal opens but all inputs empty (value="")
- [ ] Modal doesn't open at all


### If Inputs Empty, Check

```javascript

// Test goals endpoint
fetch('/api/v3/goals/snapshot')
    .then(r => r.json())
    .then(data => console.log('Goals:', data.goals))

// Should show: {daily: 500, weekly: 2500, ...}

// Manual test of modal function
if (typeof openGoalsModal === 'function') {
    openGoalsModal();
}

```text**Fix Applied:**- Removed `.target` optional chaining

- Now reads `data.goals.daily` directly (not `data.goals.daily.target`)


---

## XII. FULL INITIALIZATION TEST

### If Everything Seems Broken, Run Complete Diagnostic

```javascript

// Paste this entire block into console:

console.log('🔍 COCKPIT DIAGNOSTIC');
console.log('=' + '='.repeat(69));

// 1. Check JS loaded
console.log('\n1. FUNCTIONS AVAILABLE:');
['initializeApp', 'loadTopMovers', 'loadForecast', 'loadVIPCoins',
 'loadWatchlistByMode', 'loadHealthScore', 'updateSystemTime'].forEach(fn => {
    console.log(`   ${typeof window[fn] === 'function' ? '✅' : '❌'} ${fn}()`);
});

// 2. Check DOM elements
console.log('\n2. DOM ELEMENTS:');
['system-time', 'movers-list', 'vip-list', 'forecast-grid',
 'news-list', 'watchlist-table', 'health-score-value'].forEach(id => {
    console.log(`   ${document.getElementById(id) ? '✅' : '❌'} #${id}`);
});

// 3. Test all endpoints
console.log('\n3. API ENDPOINTS:');
Promise.all([
    fetch('/api/v3/cockpit/status').then(r => r.json()),
    fetch('/api/v3/hunter/feed?limit=3').then(r => r.json()),
    fetch('/api/v3/predictions/latest?symbol=BTC').then(r => r.json()),
    fetch('/api/v3/watchlist/enriched').then(r => r.json()),
    fetch('/api/v3/goals/snapshot').then(r => r.json())
]).then(([status, movers, predictions, watchlist, goals]) => {
    console.log(`   ✅ Status: Health=${status.ghost_health_score}`);
    console.log(`   ✅ Movers: ${movers.movers?.length || 0} items`);
    console.log(`   ✅ Predictions: ${predictions.predictions?.length || 0} items`);
    console.log(`   ✅ Watchlist: ${watchlist.items?.length || 0} items`);
    console.log(`   ✅ Goals: Daily=$${goals.goals?.daily || 0}`);

    console.log('\n' + '='.repeat(70));
    console.log('📊 DIAGNOSTIC COMPLETE - Copy output and send to developer');
}).catch(err => console.log('❌ Endpoints failed:', err));

// 4. Force initialize
console.log('\n4. FORCING INITIALIZATION...');
if (typeof initializeApp === 'function') {
    initializeApp();
    console.log('✅ initializeApp() called');
    console.log('Check if panels now have data');
} else {
    console.log('❌ initializeApp() not found');
}

```text

---

## XIII. SUMMARY CHECKLIST

After running all tests above, fill this out:

### UI Elements

- [ ] Timer animates (not stuck at 00:00:00)
- [ ] Status shows LIVE with green dot
- [ ] START/STOP/RESET buttons trigger network requests


### Data Panels

- [ ] Top Movers shows crypto/stocks
- [ ] VIP Coins shows data (or known timeout message)
- [ ] Ghost Forecast shows probabilities (not --%)
- [ ] News Feed shows articles
- [ ] Watchlist shows symbols
- [ ] Ghost Health Score shows number (not --)
- [ ] Goals modal prepopulates values


### Console

- [ ] No JavaScript errors (red text)
- [ ] initializeApp() logs "✅ Ghost Protocol Cockpit v3 initialized"
- [ ] API calls return 200 status codes


---

## XIV. REPORTING RESULTS

### If Everything Works

✅ Report: "All tests pass, Cockpit fully operational"

### If Issues Remain

Copy and send:

1.**Browser:**Chrome/Firefox/Safari version
2.**Console Output:**Full text from diagnostic script
3.**Network Tab:**Screenshot showing failed requests (if any)
4.**Specific Failures:**Which panels are empty, which buttons don't work
5.**Console Errors:**Any red error messages


### Most Useful Info

```javascript

// Run this and copy the output:
console.log('Browser:', navigator.userAgent);
console.log('Document state:', document.readyState);
console.log('initializeApp type:', typeof initializeApp);
console.log('Timer value:', document.getElementById('system-time')?.textContent);

```text

---

## XV. EXPECTED SUCCESS CRITERIA**Cockpit is FULLY OPERATIONAL when:**1. ✅ Timer moves every second

1. ✅ START/STOP buttons work (Network tab shows requests)
2. ✅ Top Movers shows 5+ predictions
3. ✅ Forecast cards show real percentages
4. ✅ News Feed shows 5+ articles
5. ✅ Watchlist shows 10+ symbols
6. ✅ Health Score shows 0-100 (not --)
7. ✅ Goals modal opens with saved values
8. ✅ No JavaScript console errors**Note:**VIP Coins may timeout (known issue, needs Redis cache).


---**Last Updated:**December 3, 2025**Deployed Commits:**8502279, 33cd320, c143509**Production URL:** <<<<<https://ghost-protocol-production.up.railway.app/cockpit>>>>>
