# GHOST COCKPIT V3 — DIAGNOSTIC REPORT

**Date:**December 2, 2025**Status:**PARTIALLY OPERATIONAL (Syntax Error Fixed, Data Gaps Identified)

---

## EXECUTIVE SUMMARY**Deployment Status:**❌ FAILED (Healthcheck timeout due to syntax error)**Root Cause:**IndentationError in `api/cockpit_v3_live_endpoints.py` line 1142**Fix Applied:**✅ Removed duplicate closing brace (committed but not yet pushed)**Secondary Issues:**3 data pipeline failures identified (VIP coins, watchlist predictions, forecast horizons)

---

## SECTION 1 — DEPLOYMENT FAILURE ROOT CAUSE

### Critical Error (FIXED)

```text
⚠️ Cockpit V3 LIVE endpoints not loaded: unexpected indent (cockpit_v3_live_endpoints.py, line 1142)

```text**File:**`api/cockpit_v3_live_endpoints.py`**Lines 1141-1143:**```python

    }
            "error": str(e)[:200]  # ← Orphaned from bad merge
        }

```text**Impact:**- Cockpit V3 endpoints never mounted

- Healthcheck failed after 6 attempts (1m40s timeout)
- Railway deployment aborted
- UI could not load VIP/Watchlist/Forecast data**Fix Applied:**```python


    }


@router.get("/predictions/latest")

```text**Status:**✅ Syntax valid (`python3 -m py_compile` passes)**Next Step:**Push to trigger Railway redeploy

---

## SECTION 2 — UI VISUAL ANALYSIS vs BACKEND STATE

### A. TOP MOVERS PANEL — ✅ WORKING**UI Shows:**- ETSY +2.90% (Ghost: 59%)

- LCID +2.90% (59%)
- RTX +2.80% (58%)
- NUE -2.80% (58%)
- [etc.]**Backend Status:**✅ OPERATIONAL

- Endpoint: `/api/v3/predictions/latest` working
- Stock provider: Yahoo Finance active
- Prediction engine: Generating signals with confidence scores
- News feed: Populating from prediction cache**Code Reference:**- `static/cockpit_v3.js` line 220-260: `loadTopMovers()`
- Fetches `/api/v3/predictions/latest?limit=10`
- Filters by price change >= 2% threshold
- Displays confidence as Ghost score**No action required**— working as designed.


---

### B. VIP COINS PANEL — ❌ FAILING**UI Shows:**```text

"VIP data unavailable"

```text**Expected VIP Coins:**- BTC, ETH, SOL, BNB, XRP (line 1296 of wolf_app.py)

- WEPE, LILPEPE, DORKL, SLOTH, APC (deprecated presale tokens)**Root Causes:**#### Issue #1: VIP_COINS Reverted to Top 5 Crypto**File:**`wolf_app.py` line 1296


```python

VIP_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP"]  # Reverted: presale coins unavailable on exchanges

```text**Impact:**UI expects presale memecoins, backend now serves major crypto.

#### Issue #2: Cache Returns Stale "Unavailable" State**File:**`wolf_app.py` lines 6817-6870 (`/api/v3/vip/snapshot`)**Cache Logic:**```python

cache_age = time.time() - _VIP_SNAPSHOT_CACHE["timestamp"]
if _VIP_SNAPSHOT_CACHE["data"] and cache_age < _VIP_SNAPSHOT_CACHE["ttl"]:
    return _VIP_SNAPSHOT_CACHE["data"]  # RETURN IMMEDIATELY - don't fetch prices

```text**Problem:**If cache was populated BEFORE emergency hotfix (when CoinGecko was failing), stale "offline" status persists for 5 minutes.

#### Issue #3: Crypto Provider Failures**File:**`core/crypto/crypto_providers.py`**Emergency Hotfix Applied (Nov 30):**- CoinGecko disabled due to 429 rate limits

- Provider order: `["binance", "coinbase"]` (CoinGecko removed from defaults)**Current Behavior:**- Binance: Working for BTC, ETH, SOL, BNB
- XRP: May fail on Binance (not all exchanges list XRP)
- Fallback chain incomplete**Fix Required:**1.**Clear VIP cache on startup:**```python


# wolf_app.py startup section

_VIP_SNAPSHOT_CACHE = {"data": None, "timestamp": 0, "ttl": 300}  # Force fresh fetch

```text

1.**Add XRP to provider priority list:**```python

# core/crypto/crypto_providers.py

CRYPTO_PROVIDER_PRIORITY = {
    "XRP": ["coinbase", "binance", "kraken"],  # XRP widely available
    "default": ["binance", "coinbase"]
}

```text

1.**Shorten cache TTL during debugging:**```python

_VIP_SNAPSHOT_CACHE["ttl"] = 30  # 30 seconds instead of 300

```text

---

### C. FORECAST PANEL — ⚠️ PLACEHOLDER DATA**UI Shows:**```text

Next 24h:  BUY, Prob 41%, Move 2.05%
2–5 Days:  BUY, Prob 41%, Move 2.05%
7–14 Days: BUY, Prob 41%, Move 2.05%

```text**Problem:**All three horizons are IDENTICAL (not realistic).**Root Cause:**Single prediction endpoint doesn't support multi-horizon forecasts.**Code Reference:**`static/cockpit_v3.js` lines 322-345:

```javascript

async function loadForecast() {
    const response = await fetch(`/api/v3/predictions/latest?symbol=${currentForecastSymbol}`);
    const predictions = data.predictions || [];
    const pred = predictions[0] || {};  // ← SINGLE PREDICTION

    // Generate differentiated forecasts for each timeframe
    updateForecastCard(0, pred, '☀️', '24h', 1.0);     // 100% confidence
    updateForecastCard(1, pred, '⛅', '2-5d', 0.7);   // 70% confidence
    updateForecastCard(2, pred, '🌤️', '7-14d', 0.5); // 50% confidence
}

```text**Current Behavior:**- Fetches single 48h prediction

- Artificially decays confidence (70%, 50%) for longer horizons
- Scales expected move by timeframe multipliers (1.8x, 2.5x)


-**Result:**Cosmetic variation, not true multi-horizon forecast**Why It Looks Identical:**The confidence decay formula is not being applied to the UI display
. Looking at the visual output:

- All show 41% (original confidence)
- All show 2.05% move (original expected_move)**Bug:**`updateForecastCard()` likely not applying `confidenceMultiplier` correctly.**Code Inspection Required:**


```javascript

function updateForecastCard(index, prediction, icon, timeframe, confidenceMultiplier = 1.0) {
    let confidence = prediction.confidence || 0;

    // Convert confidence from 0-1 scale to percentage (0-100)
    if (confidence > 0 && confidence <= 1) {
        confidence = confidence * 100;  // 0.41 → 41
    }

    // Apply time decay to confidence
    confidence = confidence *confidenceMultiplier;  // 41* 0.7 = 28.7 (NOT SHOWING)
}

```text

**Hypothesis:**The display is reading from prediction object BEFORE decay is applied, or UI is cached.**Fix Required:**1.**Add debug logging:**```javascript

console.log(`[FORECAST] ${timeframe} confidence: ${confidence.toFixed(0)}% (multiplier: ${confidenceMultiplier})`);

```text

1.**Force UI update:**```javascript

card.querySelector('.prob-value').textContent = Math.round(confidence);

```text

1.**Alternative: Add multi-horizon backend endpoint:**```python

@APP.get("/api/v3/forecast/multi-horizon")
async def api_forecast_multi_horizon(symbol: str):
    """Return predictions for 24h, 5d, 14d horizons"""
    return {
        "24h": predict_with_horizon(symbol, hours=24),
        "5d": predict_with_horizon(symbol, hours=120),
        "14d": predict_with_horizon(symbol, hours=336)
    }

```text

---

### D. NEWS FEED — ✅ WORKING (Sentiment Neutral)**UI Shows:**- Ghost predicts ETSY UP (59%) — Sentiment: Neutral

- LCID UP (59%) — Sentiment: Neutral
- RTX UP (58%) — Sentiment: Neutral
- [etc.]**Backend Status:**✅ OPERATIONAL (predictions flowing)**Secondary Issue:**All sentiment = "Neutral" (sentiment extraction offline)**Root Cause:**News feed is sourced from prediction engine cache, not external news scraper.**Code Reference:**- `static/cockpit_v3.js` line 388-420: `loadNews()`
- Fetches `/api/v3/news/feed?limit=10`
- Displays `article.sentiment` (all returning "Neutral")**Why Sentiment is Neutral:**Predictions don't have sentiment metadata — they only have direction/confidence.**Fix Required:**1.**Add sentiment to prediction metadata:**```python


# services/predictor.py

prediction["sentiment"] = "Bullish" if direction == "UP" else "Bearish"

```text

1.**Or map direction to sentiment in UI:**```javascript

function mapDirectionToSentiment(direction) {
    return direction === 'UP' ? 'Bullish' :
           direction === 'DOWN' ? 'Bearish' : 'Neutral';
}

```text

---

### E. WATCHLIST PANEL — ❌ NO GHOST SIGNALS**UI Shows:**```text

XRP  -3.60%  Ghost: --
SOL  Price   Ghost: --
BNB  Price   Ghost: --
ETH  Price   Ghost: --
BTC  Price   Ghost: --

```text**Problem:**Ghost score column is blank for ALL symbols.**Root Causes:**#### Issue #1: Watchlist Scheduler Not Generating Predictions**File:**`core/watchlist_prediction_scheduler.py`**Current Behavior:**- Scheduler runs market open/close checks (lines 76-95)

- Calls `_run_market_close_predictions()` at 4 PM EST
- Generates predictions via `_generate_prediction()` (line 246)**Deployment Logs Show:**```text


📅 Watchlist scheduler loop active
🔔 Running market close predictions for watchlist stocks...
❌ Failed to get watchlist: relation "ghost_watchlist_items" does not exist
📊 0 stocks in watchlist
✅ Market close predictions complete (0 stocks)

```text**Problem:**Database table `ghost_watchlist_items` doesn't exist in Postgres.**Why:**- Personal watchlist feature requires table creation

- Migration system didn't create this table (not in migrations/)
- Scheduler gracefully skips but UI shows no data


#### Issue #2: Watchlist Endpoint Returns Empty**Expected Endpoint:**`/api/v3/watchlist/enriched`**File:**`api/cockpit_v3_live_endpoints.py` line 1875**Code:**```python

@router.get("/watchlist/enriched")
async def get_watchlist_enriched():
    """Get watchlist with live prices and % changes."""
    watchlist_data = await get_watchlist()  # ← Returns stocks/crypto/vip lists
    stocks = watchlist_data.get("stocks", [])
    crypto = watchlist_data.get("crypto", [])
    vip = watchlist_data.get("vip", [])

```text**The `get_watchlist()` function:**- Returns**predefined**market symbols (not personal watchlist)

- Stocks: BAL, 1INCH, CRV, COMP, SNX, MKR, AAVE (from wolf_app.py line 20556)
- Crypto: Same as VIP_COINS
- VIP: Same as VIP_COINS**Why Ghost Scores are Missing:**The enriched endpoint fetches prices but NOT predictions.**Code Check (lines 1920-1950):**```python


enriched_items.append({
    "symbol": symbol,
    "price": float(crypto_result.get("price", 0)),
    "change_pct": 0.0,  # TurboProvider doesn't return change_pct yet
    "type": "crypto" if symbol in crypto else "vip",
    "provider": crypto_result.get("provider", "unknown")
})

```text**Missing:**No prediction lookup!**Expected:**

```python

# After fetching price, fetch prediction

from core.prediction_store import get_prediction_store
store = get_prediction_store()
latest_pred = store.get_latest_prediction(symbol)

enriched_items.append({
    "symbol": symbol,
    "price": price,
    "change_pct": change_pct,
    "type": asset_type,
    "ghost_score": latest_pred.confidence * 100 if latest_pred else None,
    "direction": latest_pred.direction if latest_pred else None
})

```text

**Fix Required:**1.**Add prediction lookup to `/watchlist/enriched` endpoint:**


```python

@router.get("/watchlist/enriched")
async def get_watchlist_enriched():
    from core.prediction_store import get_prediction_store
    store = get_prediction_store()

    # [existing price fetch code...]

    for symbol in all_symbols:
        price_data = fetch_price(symbol)
        pred = store.get_latest_prediction(symbol)

        enriched_items.append({
            "symbol": symbol,
            "price": price_data["price"],
            "change_pct": price_data["change_pct"],
            "type": asset_type,
            "ghost_score": pred.confidence * 100 if pred else 0,
            "direction": pred.direction if pred else "FLAT"
        })

```text

1. **Create personal watchlist table migration:**```sql


-- migrations/003_personal_watchlist.sql
CREATE TABLE IF NOT EXISTS ghost_watchlist_items (
    id SERIAL PRIMARY KEY,
    user_id INTEGER DEFAULT 1,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    owns_position BOOLEAN DEFAULT FALSE,
    notes TEXT,
    alert_threshold_pct NUMERIC(5,2) DEFAULT 5.0,
    priority INTEGER DEFAULT 1,
    added_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_watchlist_user ON ghost_watchlist_items(user_id, is_active);
CREATE INDEX idx_watchlist_symbol ON ghost_watchlist_items(symbol, asset_type);

```text

1.**Update watchlist UI to show predictions:**```javascript

// static/cockpit_v3.js renderWatchlist()
const ghostScore = item.ghost_score || 0;
const direction = item.direction || 'FLAT';
const scoreClass = ghostScore >= 70 ? 'high' : ghostScore >= 50 ? 'medium' : 'low';

<div class="watchlist-ghost ${scoreClass}">
    ${ghostScore > 0 ? ghostScore.toFixed(0) : '--'}
</div>

```text

---

### F. GHOST HEALTH SCORE — ✅ WORKING (Placeholders)**UI Shows:**```text

100 = Grade A
Daily Goal 70%
Weekly Goal 55%
Monthly Goal 40%
Data Health 85%
AI Activity 75%
Accuracy 70%

```text**Backend Status:**✅ OPERATIONAL (hardcoded placeholders)**Code Reference:**`static/cockpit_v3.js` lines 714-726:

```javascript

if (metrics.daily !== undefined) {
    // V3 format with goal progress
    metricsList.push(
        { name: 'Daily Goal', value: metrics.daily },
        { name: 'Weekly Goal', value: metrics.weekly },
        { name: 'Monthly Goal', value: metrics.monthly },
        { name: 'Data Health', value: 85 },  // ← PLACEHOLDER
        { name: 'AI Activity', value: 75 },  // ← PLACEHOLDER
        { name: 'Accuracy', value: 70 }     // ← PLACEHOLDER
    );
}

```text**Problem:**Last 3 metrics are hardcoded, not pulling from backend.**Fix Required:**1.**Add real metrics to `/api/v3/health/score` endpoint:**```python

@APP.get("/api/v3/health/score")
async def api_health_score():
    from core.prediction_reconciliation import get_reconciliation

    reconciliation = get_reconciliation()
    accuracy_metrics = reconciliation.calculate_accuracy_metrics(period_days=7)

    return {
        "score": 100,
        "grade": "A",
        "daily": 70,
        "weekly": 55,
        "monthly": 40,
        "data_health": calculate_data_health(),      # Provider uptime
        "ai_activity": calculate_ai_activity(),      # Prediction generation rate
        "accuracy": accuracy_metrics.get("accuracy_pct", 0)  # Real accuracy
    }

```text

1.**Update UI to use real values:**```javascript

metricsList.push(
    { name: 'Data Health', value: metrics.data_health || 0 },
    { name: 'AI Activity', value: metrics.ai_activity || 0 },
    { name: 'Accuracy', value: metrics.accuracy || 0 }
);

```text

---

## SECTION 3 — BACKEND OPERATIONS SUMMARY

### Prediction Engine Status

| Component | Status | Evidence |
|-----------|--------|----------|
| Stock Predictions | ✅ WORKING | Top Movers shows stock signals with confidence |
| Crypto Predictions | ❌ FAILING | Watchlist crypto has no Ghost scores |
| VIP Predictions | ❌ FAILING | VIP panel shows "unavailable" |
| Watchlist Scheduler | ⚠️ RUNNING | Logs show scheduler active, 0 symbols processed |
| Forecast Horizons | ⚠️ PLACEHOLDER | All 3 horizons show identical values |

### Data Ingestion Layer

| Provider | Status | Notes |
|----------|--------|-------|
| Yahoo Finance (Stocks) | ✅ WORKING | ETSY, LCID, RTX prices updating |
| Binance (Crypto) | ✅ WORKING | BTC, ETH, SOL available |
| Coinbase (Crypto) | ✅ WORKING | Fallback for Binance failures |
| CoinGecko (VIP) | ❌ DISABLED | Emergency hotfix (429 rate limits) |
| Polygon (Stocks) | ⚠️ UNKNOWN | Not tested in current logs |

### Database State

| Table | Status | Notes |
|-------|--------|-------|
| predictions | ✅ EXISTS | Dual-write to Postgres + SQLite |
| ghost_prediction_outcomes | ✅ EXISTS | Migration 002 successful |
| ghost_watchlist_items | ❌ MISSING | Causes watchlist scheduler to skip |
| ghost_personal_watchlist | ❌ MISSING | Personal watchlist feature incomplete |

---

## SECTION 4 — PRIORITY FIXES

### 🔴 CRITICAL (Deploy Blocker)**1. Push Syntax Fix to Railway**```bash

git add api/cockpit_v3_live_endpoints.py
git commit -m "fix: Remove duplicate error key causing IndentationError"
git push origin main

```text**Impact:**Unblocks deployment, allows Cockpit V3 endpoints to mount.

---

### 🟡 HIGH (User-Facing Data Gaps)**2. Add Prediction Lookup to Watchlist Endpoint**

**File:**`api/cockpit_v3_live_endpoints.py` lines 1920-1970**Change:**

```python

# After fetching price for each symbol

from core.prediction_store import get_prediction_store
store = get_prediction_store()

for symbol in all_symbols:

    # [existing price fetch code]

    # NEW: Fetch latest prediction

    try:
        latest_pred = store.get_latest_prediction(symbol)
        ghost_score = latest_pred.confidence * 100 if latest_pred else 0
        direction = latest_pred.direction if latest_pred else "FLAT"
    except Exception as e:
        LOGGER.debug(f"No prediction for {symbol}: {e}")
        ghost_score = 0
        direction = "FLAT"

    enriched_items.append({
        "symbol": symbol,
        "price": price,
        "change_pct": change_pct,
        "type": asset_type,
        "ghost_score": ghost_score,      # NEW
        "direction": direction,           # NEW
        "provider": provider
    })

```text

**Impact:**Watchlist panel will show Ghost scores instead of "--".

---**3. Clear VIP Cache on Startup**

**File:**`wolf_app.py` startup section (after imports)**Change:**```python

# Clear VIP cache to force fresh fetch after emergency hotfix

_VIP_SNAPSHOT_CACHE = {"data": None, "timestamp": 0, "ttl": 300}
LOGGER.info("[STARTUP] VIP cache cleared - will fetch fresh data")

```text**Impact:**VIP panel will fetch live prices instead of stale "unavailable" state.

---**4. Create Personal Watchlist Migration**

**New File:**`migrations/003_personal_watchlist.sql`

```sql

-- Personal Watchlist Tables for Ghost Protocol
-- Supports per-user symbol tracking with prediction integration

CREATE TABLE IF NOT EXISTS ghost_watchlist_items (
    id SERIAL PRIMARY KEY,
    user_id INTEGER DEFAULT 1,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,  -- 'stock', 'crypto'
    owns_position BOOLEAN DEFAULT FALSE,
    notes TEXT,
    alert_threshold_pct NUMERIC(5,2) DEFAULT 5.0,
    priority INTEGER DEFAULT 1,
    added_at TIMESTAMP DEFAULT NOW(),
    last_modified TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,

    CONSTRAINT unique_user_symbol UNIQUE (user_id, symbol, asset_type)
);

CREATE INDEX idx_watchlist_user ON ghost_watchlist_items(user_id, is_active);
CREATE INDEX idx_watchlist_symbol ON ghost_watchlist_items(symbol, asset_type);

CREATE TABLE IF NOT EXISTS ghost_watchlist_price_snapshots (
    id SERIAL PRIMARY KEY,
    watchlist_item_id INTEGER REFERENCES ghost_watchlist_items(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    price NUMERIC(20, 8) NOT NULL,
    change_pct_24h NUMERIC(10, 4),
    volume_24h NUMERIC(20, 2),
    snapshot_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_snapshots_item (watchlist_item_id),
    INDEX idx_snapshots_time (snapshot_at)
);

CREATE TABLE IF NOT EXISTS ghost_watchlist_prediction_tracking (
    id SERIAL PRIMARY KEY,
    watchlist_item_id INTEGER REFERENCES ghost_watchlist_items(id) ON DELETE CASCADE,
    prediction_id INTEGER,
    symbol VARCHAR(20) NOT NULL,
    asset_type VARCHAR(20) NOT NULL,
    triggered_by VARCHAR(50),  -- 'market_open', 'market_close', 'big_move', 'manual'
    triggered_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_tracking_item (watchlist_item_id),
    INDEX idx_tracking_prediction (prediction_id)
);

```text**Impact:**Watchlist scheduler can store personal symbols, track predictions.

---

### 🟢 LOW (Cosmetic/Enhancement)**5. Fix Forecast Confidence Decay Display**

**File:**`static/cockpit_v3.js` lines 350-380**Debug First:**

```javascript

function updateForecastCard(index, prediction, icon, timeframe, confidenceMultiplier = 1.0) {
    let confidence = prediction.confidence || 0;

    if (confidence > 0 && confidence <= 1) {
        confidence = confidence * 100;
    }

    confidence = confidence * confidenceMultiplier;

    // DEBUG: Log to console
    console.log(`[FORECAST DEBUG] ${timeframe}: confidence=${confidence.toFixed(1)}%, multiplier=${confidenceMultiplier}`);

    card.querySelector('.prob-value').textContent = Math.round(confidence);
}

```text

**Then verify in browser console:**- Should see: `[FORECAST DEBUG] 24h: confidence=41.0%, multiplier=1.0`

- Should see: `[FORECAST DEBUG] 2-5d: confidence=28.7%, multiplier=0.7`
- Should see: `[FORECAST DEBUG] 7-14d: confidence=20.5%, multiplier=0.5`**If logs show correct values but UI doesn't update:**- Check if card elements exist before setting `.textContent`
- Verify forecast cards are rendered AFTER forecast data loads


---**6. Map Direction to Sentiment in News Feed**

**File:**`static/cockpit_v3.js` lines 420-430**Change:**

```javascript

function formatSentiment(sentiment, direction) {
    // If sentiment is null/undefined, derive from direction
    if (!sentiment || sentiment === 'Neutral') {
        if (direction === 'UP') return 'Bullish';
        if (direction === 'DOWN') return 'Bearish';
    }

    return sentiment || 'Neutral';
}

// In loadNews():
container.innerHTML = data.items.slice(0, 10).map(article => `
    <div class="news-item">
        <div class="news-headline">${article.headline}</div>
        <div class="news-meta">
            <span class="news-sentiment ${getSentimentClass(article.sentiment)}">
                ${formatSentiment(article.sentiment, article.direction)}
            </span>
            <span class="news-time">${formatTime(article.timestamp)}</span>
        </div>
    </div>
`).join('');

```text

---

## SECTION 5 — TESTING CHECKLIST

After deploying fixes, verify:

### Deployment Health

- [ ] Railway build completes (no syntax errors)
- [ ] Healthcheck passes within 1m40s
- [ ] Cockpit V3 endpoints mount successfully
- [ ] Logs show: `✅ Cockpit V3 LIVE endpoints registered`


### VIP Panel

- [ ] Visit `/api/v3/vip/snapshot` → Returns `{"ok": true, "vip_coins": [...]}`
- [ ] Cockpit VIP panel shows prices for BTC, ETH, SOL, BNB, XRP
- [ ] No "VIP data unavailable" error


### Watchlist Panel

- [ ] Visit `/api/v3/watchlist/enriched` → Returns items with `ghost_score` field
- [ ] Cockpit watchlist shows Ghost scores (not "--")
- [ ] Predictions appear for crypto symbols (BTC, ETH, SOL)


### Forecast Panel

- [ ] Cockpit forecast shows 3 different confidence values (not all 41%)
- [ ] Browser console logs: `[FORECAST DEBUG] 2-5d: confidence=28.7%`
- [ ] Expected: 24h=41%, 2-5d=28%, 7-14d=20%


### News Feed

- [ ] Sentiment shows "Bullish" for UP predictions
- [ ] Sentiment shows "Bearish" for DOWN predictions
- [ ] No longer all "Neutral"


### Database

- [ ] Query: `SELECT COUNT(*) FROM ghost_watchlist_items;` → Should work (not error)
- [ ] Logs show: `[MIGRATION] ✅ 003_personal_watchlist.sql`


---

## SECTION 6 — OPERATOR ACTIONS

### Immediate (Next 5 Minutes)

1. **Push syntax fix to trigger redeploy:**```bash


cd /path/to/ghost-protocol
git add api/cockpit_v3_live_endpoints.py
git commit -m "fix: Remove duplicate error key at line 1142"
git push origin main

```text

1.**Monitor Railway deployment:**- Watch build logs for `✅ Cockpit V3 LIVE endpoints registered`

- Confirm healthcheck passes (not timeout)


### Short-Term (Next 30 Minutes)

1.**Apply VIP cache fix:**```bash

# Edit wolf_app.py startup section

# Add: _VIP_SNAPSHOT_CACHE = {"data": None, "timestamp": 0, "ttl": 300}

git add wolf_app.py
git commit -m "fix: Clear VIP cache on startup to fetch fresh prices"
git push

```text

1.**Apply watchlist prediction lookup:**```bash

# Edit api/cockpit_v3_live_endpoints.py lines 1920-1970

# Add prediction store lookup after price fetch

git add api/cockpit_v3_live_endpoints.py
git commit -m "feat: Add Ghost scores to watchlist enriched endpoint"
git push

```text

1.**Create personal watchlist migration:**```bash

# Create migrations/003_personal_watchlist.sql

git add migrations/003_personal_watchlist.sql
git commit -m "feat: Add personal watchlist database tables"
git push

```text

### Medium-Term (Next 2 Hours)

1.**Test VIP panel:**```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/vip/snapshot>>>>> | jq

```text

1.**Test watchlist enriched:**```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/watchlist/enriched>>>>> | jq

```text

1.**Verify predictions flowing:**```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=BTC>>>>> | jq

```text

### Long-Term (Next 24 Hours)

1.**Monitor accuracy pipeline:**- Check `/api/v3/accuracy/summary` daily

- Expect "No reconciled predictions found" for 48 hours (normal)
- After Dec 4, should show accuracy metrics


1.**Test personal watchlist UI:**- Open Cockpit V3 → Personal tab

- Add XRP to watchlist
- Verify Ghost score appears
- Check Railway logs for `[WATCHLIST] Added symbol: XRP`


---

## APPENDIX A — FILE CHANGE SUMMARY

| File | Lines Changed | Type | Priority |
|------|---------------|------|----------|
| `api/cockpit_v3_live_endpoints.py` | 1142-1143 | Fix | 🔴 Critical |
| `api/cockpit_v3_live_endpoints.py` | 1920-1970 | Feature | 🟡 High |
| `wolf_app.py` | Startup section | Fix | 🟡 High |
| `migrations/003_personal_watchlist.sql` | New file | Feature | 🟡 High |
| `static/cockpit_v3.js` | 350-380 | Debug | 🟢 Low |
| `static/cockpit_v3.js` | 420-430 | Enhancement | 🟢 Low |

---

## APPENDIX B — API ENDPOINT STATUS

| Endpoint | Status | Used By | Notes |
|----------|--------|---------|-------|
| `/api/v3/vip/snapshot` | ⚠️ Cache Stale | VIP Panel | Returns offline status (needs cache clear) |
| `/api/v3/watchlist/enriched` | ⚠️ Missing Predictions | Watchlist Panel | Prices work, Ghost scores missing |
| `/api/v3/predictions/latest` | ✅ Working | Top Movers, Forecast | Stock predictions flowing |
| `/api/v3/news/feed` | ✅ Working | News Feed | Predictions shown as news items |
| `/api/v3/health/score` | ⚠️ Placeholders | Health Panel | Daily/Weekly/Monthly real, others hardcoded |
| `/api/v3/accuracy/summary` | ✅ Expected | Accuracy Panel | "No data" expected (Day 0) |

---**Document Version:**1.0**Last Updated:**Dec 2, 2025 23:55 UTC**Next Review:** After Railway redeploy (Dec 3, 2025)
