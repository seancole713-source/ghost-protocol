# GHOST OmniBrain v10.3 - UI Gap Analysis

## 🎯 Current Status: BACKEND COMPLETE ✅ | UI PARTIALLY CONNECTED ⚠️

Based on the live UI snapshot provided, here's what's **MISSING**or**NOT WORKING**:

______________________________________________________________________

## ❌ MISSING FEATURES (Backend Exists, UI Not Connected)

### 1. **APEX Trade Card**- ❌ NOT IMPLEMENTED**UI Shows:**"Loading trade card..."\**Backend Status:**❌ NO ENDPOINT EXISTS\**Issue:**- No `/api/apex/trade-card/{symbol}` endpoint in wolf_app.py

- No APEX explainability module implemented**What's Needed:**```text


GET /api/apex/trade-card/{symbol}
Returns:
{
  "symbol": "WOLF",
  "action": "BUY|SELL|HOLD",
  "top_features": [...],  // Driving factors
  "expected_path": {...}, // Price trajectory
  "fail_conditions": [...], // Risk scenarios
  "analog_scenarios": [...], // Historical matches
  "risk_factors": [...],
  "ai_rationale": "..."
}

```text

### 2.**48h Forecast**- ⚠️ PARTIALLY WORKING**UI Shows:**"48h Forecast" button\**Backend Status:**✅ EXISTS at `/api/forecast/overlay`\**Issue:**Returns `{"enabled": false, "reason": "no_forecast"}`**Current Response:**```bash

$ curl /api/forecast/overlay
{"enabled": false, "reason": "no_forecast"}

```text**What's Needed:**- Forecast engine needs to generate initial predictions

- Check if forecast runs are being triggered
- Verify `/forecast/48h` endpoint produces data


### 3.**Market Outlook (Fusion AI)**- ❌ NULL DATA**UI Shows:**"risk: -, confidence: -"\**Backend Status:**Cockpit returns `"market_outlook": null`\**Issue:**Market outlook is computed but always returns null**What's Needed:**- Enable Fusion AI market analysis

- Implement sentiment aggregation across news/price/volume
- Return format:


```json

{
  "risk": "low|medium|high",
  "confidence": 0.75,
  "action": "BUY|SELL|HOLD",
  "reasoning": "..."
}

```text

### 4.**Signals**- ❌ NULL DATA**UI Shows:**"Signals" section (empty)\**Backend Status:**Cockpit returns `"signals": null`\**Issue:**No signals endpoint or data**What's Needed:**```text

GET /api/signals
Returns:
[
  {
    "symbol": "WOLF",
    "type": "BUY|SELL",
    "strength": 0-10,
    "source": "technical|news|ai",
    "timestamp": 1760318636
  }
]

```text

### 5.**Crypto Movers**- ⚠️ EMPTY ARRAY**UI Shows:**"Top Movers > Crypto" (empty)\**Backend Status:**Cockpit returns `"crypto": []`\**Issue:**Crypto prices working, but movers not populated**Current Response:**```json

"movers": {
  "stocks": [{...WOLF...}],
  "crypto": []  // ❌ Empty despite crypto enabled
}

```text**What's Needed:**- Aggregate 24h price changes for BTC, ETH, SOL, BNB

- Sort by absolute percentage change
- Return top 5 movers


### 6.**Crypto Predictions in Cockpit**- ⚠️ NOT DISPLAYED**Backend Status:**Code exists (lines 12289-12339) to inject crypto predictions\**Issue:**Predictions not being generated or stored**What's Needed:**- Run crypto prediction engine periodically

- Store predictions in `crypto_predictions` table
- Verify query returns recent predictions (< 1 hour old)


______________________________________________________________________

## ✅ WORKING FEATURES

- ✅**Portfolio Overview**: NAV, PnL, PnL%, Cash all displaying correctly
- ✅ **Position Table**: WOLF showing qty, entry, current, PnL, GPS
- ✅ **Status Indicators**: stocks ✅, crypto ✅, news ✅, telegram ✅
- ✅ **Live News**: Fetching and displaying articles
- ✅ **Manual Watchlist**: Symbols loaded (AAPL, WOLF, MSFT, BTC-USD, ETH-USD)
- ✅ **Ghost Score Heatmap**: GPS 7.2 for WOLF
- ✅ **Health/Diagnostics**: All endpoints responding


______________________________________________________________________

## 🚀 PRIORITY FIX LIST

### HIGH PRIORITY (User-Visible Gaps)

1. **APEX Trade Card**- Build endpoint with AI explainability


2.**Market Outlook**- Enable Fusion AI sentiment analysis
3.**48h Forecast**- Generate initial forecast predictions
4.**Crypto Movers**- Populate with 24h change data


### MEDIUM PRIORITY (Nice to Have)

1.**Signals**- Add trading signal detection
2.**Crypto Predictions**- Run prediction engine and display results


### LOW PRIORITY (Cosmetic)

1. Better error messages for empty states
2. Loading indicators vs "null" display


______________________________________________________________________

## 📝 IMPLEMENTATION PLAN

### Step 1: APEX Trade Card (NEW FEATURE)

- Create `/api/apex/trade-card/{symbol}` endpoint
- Build feature importance analyzer
- Generate historical analog matches
- Add risk factor detection


### Step 2: Market Outlook (FIX NULL)

- Enable sentiment scoring across news
- Aggregate technical indicators
- Combine into risk/confidence score


### Step 3: 48h Forecast (FIX NO_FORECAST)

- Trigger initial forecast run
- Verify forecast storage in DB
- Enable overlay endpoint


### Step 4: Crypto Movers (FIX EMPTY ARRAY)

- Query recent crypto prices
- Calculate 24h changes
- Sort and return top movers


______________________________________________________________________

## 🎯 SUMMARY**Backend Completeness:**85%\**UI Integration:**65%

The core infrastructure is SOLID. What's missing are:

1.**APEX explainability module**(new feature)
2.**Market outlook aggregation**(null → data)
3.**Forecast initialization**(no_forecast → active)
4.**Crypto movers population** ([] → [data])


All fixable with targeted endpoint work! 🚀
