# 🎯 GHOST UI - What's Missing (Quick Reference)

## ❌ **NOT WORKING (User Sees Issues)**

### 1. **APEX Trade Card** → "Loading trade card..."

- **Problem:** No `/api/apex/trade-card/{symbol}` endpoint
- **Fix:** Build APEX explainability module (NEW FEATURE)

### 2. **48h Forecast** → Returns "no_forecast"

- **Problem:** No forecast data in database
- **Fix:** Initialize forecast engine, generate first prediction

### 3. **Market Outlook** → Shows "risk: -, confidence: -"

- **Problem:** `market_outlook` returns `null` in cockpit
- **Fix:** Enable Fusion AI sentiment aggregation

### 4. **Signals** → Empty section

- **Problem:** `signals` returns `null` in cockpit
- **Fix:** Add `/api/signals` endpoint with trading signals

### 5. **Crypto Movers** → Empty list

- **Problem:** `movers.crypto` returns `[]` despite crypto enabled
- **Fix:** Fetch 24h crypto changes and populate movers

### 6. **Crypto Predictions** → Not visible

- **Problem:** No crypto predictions in database
- **Fix:** Run crypto prediction engine, store results

______________________________________________________________________

## ✅ **WORKING PERFECTLY**

- Portfolio (NAV, PnL, positions)
- Status indicators (all green)
- Live news feed
- Price data (stocks + crypto)
- Watchlist
- GPS heatmap
- Health/diagnostics

______________________________________________________________________

## 🚀 **QUICK WIN: Top 3 Fixes**

1. **Crypto Movers** (30 min) - Just fetch prices and calculate changes
2. **Market Outlook** (1 hour) - Aggregate existing news sentiment
3. **48h Forecast** (2 hours) - Initialize forecast engine

**APEX Trade Card** is a bigger feature (4-6 hours) but high impact!
