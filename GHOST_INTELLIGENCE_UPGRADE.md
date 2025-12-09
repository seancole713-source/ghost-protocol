# 🧠 GHOST INTELLIGENCE UPGRADE PLAN

**Date**: October 13, 2025\
**Goal**: Make Ghost smarter with better predictions and real-time market awareness

______________________________________________________________________

## 📊 Current State Analysis

### What Ghost Got Wrong

- **Predicted**: $31.10 (previous close)
- **Actual**: $30.50 (-1.93% down)
- **Missed**: 18% intraday swing ($28.80-$34.19)
- **Opening gap**: $33.52 → $30.50 (9% drop in minutes)

### Root Causes

1. ✅ **API keys present**(AlphaVantage: `3WNNLA81KS7BG4AK`, Polygon:

   `8VIvELVXiLG30K2l1348RzSurffLM0jR`)

1. ❌**Yahoo Finance rate limiting**(429 errors blocking real-time data)
1. ❌**Stale cache**(returning yesterday's close instead of live price)
1. ❌**No intraday volatility tracking**(missing high/low/volume data)
1. ❌**Simple drift model**(doesn't account for gaps, momentum, volatility)

______________________________________________________________________

## 🎯 Improvement Strategy

### Phase 1: Fix Real-Time Data (IMMEDIATE)**Priority**: 🔴 CRITICAL

1. **Switch primary provider from Yahoo to AlphaVantage**- Set `PRICE_YAHOO_FIRST=0` in Railway
   - AlphaVantage has 25 requests/day free tier (no rate limiting for hourly checks)

1.**Add intraday price tracking**- Fetch high/low/volume from AlphaVantage `TIME_SERIES_INTRADAY`

- Store in `realized_prices` table with metadata

1.**Reduce cache TTL during market hours**- Current: 5 minutes

- New: 1 minute during trading hours (9:30 AM - 4 PM ET)
- Off-hours: Keep 5 minutes**Expected Impact**: Real-time prices within 1 minute, no more stale $31.10

______________________________________________________________________

### Phase 2: Enhance Prediction Model (HIGH PRIORITY)

**Priority**: 🟡 HIGH

1. **Add volatility-aware forecasting**- Calculate ATR (Average True Range) from intraday data
   - Widen confidence bands when ATR > threshold
   - Example: 18% intraday range → 3x wider bands

1.**Detect and model price gaps**- Track overnight gaps (close → open)

- Adjust drift based on gap direction
- WOLF gap: $31.10 → $33.52 (+7.8%) then crashed

1.**Volume-weighted momentum**- High volume + price drop = strong sell signal

- Today: 2.7M volume (21% of avg 12.9M) → weak conviction

1.**News sentiment integration**- Current: Basic sentiment score

- New: Real-time news impact scoring
- Weight: Recent news (< 1hr) gets 3x weight**Expected Impact**: MAP drops from current ~15% to \<10%

______________________________________________________________________

### Phase 3: Market Context Intelligence (MEDIUM)

**Priority**: 🟢 MEDIUM

1. **Sector correlation**- Track SOX index (semiconductor sector)
   - When sector down, WOLF likely follows
   - Today: Likely sector-wide weakness

1.**Market regime detection**- Bull market: Lower SL threshold (-5%)

- Bear market: Tighter SL (-2%)
- High volatility: Widen TP (+10%)

1.**Earnings calendar awareness**- Next WOLF earnings: Aug 20, 2025

- Increase volatility bands 2 weeks before/after

1.**Options flow signals**(future)

- Unusual options activity
- Put/call ratio shifts**Expected Impact**: Ghost understands "why" moves happen, not just "what"

______________________________________________________________________

### Phase 4: Adaptive Learning (ADVANCED)

**Priority**: 🔵 LOW

1. **Prediction error tracking**- Store forecast vs actual in database
   - Calculate rolling MAP per time horizon
   - Auto-adjust model parameters

1.**Regime-specific models**- Train separate models for:

     - Bull market
     - Bear market
     - High volatility
     - Low volatility

1.**Feature importance analysis**- Which signals matter most?

- News sentiment? Volume? Gaps?
- Drop weak features, amplify strong ones**Expected Impact**: Ghost learns from mistakes, gets smarter over time

______________________________________________________________________

## 🛠️ Implementation Plan

### Immediate Fixes (Today)

```bash

# 1. Switch to AlphaVantage primary

railway variables --set "PRICE_YAHOO_FIRST=0"

# 2. Reduce cache TTL for real-time updates

railway variables --set "PRICE_CACHE_TTL=60"

# 3. Enable intraday data fetching

railway variables --set "INTRADAY_TRACKING=1"

```text

### Code Changes Required

1. **Add intraday data fetcher**:


   ```python

   def _fetch_intraday_data(symbol: str) -> dict:
       """Fetch high, low, volume, VWAP from AlphaVantage"""
       url = f"<<<<<https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={ALPHAVANTAGE_KEY}">>>>>

       # Parse and return {high, low, volume, vwap}

   ```text

1. **Enhance forecast with volatility**:


   ```python

   def _generate_forecast_grid(symbol: str = WOLF) -> dict:

       # Current: Simple drift model

       # New: drift + ATR-based band widening

       atr = _calculate_atr(symbol, period=14)
       band_multiplier = 1.0 + (atr / price * 10)  # Scale bands by volatility

   ```text

1. **Real-time price override during market hours**:


   ```python

   def get_wolf_price():

       # If market open and cache > 1 min, force refresh

       if _is_market_open_now() and cache_age > 60:
           quotes = []  # Force provider fetch

   ```text

______________________________________________________________________

## 📈 Success Metrics

### Before (Current)

- **Price freshness**: 5+ minutes stale
- **Prediction MAP**: ~15%
- **Intraday awareness**: None (only EOD data)
- **Gap detection**: None
- **Volatility adjustment**: Fixed bands


### After (Target)

- **Price freshness**: \<1 minute during market hours
- **Prediction MAP**: \<10%
- **Intraday awareness**: High/low/volume tracked every 5 min
- **Gap detection**: Opening gaps identified and modeled
- **Volatility adjustment**: Dynamic bands based on ATR


______________________________________________________________________

## 🚀 Quick Wins (Next 30 Minutes)

1. **Switch to AlphaVantage**→ Instant real-time data


2.**Reduce cache TTL**→ 1-minute refresh during trading
3.**Add gap detection**→ Catch opening moves like $33.52 → $30.50


______________________________________________________________________

## 📝 Long-Term Vision

Ghost should:

-**See**real-time price moves as they happen
-**Understand**why moves occur (news, sector, volume)
-**Predict**with context (volatility, regime, earnings)
-**Learn**from mistakes (adaptive parameters)
-**Explain**reasoning (transparent AI decisions)


______________________________________________________________________**Status**: 🟡 READY TO IMPLEMENT\
**Estimated Time**: 2-3 hours for Phase 1+2\
**Risk**: Low (all changes are additive, no breaking changes)
