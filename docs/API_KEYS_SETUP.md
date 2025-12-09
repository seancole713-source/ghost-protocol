# Ghost Protocol - API Keys Setup Guide

## Overview

Ghost Protocol uses multiple data providers to fetch prices, technical indicators, and news. This guide explains how to
configure API keys for optimal system performance.

---

## Provider Hierarchy

Ghost uses a **fallback chain**for price data:

1.**Polygon.io**(Paid - Primary for stocks)
2.**Alpha Vantage**(Free tier - Primary for crypto + news)
3.**Yahoo Finance Scraper**(Free - HTTP scraping)
4.**yfinance Library**(Free - Python library)

When API keys are**not configured**, Ghost falls back to free sources (Yahoo + yfinance).

---

## Current Configuration Status

Run this command to check your API key configuration:

```bash
curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/system/diagnostics>>>>> | jq '.api_keys'

```text

**Expected Output:**```json

{
  "POLYGON_KEY": true,
  "ALPHAVANTAGE_KEY": true,
  "ALPHA_VANTAGE_API_KEY": true
}

```text

---

## API Key Configuration

### 1. Polygon.io (Recommended for Stocks)**Purpose:**Real-time stock prices, historical OHLCV data**Free Tier:**No (paid plans start at $29/month)**Sign Up:**<<<<<https://polygon.io/>**Railway>>>> Configuration:**1. Go to Railway dashboard: <<<<<https://railway.app/project/ghost-protocol-production>>>>>

1. Click on your service →**Variables**tab
2. Add variable:


   -**Key:**`POLYGON_KEY`
   -**Value:**`your_polygon_api_key_here`

1. Click**Deploy**


**Testing:**```bash

# Test if Polygon is working

curl "<<<<<https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey=YOUR_KEY">>>>>

```text

---

### 2. Alpha Vantage (Recommended for Crypto + News)**Purpose:**Crypto prices, stock prices, news sentiment**Free Tier:**Yes (5 API calls/minute, 500 calls/day)**Sign Up:**<<<<<https://www.alphavantage.co/support/#api-key>**Railway>>>> Configuration:**1. Go to Railway dashboard →**Variables**tab

1. Add TWO variables (both needed):


   -**Key:**`ALPHAVANTAGE_KEY`**Value:**`your_alphavantage_key_here`

   -**Key:**`ALPHA_VANTAGE_API_KEY`**Value:**`your_alphavantage_key_here` (same key)

1. Click**Deploy**


**Why two variables?**- `ALPHAVANTAGE_KEY` used by price engine (wolf_app.py)

- `ALPHA_VANTAGE_API_KEY` used by news sentiment (core/news_sentiment.py)**Testing:**```bash


# Test stock price

curl "<<<<<https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=YOUR_KEY">>>>>

# Test news sentiment

curl "<<<<<https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=YOUR_KEY">>>>>

```text

---

## Ghost Score Impact

Ghost Score is calculated from:

-**Data Quality (40%):**% of symbols with valid price data
-**Prediction Coverage (35%):**% of symbols with predictions
-**Risk Behavior (25%):**Position sizing + drawdown compliance**Without API keys:**- Ghost relies on free sources (Yahoo + yfinance)

- Success rate: ~50-70% (rate limited, less reliable)
- Ghost Score: 40-60 (F/D grade)**With API keys configured:**- Ghost uses paid providers (Polygon + Alpha Vantage)
- Success rate: ~80-95%
- Ghost Score: 65-85 (C/B grade)


---

## Verifying Setup

### 1. Check API Key Status

```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/system/diagnostics>>>>> | jq '.api_keys'

```text

### 2. Check Provider Health

```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/providers/health>>>>>

```text

### 3. Warm Up Predictions (Test All Symbols)

```bash

cd /Users/studio713/ghost-protocol
python3 scripts/warm_up_predictions.py

```text**Expected output with API keys:**```text

✅ Success: 23/25 (92%)
❌ Failed:  2/25

```text

### 4. Check Ghost Score

```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/goals/snapshot>>>>> | jq '.ghost_score'

```text**Expected with API keys:**65-85 (C/B grade)

---

## Troubleshooting

### Issue: `POLYGON_KEY` shows `false` in diagnostics**Solution:**1. Verify you added the variable in Railway dashboard

1. Click**Deploy**after adding variables
2. Wait 60 seconds for deployment
3. Re-check diagnostics endpoint


### Issue: Predictions still failing with API keys configured**Possible causes:**1.**Rate limiting:**Alpha Vantage free tier = 5 calls/min

   - Solution: Upgrade to paid tier or add delays between calls


1.**Invalid API key:**Test keys manually with curl commands above
2.**Symbol format issues:**Some crypto symbols need suffix (BTC-USD vs BTC)**Debug commands:**```bash

# Check failing symbols

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/system/diagnostics>>>>> | jq '.prediction_stats.failing_symbols'

# Test specific symbol prediction

curl -X POST "<<<<<https://ghost-protocol-production.up.railway.app/api/predict/run">>>>> \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL"}'

```text

### Issue: News feed returns empty array**Cause:**`ALPHA_VANTAGE_API_KEY` environment variable not set**Solution:**1. Add `ALPHA_VANTAGE_API_KEY` variable in Railway (same value as `ALPHAVANTAGE_KEY`)

1. Deploy and wait 60 seconds
2. Test: `curl "<<<<<https://ghost-protocol-production.up.railway.app/api/v3/news/feed?symbol=AAPL&limit=5"`>>>>>


---

## Cost Estimation

| Provider | Free Tier | Paid Tier | Recommended |
|----------|-----------|-----------|-------------|
|**Polygon.io**| ❌ No | $29/month (Starter) | Yes (stocks) |
|**Alpha Vantage**| ✅ Yes (500/day) | $49/month (Unlimited) | Yes (crypto + news) |
|**Yahoo/yfinance**| ✅ Yes | Free | Fallback only |**Total cost for optimal performance:**$29-78/month

---

## Phase-2 Status

### ✅ Completed Fixes

- [x] Ghost Score calculation (batch → individual predictions)
- [x] Provider fallback logic (prioritize free sources when no keys)
- [x] News feed integration (Alpha Vantage NEWS_SENTIMENT API)
- [x] System diagnostics endpoint
- [x] API key documentation


### 📊 Current Metrics (Without API Keys)

- Ghost Score: 41-52 (F)
- Prediction coverage: 12/47 symbols (26%)
- Success rate: 48-52%
- News feed: Empty (no API key)


### 🎯 Expected Metrics (With API Keys)

- Ghost Score: 65-85 (C/B)
- Prediction coverage: 40-45/47 symbols (85-95%)
- Success rate: 80-95%
- News feed: 5-10 articles per symbol


---

## Next Steps

1.**Get API keys:**- Sign up for Alpha Vantage (free): <<<<<https://www.alphavantage.co/support/#api-key>>>>>

   - Consider Polygon.io (paid): <<<<<https://polygon.io/pricing>>>>>


1.**Configure Railway:**- Add `POLYGON_KEY` (if using Polygon)

   - Add `ALPHAVANTAGE_KEY` (required)
   - Add `ALPHA_VANTAGE_API_KEY` (same as above - for news)


1.**Deploy and verify:**- Wait 60 seconds after adding variables

   - Run warm-up script: `python3 scripts/warm_up_predictions.py`
   - Check Ghost Score: `curl .../api/v3/goals/snapshot`


1.**Monitor:**

   - Use `/api/v3/system/diagnostics` endpoint daily
   - Watch for rate limit errors in logs
   - Track Ghost Score trends over time


---

## Support

- Diagnostic Endpoint: `/api/v3/system/diagnostics`
- Provider Health: `/api/v3/providers/health`
- Ghost Score: `/api/v3/goals/snapshot`
- Warm-Up Script: `scripts/warm_up_predictions.py`


For issues, check Railway logs:

```bash

railway logs --project ghost-protocol-production

```text
