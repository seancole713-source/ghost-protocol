# GHOST PROVIDER ARCHITECTURE - BEFORE STATE

**Date**: November 25, 2025
**Analysis By**: Ghost Provider Architect

---

## CURRENT PROVIDER STACK

### Stock/ETF/Index Providers

**Primary**: wolf_app.py `_get_price_quorum()`

- Priority: Polygon → Yahoo → AlphaVantage → yfinance
- Recent fix (commit 38e2f33): Inverted priority Polygon-first
- Status: ✅ Working when providers available


**Spot Price Providers**:

1. **Polygon**(`POLYGON_API_KEY` required)
   - Coverage: US stocks, ETFs, indices
   - Rate: Free tier limited, paid unlimited
   - Status: Configured but hitting quotas


1.**Yahoo Finance**(HTTP scraper)

   - Coverage: Global stocks
   - Rate: Rate-limited (429 errors frequent)
   - Status: ⚠️ Unreliable in production


1.**AlphaVantage**(`ALPHAVANTAGE_API_KEY` optional)

   - Coverage: US/global stocks
   - Rate: Free tier very limited
   - Status: Emergency fallback only


1.**yfinance**(library)

   - Coverage: Global stocks
   - Rate: Subject to Yahoo rate limits
   - Status: ⚠️ Unreliable**Historical OHLCV**:

- `technical_engine.py`: Polygon → yfinance → CoinGecko (crypto)
- `volume_engine.py`: Polygon → yfinance → CoinGecko (crypto)
- Status: ✅ Working for Polygon, ⚠️ failing for Yahoo


### Crypto Providers

**Spot Price**: `core/crypto/crypto_providers.py`

**Provider Order**(configurable via `CRYPTO_QUORUM`):

1.**CoinGecko**(primary)

   - Coverage: 10,000+ coins
   - Rate: Free 50 calls/min
   - Status: ✅ Working


1.**Binance**(secondary)

   - Coverage: Major pairs (BTCUSDT, ETHUSDT, etc.)
   - Rate: Unlimited public API
   - Status: ✅ Working for spot


1.**Coinbase**(tertiary)

   - Coverage: Limited major coins
   - Rate: Unlimited public API
   - Status: ✅ Working**Historical OHLCV**:

- ❌ **NOT IMPLEMENTED**- Current: Falls back to Yahoo (doesn't have crypto)
- Result: Crypto predictions get 0/16 technical, 0/5 volume features


---

## FAILURE MODES IDENTIFIED

### 1. Yahoo Finance Rate Limiting (CRITICAL)**Symptoms**

```text
429 Client Error: Too Many Requests
[TECH] NVDA: ALL PROVIDERS FAILED - ['polygon', 'yahoo']

```text

**Impact**:

- Stocks: 5/27 features (18.5%) when all providers fail
- Result: 40% FLAT predictions


**Cause**: Railway IP hitting Yahoo's aggressive rate limits

### 2. Polygon API Quota Exhaustion (HIGH)

**Symptoms**:

```text

[WARN] [TECH] AAPL: Polygon failed, trying Yahoo
Polygon: HTTP 403 (quota exceeded)

```text

**Impact**:

- Falls back to rate-limited Yahoo
- Same result as #1


**Cause**: Free tier only 5 calls/min

### 3. Crypto OHLCV Missing (CRITICAL)

**Symptoms**:

```text

[ERRO] [TECH] BTC: ALL PROVIDERS FAILED - ['yahoo', 'coingecko']
[WOLF] Extracted 5/25 features (crypto)

```text

**Impact**:

- BTC/ETH predictions: 40% FLAT
- Only get price + sentiment (5/25 features)
- Missing all technical indicators


**Cause**: No Binance OHLCV integration

### 4. No Caching Layer (MEDIUM)

**Symptoms**:

- Every prediction = fresh API calls
- Repeated calls for same symbol within seconds
- Rapid quota exhaustion


**Impact**:

- 3-5x more API calls than needed
- Rate limits hit faster
- Higher latency (500-1000ms per prediction)


**Cause**: No Redis caching implemented

### 5. Single Provider Failure Kills Feature (LOW)

**Symptoms**:

```text

yfinance failed for MSFT: 429
technical_engine returns 0/16 features

```text

**Impact**:

- One provider down = entire pillar fails
- No graceful degradation


**Cause**: Insufficient retry logic (now partially fixed)

---

## PROVIDER HEALTH MATRIX

| Provider | Type | Coverage | Rate Limit | Reliability | OHLCV | Cost |
|----------|------|----------|------------|-------------|-------|------|
| **Polygon**| Stock | US | 5/min free | 🟢 High | ✅ Yes | Free/Paid |
|**Yahoo**| Stock | Global | Aggressive | 🔴 Low | ⚠️ Unstable | Free |
|**AlphaVantage**| Stock | Global | 5/min | 🟡 Medium | ✅ Limited | Free/Paid |
|**yfinance**| Stock | Global | Yahoo limits | 🔴 Low | ⚠️ Unstable | Free |
|**CoinGecko**| Crypto | 10k+ | 50/min | 🟢 High | ❌**No**| Free/Paid |
|**Binance**| Crypto | Major | Unlimited | 🟢 High | ❌**No**| Free |
|**Coinbase**| Crypto | Limited | Unlimited | 🟡 Medium | ❌**No**| Free |

---

## CURRENT ARCHITECTURE DIAGRAM

```text

┌─────────────────────────────────────────────────────┐
│                  GHOST PREDICTION                   │
│                   (wolf_app.py)                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ├─────► Feature Orchestrator
                   │       (get_all_features)
                   │               │
                   │               ├─────► Price Engine
                   │               │       └─► _get_price_quorum()
                   │               │           ├─► Polygon
                   │               │           ├─► Yahoo (429!)
                   │               │           └─► yfinance
                   │               │
                   │               ├─────► Technical Engine
                   │               │       └─► _fetch_historical_data()
                   │               │           ├─► Polygon (5/min)
                   │               │           ├─► yfinance (429!)
                   │               │           └─► CoinGecko (crypto, NO OHLCV!)
                   │               │
                   │               ├─────► Volume Engine
                   │               │       └─► _fetch_historical_data()
                   │               │           ├─► Polygon (5/min)
                   │               │           ├─► yfinance (429!)
                   │               │           └─► Coin Gecko (crypto, NO OHLCV!)
                   │               │
                   │               ├─────► Sentiment Engine
                   │               │       └─► NewsAPI (working)
                   │               │
                   │               ├─────► World Context
                   │               │       └─► get_stock_price (SPY/VIX)
                   │               │
                   │               └─────► Flow Engine
                   │                       └─► (Not implemented)
                   │
                   └─────► Crypto Price Quorum
                           (crypto_providers.py)
                           ├─► CoinGecko (spot only)
                           ├─► Binance (spot only)
                           └─► Coinbase (spot only)

```text

---

## CONFIGURATION

### Environment Variables**Required**

- `POLYGON_API_KEY` - Polygon.io API key (free/paid)
- `REDIS_URL` - Upstash Redis URL (caching)


**Optional**:

- `ALPHAVANTAGE_API_KEY` - AlphaVantage fallback
- `CRYPTO_QUORUM` - Crypto provider order (default: coingecko,binance,coinbase)
- `PRICE_YAHOO_FIRST` - Use Yahoo before Polygon (deprecated, now ignored)


**Missing**(Need to add):

- `BINANCE_API_KEY` - Not required for public API, but good for rate limits
- `CACHE_TTL_PRICE` - Price cache TTL (should be 30-90s)
- `CACHE_TTL_OHLCV` - OHLCV cache TTL (should be 5min)


---

## FEATURE EXTRACTION STATS (CURRENT)

### Stocks (When Working)**AAPL**(Polygon success)

```text

Available: 25/26 (96.2%)
  Price Engine: 2/2
  Technical Engine: 9/16  ← Limited by lookback
  Volume Engine: 5/5
  Sentiment Engine: 2/2
  World Context: 1/1

```text

### Stocks (When Failing)**NVDA**(All providers fail)

```text

Available: 5/27 (18.5%)
  Price Engine: 2/2  ← AlphaVantage fallback
  Technical Engine: 0/16  ← NO OHLCV
  Volume Engine: 0/5  ← NO OHLCV
  Sentiment Engine: 2/2
  World Context: 1/1
Result: 40% FLAT prediction

```text

### Crypto (Always Failing)**BTC**

```text

Available: 5/25 (20%)
  Price Engine: 1/1  ← Binance spot works
  Technical Engine: 0/16  ← NO OHLCV!
  Volume Engine: 0/5  ← NO OHLCV!
  Sentiment Engine: 2/2
  World Context: 1/1
Result: 40% FLAT prediction

```text

---

## ROOT CAUSES SUMMARY

1. **Yahoo Rate Limiting**: Primary historical data source failing
2. **Polygon Quota**: Free tier exhausted quickly
3. **No Crypto OHLCV**: Binance/CoinGecko not integrated for historical data
4. **No Caching**: 3-5x redundant API calls
5. **Insufficient Retries**: Single failure = feature loss


---

## NEXT: DESIGN NEW ARCHITECTURE

See `GHOST_PROVIDER_ARCHITECTURE_AFTER.md` for the hardened provider stack design.

**Target Metrics**:

- Stocks: 24-26/26 features (92-100%)
- Crypto: 20-25/25 features (80-100%)
- Prediction variance: 45-75% confidence (not 40%)
- Direction variance: UP/DOWN/FLAT mix (not all FLAT)
- API call reduction: 80% via caching
- Provider success rate: 95%+ uptime
