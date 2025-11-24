# GHOST FEATURE PIPELINE MAP

**Date:** November 24, 2025  
**Purpose:** Complete map of the feature extraction pipeline showing what Ghost needs to produce quality predictions

---

## ARCHITECTURE OVERVIEW

Ghost's prediction engine uses a **6-pillar feature orchestrator** that extracts ~40 technical features from multiple data sources:

```
┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION REQUEST                        │
│               /api/predict/run (symbol)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE ORCHESTRATOR                           │
│         core/data_pillars/feature_orchestrator.py           │
│                                                              │
│  Coordinates all 6 data pillars in parallel/sequential     │
│  Returns: {features: {...}, available_count, errors}        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │PILLAR 1│  │PILLAR 2│  │PILLAR 3│
    │ PRICE  │  │TECHNICAL│  │ VOLUME│
    └────────┘  └────────┘  └────────┘
        │            │            │
        ▼            ▼            ▼
    ┌────────┐  ┌────────┐  ┌────────┐
    │PILLAR 4│  │PILLAR 5│  │PILLAR 6│
    │SENTIMENT│  │ WORLD  │  │  FLOW │
    └────────┘  └────────┘  └────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              CONFIDENCE & DIRECTION LOGIC                   │
│                wolf_app.py:5930-6020                        │
│                                                              │
│  Uses features to calculate:                                │
│  - direction: UP/DOWN/FLAT                                 │
│  - confidence: 0.40-0.85 (40-85%)                          │
│  - signal_strength: count of aligned indicators            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREDICTION OUTPUT                         │
│  {prediction_id, symbol, direction, confidence, forecast}   │
│              Saved to ghost_predictions.db                  │
└─────────────────────────────────────────────────────────────┘
```

---

## PILLAR 1: PRICE ENGINE

**File:** `core/data_pillars/price_engine.py`  
**Purpose:** Multi-provider price data with consensus quorum  
**Status:** ⚠️ PARTIALLY WORKING

### Signals Provided

| Signal Name | Description | Data Type | Availability |
|------------|-------------|-----------|--------------|
| `PRICE` | Current market price | float | ✅ High |
| `PREV_CLOSE` | Previous close price | float | ✅ High |
| `BID_ASK_SPREAD` | Bid/ask spread % | float | ❌ Limited |
| `VWAP` | Volume-weighted average | float | ❌ Limited |
| `PROVIDER_QUALITY` | Provider quality score | float | ❌ Not implemented |
| `STALENESS_SECONDS` | Data age | int | ❌ Not implemented |
| `MARKET_CAP` | Market capitalization (crypto) | float | ✅ Moderate |
| `VOLUME_24H` | 24h trading volume (crypto) | float | ✅ Moderate |

### Provider Stack

**Stocks:**
- Polygon (primary)
- Alpha Vantage (fallback)
- Yahoo Finance (tertiary)

**Crypto:**
- CoinGecko (primary)
- Binance (fallback)
- Coinbase (tertiary)

### Known Issues

1. **No Bollinger Position**: Missing `BOLLINGER_POSITION` feature (referenced in confidence logic)
2. **Limited Fallback**: If primary provider fails, entire pillar may fail
3. **No Caching**: Every prediction refetches prices (no TTL cache)

---

## PILLAR 2: TECHNICAL ENGINE

**File:** `core/data_pillars/technical_engine.py`  
**Purpose:** Calculate 50+ technical indicators from historical OHLCV data  
**Status:** ⚠️ DEGRADED (only extracting 5/15 indicators)

### Signals Provided

| Signal Name | Description | Calculation | Availability |
|------------|-------------|-------------|--------------|
| `RSI_14` | Relative Strength Index | 14-period RSI | ⚠️ Intermittent |
| `MACD_HISTOGRAM` | MACD histogram value | MACD(12,26) - Signal(9) | ⚠️ Intermittent |
| `MACD_SIGNAL` | MACD signal line | EMA of MACD(9) | ⚠️ Intermittent |
| `SMA_20` | Simple moving average 20 | 20-period SMA | ❌ Missing |
| `SMA_50` | Simple moving average 50 | 50-period SMA | ❌ Missing |
| `SMA_200` | Simple moving average 200 | 200-period SMA | ❌ Missing |
| `EMA_12` | Exponential MA 12 | 12-period EMA | ❌ Missing |
| `EMA_26` | Exponential MA 26 | 26-period EMA | ❌ Missing |
| `BB_UPPER` | Bollinger upper band | SMA(20) + 2*StdDev | ❌ Missing |
| `BB_MIDDLE` | Bollinger middle band | SMA(20) | ❌ Missing |
| `BB_LOWER` | Bollinger lower band | SMA(20) - 2*StdDev | ❌ Missing |
| `ATR_14` | Average True Range | 14-period ATR | ❌ Missing |
| `STOCH_K` | Stochastic %K | 14-period stochastic | ❌ Missing |
| `STOCH_D` | Stochastic %D | 3-period SMA of %K | ❌ Missing |
| `WILLIAMS_R` | Williams %R | 14-period Williams %R | ❌ Missing |

### Data Source

**Historical Data:** Yahoo Finance (`yfinance` library)
- Lookback: 90 days
- Minimum bars: 20 required for calculations
- Update frequency: On-demand (no cache)

### Known Issues

1. **Insufficient Historical Data**: `yfinance` often returns < 20 bars
2. **No Error Handling**: Single indicator failure kills entire pillar
3. **Missing Indicators**: Only 5/15 indicators actually calculating
4. **No Crypto Support**: Crypto symbols fail with yfinance
5. **Calculation Errors**: Missing try/except wrappers around indicator functions

---

## PILLAR 3: VOLUME ENGINE

**File:** `core/data_pillars/volume_engine.py`  
**Purpose:** Analyze trading volume patterns and volatility  
**Status:** ⚠️ DEGRADED (same yfinance issues as technical)

### Signals Provided

| Signal Name | Description | Calculation | Availability |
|------------|-------------|-------------|--------------|
| `VOLUME_SPIKE` | Current vol vs 20-day avg | (current/avg - 1.0) | ⚠️ Intermittent |
| `VOLATILITY_20D` | 20-day realized volatility | StdDev * sqrt(252) * 100 | ⚠️ Intermittent |
| `VOLATILITY_60D` | 60-day realized volatility | StdDev * sqrt(252) * 100 | ❌ Missing |
| `VOLUME_MA_20` | 20-day volume average | 20-period MA of volume | ⚠️ Intermittent |
| `VOLUME_ROC` | Volume rate of change | 10-day ROC | ❌ Missing |

### Data Source

Same as Technical Engine - Yahoo Finance via `yfinance`

### Known Issues

1. **Shared yfinance Failures**: When technical engine fails, volume engine fails
2. **No Independent Fetch**: Should share OHLCV data with technical engine (performance)
3. **Division by Zero**: No guards for zero volume scenarios

---

## PILLAR 4: SENTIMENT ENGINE

**File:** `core/data_pillars/sentiment_engine.py`  
**Purpose:** Aggregate news sentiment scores  
**Status:** ⚠️ DEGRADED (news API availability issues)

### Signals Provided

| Signal Name | Description | Calculation | Availability |
|------------|-------------|-------------|--------------|
| `NEWS_SENTIMENT_SCORE` | Overall sentiment (-1 to +1) | Aggregated article sentiment | ⚠️ Intermittent |
| `NEWS_COUNT_24H` | Number of articles (24h) | Count of articles | ✅ Moderate |
| `BULLISH_RATIO` | Bullish/total ratio | bullish/(bullish+bearish) | ⚠️ Intermittent |

### Data Source

**Module:** `core/news_sentiment.py`
- Polygon News API (primary)
- Alpha Vantage news (fallback)

### Known Issues

1. **API Key Dependencies**: Requires valid Polygon/Alpha API keys
2. **Rate Limiting**: News APIs have strict rate limits
3. **Low Confidence**: Sentiment is inherently noisy (0.7 confidence)
4. **No Caching**: Refetches news every prediction

---

## PILLAR 5: WORLD CONTEXT ENGINE

**File:** `core/data_pillars/world_context_engine.py`  
**Purpose:** Track global market regime and macro signals  
**Status:** ✅ WORKING

### Signals Provided

| Signal Name | Description | Calculation | Availability |
|------------|-------------|-------------|--------------|
| `SPY_PRICE` | S&P 500 ETF price | Real-time SPY price | ✅ High |
| `SPY_CHANGE` | SPY daily change % | (current - prev)/prev * 100 | ✅ High |
| `VIX_LEVEL` | Volatility Index | Current VIX level | ✅ Moderate |
| `MARKET_REGIME` | Bull/Bear/Neutral classification | SPY+VIX rules | ✅ Moderate |

### Data Source

**Module:** `core/world_context.py`
- Uses price quorum for SPY
- Uses price quorum for VIX

### Known Issues

1. **No BTC Tracking**: Missing BTC as crypto risk proxy (mentioned in design)
2. **No DXY**: Missing dollar index tracking
3. **Simple Regime Logic**: Bull/bear classification could be more sophisticated

---

## PILLAR 6: FLOW ENGINE

**File:** `core/data_pillars/flow_engine.py`  
**Purpose:** Order flow and on-chain metrics  
**Status:** ❌ NOT IMPLEMENTED (requires Level 2 data subscriptions)

### Signals Provided

| Signal Name | Description | Data Type | Availability |
|------------|-------------|-----------|--------------|
| `BID_ASK_SPREAD` | Bid/ask spread % | float | ❌ Unavailable |
| `ORDER_IMBALANCE` | Buy/sell order ratio | float | ❌ Unavailable |
| `WHALE_ACTIVITY` | Large order detection (crypto) | int | ❌ Unavailable |
| `ON_CHAIN_VOLUME` | On-chain transaction volume | float | ❌ Unavailable |

### Data Source Requirements

**Stocks:** Level 2 market data subscription (expensive)  
**Crypto:** Binance/Coinbase orderbook APIs (partially available)

### Current Status

All flow signals return `data_available=False` with error message:  
_"Level 2 data subscription required for orderbook access"_

This pillar is **intentionally degraded** - not a bug, just missing paid data.

---

## FEATURE ORCHESTRATOR

**File:** `core/data_pillars/feature_orchestrator.py`  
**Main Function:** `get_all_features(symbol, **kwargs) -> dict`

### Orchestrator Flow

```python
def get_all_features(symbol: str):
    1. Call price_engine.get_signals(symbol)
    2. Call technical_engine.get_signals(symbol)
    3. Call volume_engine.get_signals(symbol)
    4. Call sentiment_engine.get_signals(symbol)
    5. Call world_context_engine.get_signals()  # Global, no symbol
    6. Call flow_engine.get_signals(symbol)
    
    7. Merge all features into flat dict
    8. Count available_count (non-None values)
    9. Return {features: {...}, available_count, feature_count, errors: [...]}
```

### Return Format

```python
{
    "ok": True,
    "symbol": "AAPL",
    "timestamp": 1732467200.0,
    "features": {
        "PRICE": 185.25,
        "RSI_14": 67.5,
        "MACD_HISTOGRAM": 0.45,
        "VOLUME_SPIKE": 0.23,
        "NEWS_SENTIMENT_SCORE": 0.65,
        "SPY_PRICE": 450.12,
        # ... 40+ more features
    },
    "feature_count": 45,          # Total feature keys
    "available_count": 12,        # Non-None features (PROBLEM!)
    "unavailable_count": 33,      # Missing features
    "feature_availability": {
        "price_engine": "2/8",
        "technical_engine": "5/15",  # LOW!
        "volume_engine": "2/5",
        "sentiment_engine": "1/3",
        "world_context_engine": "2/4",
        "flow_engine": "0/4"
    },
    "execution_time_ms": 234.5,
    "errors": [
        "Insufficient historical data for AAPL (need 20+ bars, got 0)",
        "News API unavailable",
        "Flow engine: Level 2 data subscription required"
    ]
}
```

### Current Performance

**BEFORE FIX:**
```
Feature Count: 45 total
Available: 5-12 features (11-27%)
Unavailable: 33-40 features (73-89%)
```

**TARGET AFTER FIX:**
```
Feature Count: 40 total (excluding flow engine)
Available: 30-35 features (75-88%)
Unavailable: 5-10 features (12-25%)
```

### Log Line Location

**File:** `wolf_app.py:5883`

```python
LOGGER.info(
    f"[{symbol}] Extracted {feature_data['available_count']}/{feature_data['feature_count']} features "
    f"in {feature_data['execution_time_ms']:.0f}ms"
)
```

**Example Output:**
```
[AAPL] Extracted 5/45 features in 66ms
```

---

## CONFIDENCE & DIRECTION LOGIC

**File:** `wolf_app.py:5930-6020`  
**Function:** Inline logic within `api_predict_run()`

### Confidence Calculation Formula

```python
# Starting point
base_confidence = 0.45  # 45% baseline

# Signal strength tracker
signal_strength = 0

# RSI signals
if RSI_14 > 70:
    direction = "DOWN"
    base_confidence += 0.08
    signal_strength += 1
elif RSI_14 < 30:
    direction = "UP"
    base_confidence += 0.08
    signal_strength += 1

# MACD signals
if MACD_HISTOGRAM > 0:
    direction = "UP"
    base_confidence += 0.06
    signal_strength += 1
elif MACD_HISTOGRAM < 0:
    direction = "DOWN"
    base_confidence += 0.06
    signal_strength += 1

# Bollinger position
if BOLLINGER_POSITION > 0.9 and direction == "DOWN":
    base_confidence += 0.05
    signal_strength += 1
elif BOLLINGER_POSITION < 0.1 and direction == "UP":
    base_confidence += 0.05
    signal_strength += 1

# Volume confirmation
if VOLUME_SPIKE > 1.5:
    base_confidence += 0.05
    signal_strength += 1

# Sentiment alignment
if NEWS_SENTIMENT_SCORE > 0.3 and direction == "UP":
    base_confidence += 0.07
    signal_strength += 1
elif NEWS_SENTIMENT_SCORE < -0.3 and direction == "DOWN":
    base_confidence += 0.07
    signal_strength += 1

# Signal convergence bonus
if signal_strength >= 4:
    base_confidence += 0.05
elif signal_strength >= 3:
    base_confidence += 0.03
elif signal_strength <= 1:
    base_confidence -= 0.05  # PENALTY

# Apply bounds
confidence = max(0.40, min(0.85, base_confidence))
```

### Direction Logic

```
IF no strong signals:
    direction = "FLAT"

IF RSI < 30 OR MACD_HISTOGRAM > 0 OR positive momentum:
    direction = "UP"

IF RSI > 70 OR MACD_HISTOGRAM < 0 OR negative momentum:
    direction = "DOWN"
```

### Current Behavior (PROBLEM)

**With 5/45 features available:**
- RSI: None → No signal
- MACD_HISTOGRAM: None → No signal
- BOLLINGER_POSITION: None → No signal
- VOLUME_SPIKE: None → No signal
- NEWS_SENTIMENT_SCORE: None → No signal

**Result:**
- `signal_strength = 0`
- Penalty: `base_confidence -= 0.05`
- Final: `0.45 - 0.05 = 0.40` (minimum)
- Direction: `"FLAT"` (no signals to change it)

**ALL PREDICTIONS:**
```json
{
    "confidence": 0.4,
    "direction": "FLAT"
}
```

---

## ROOT CAUSE ANALYSIS

### Why Only 5/45 Features Extract?

**Pillar 1 (Price):** ✅ Works (2/8 signals)
- `PRICE` ✅
- `PREV_CLOSE` ✅ (sometimes)
- Others: ❌ Not implemented

**Pillar 2 (Technical):** ❌ BROKEN (0-5/15 signals)
- **Root cause:** `yfinance` returns insufficient historical data
- `hist = ticker.history(start=start_date, end=end_date)`
- Often returns 0-10 bars instead of 90+ needed
- Indicators require 20-200 bars minimum
- No fallback providers
- No error recovery

**Pillar 3 (Volume):** ❌ BROKEN (0-2/5 signals)
- **Root cause:** Same yfinance data dependency
- Shares failure mode with Technical Engine

**Pillar 4 (Sentiment):** ⚠️ DEGRADED (0-2/3 signals)
- **Root cause:** News API rate limits / missing keys
- Polygon News API requires paid subscription
- Alpha Vantage has strict rate limits (5 calls/min)

**Pillar 5 (World Context):** ✅ WORKS (2-4/4 signals)
- No issues - fetches SPY/VIX reliably

**Pillar 6 (Flow):** ❌ NOT IMPLEMENTED (0/4 signals)
- Intentionally disabled (requires Level 2 data)

### Critical Failures

1. **yfinance Historical Data Fetch**
   - File: `technical_engine.py:101`, `volume_engine.py:69`
   - Issue: Returns empty/insufficient data
   - Impact: 20/45 features unavailable

2. **No Fallback Providers**
   - Technical engine has no backup if yfinance fails
   - Should use Polygon historical API
   - Should use Alpha Vantage daily data

3. **Missing BOLLINGER_POSITION**
   - Confidence logic references this feature
   - Technical engine calculates BB bands but not position
   - File: `technical_engine.py:215` (calculates bands)
   - Missing: `(current_price - BB_LOWER) / (BB_UPPER - BB_LOWER)`

4. **No Error Recovery**
   - Single indicator failure kills entire pillar
   - No try/except around individual calculations
   - One missing feature → entire engine returns 0 signals

5. **Crypto Symbol Incompatibility**
   - yfinance doesn't recognize BTC, ETH, SOL
   - Must use `-USD` suffix (BTC-USD)
   - Or use separate crypto data source

---

## FIX PRIORITY

### P0 - CRITICAL (Must fix to restore predictions)

1. **Fix yfinance Historical Data Fetch**
   - Add fallback to Polygon historical bars API
   - Add crypto symbol handling (BTC → BTC-USD)
   - Increase error handling/logging

2. **Add BOLLINGER_POSITION Calculation**
   - File: `technical_engine.py:215`
   - Calculate: `(current_price - BB_LOWER) / (BB_UPPER - BB_LOWER)`
   - Add as signal: `BB_POSITION`

3. **Add Error Recovery to Technical Engine**
   - Wrap each indicator in try/except
   - Continue if one indicator fails
   - Log specific failures

### P1 - HIGH (Improves quality significantly)

4. **Share OHLCV Data Between Pillars**
   - Technical and Volume engines duplicate yfinance calls
   - Cache historical data for 5-10 minutes
   - Performance + reliability improvement

5. **Add Polygon Historical Fallback**
   - When yfinance fails, try Polygon /v2/aggs
   - Requires POLYGON_API_KEY env var
   - Better data quality

6. **Fix Crypto Symbol Support**
   - Detect crypto symbols
   - Use CoinGecko historical API for crypto
   - Or use Binance OHLCV endpoint

### P2 - MEDIUM (Nice to have)

7. **Optimize Feature Calculation**
   - Don't calculate unused indicators
   - Cache expensive calculations
   - Parallel pillar execution

8. **Add Confidence-Based Feature Weighting**
   - High-confidence features (1.0) weighted more
   - Low-confidence features (0.5-0.7) weighted less
   - Unavailable features (0.0) ignored

---

## SUCCESS METRICS

**Current State:**
- Feature availability: 11-27% (5-12/45)
- Confidence: Always 40% (stuck at minimum)
- Direction: Always FLAT (no signals)
- Prediction quality: Poor (no trading value)

**Target State:**
- Feature availability: 75-88% (30-35/40)
- Confidence: 40-75% range (varies by signal strength)
- Direction: UP/DOWN/FLAT mix (based on real signals)
- Prediction quality: Tradeable (directional bias + confidence variation)

**Test Symbols:**
- SPY (S&P 500) - Should hit 85%+ features
- AAPL (Large cap stock) - Should hit 80%+ features
- BTC (Crypto) - Should hit 70%+ features (no flow data)
- WOLF (Small cap) - Should hit 65%+ features (limited news)

---

## NEXT STEPS

1. ✅ Create pipeline map (this document)
2. ⏳ Add diagnostic endpoint `/api/dev/features/diagnostic?symbol=MSFT`
3. ⏳ Fix Technical Engine yfinance fallback
4. ⏳ Add BOLLINGER_POSITION calculation
5. ⏳ Add error recovery to all pillars
6. ⏳ Test locally with MSFT, AAPL, BTC, SOL
7. ⏳ Deploy to Railway
8. ⏳ Verify production feature extraction
9. ⏳ Update documentation

---

**Document Version:** 1.0  
**Last Updated:** November 24, 2025  
**Next Review:** After pillar fixes deployed
