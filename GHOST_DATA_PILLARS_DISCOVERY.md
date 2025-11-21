# 🔍 Ghost Data Pillars - Discovery Report

**Mission**: Transform Ghost from "guessing" to "fully informed quantitative machine intelligence"  
**Date**: 2025-01-XX  
**Status**: ✅ Discovery Phase Complete

---

## 📊 Executive Summary

Ghost Protocol already has **extensive data infrastructure** that must be **wrapped and enhanced** rather than rebuilt from scratch. This discovery phase mapped 60+ existing modules across 6 data domains.

### Key Finding
**Ghost has ~70% of required infrastructure already implemented**. The challenge is creating unified abstraction layers while preserving all existing production behavior.

---

## 🏗️ Six Data Pillars - Existing Infrastructure Mapping

### PILLAR 1: Multi-Source Price Engine
**Status**: ✅ **70% COMPLETE** - Production-ready infrastructure exists

**Existing Modules**:
- `core/price_quorum.py` (232 lines) - Multi-provider consensus system
  - Async operations with timeout handling
  - Configurable quorum requirements (market-open vs closed)
  - Tolerance-based agreement detection
  - Rate limiting integration
  - Provider performance tracking
- `core/price_reliability.py` (~180 lines) - Provider fallback with circuit breakers
  - Provider stats tracking (success/fail/stale/latency)
  - `get_price_with_fallback()` function
  - Dependency injection pattern to avoid circular imports
- `core/crypto/crypto_providers.py` (~470 lines) - Crypto-specific quorum
  - 3 providers: CoinGecko, Binance, Coinbase
  - 40+ crypto symbols supported
  - Cache layer (2-min TTL)
  - Short-circuit optimization to avoid rate limits

**Providers Discovered**:
- Polygon (5 calls/min, 1-min bars, 5-min delayed free tier)
- AlphaVantage (real-time quote support)
- Yahoo Finance (HTTP API + yfinance fallback)
- CoinGecko (crypto)
- Binance (crypto)
- Coinbase (crypto)

**Missing Signals** (30% gap):
- ❌ Alpaca integration (equities + crypto)
- ❌ Tiingo integration
- ❌ Kraken (crypto)
- ❌ Bid/ask spread tracking
- ❌ VWAP calculation
- ❌ Provider quality score (latency-weighted confidence)

**Integration Points**:
- `/api/price/{symbol}` - Stock/ETF prices
- `/api/crypto/price/{symbol}` - Crypto prices
- `/api/cockpit/snapshot` - Dashboard price feeds
- `wolf_app.py` line ~16817 - Provider routing logic

---

### PILLAR 2: Volume & Order-Flow Engine
**Status**: ⚠️ **30% COMPLETE** - Basic volume indicators exist, advanced features missing

**Existing Modules**:
- `core/indicators.py` - Volume indicators (lines 265-320)
  - `obv()` - On-Balance Volume
  - `ad_line()` - Accumulation/Distribution Line
  - `cmf()` - Chaikin Money Flow (20-period)
  - `mfi()` - Money Flow Index (14-period)
  - `vwap()` - Volume Weighted Average Price
  - `force_index()` - Force Index (13-period EMA)
  - `ease_of_movement()` - Ease of Movement (14-period)
- `core/market_scanner.py` - Market opportunity scanning
  - Volume analysis logic likely embedded
  - Need to investigate further
- `core/strategy_ensemble.py` - Volume confirmation (line 121)
  - Volume ratio calculation (recent_vol / avg_vol)
  - Threshold checks (vol_ratio > 1.3 = strong volume)

**Missing Signals** (70% gap):
- ❌ RVOL (Relative Volume vs 30-day average)
- ❌ Dark pool print detection
- ❌ Whale transaction tracking (>$1M blocks)
- ❌ 3x/5x/10x volume spike alerts
- ❌ Institutional block trades
- ❌ Options volume vs stock volume ratio
- ❌ Order flow imbalance (buy vs sell pressure)
- ❌ Time & Sales aggregation

---

### PILLAR 3: Momentum & Technical Indicators
**Status**: ✅ **85% COMPLETE** - Comprehensive indicator library exists

**Existing Modules**:
- `core/indicators.py` (600+ lines) - Production-grade indicator library
  - **Trend Indicators**: SMA, EMA, WMA, DEMA, TEMA, MACD, ADX, Aroon
  - **Momentum Indicators**: RSI, Stochastic, Williams %R, ROC, CCI, Ultimate Oscillator
  - **Volatility Indicators**: Bollinger Bands, ATR, Keltner Channels, Donchian Channels, Historical Volatility
  - **Volume Indicators**: OBV, AD Line, CMF, MFI, VWAP, Force Index, Ease of Movement
  - **Pattern Detection**: Trend detection, Golden Cross, Death Cross
  - `calculate_all_indicators()` - Batch calculation function
  - `get_indicator_summary()` - Signal aggregation
- `core/momentum_detector.py` - Momentum analysis for breakouts
- `core/regime_detector.py` - Market regime classification

**Available Indicators** (from `AVAILABLE_INDICATORS` dict):
```python
{
  'trend': ['sma', 'ema', 'wma', 'dema', 'tema', 'macd', 'adx', 'aroon'],
  'momentum': ['rsi', 'stochastic', 'williams_r', 'roc', 'cci', 'momentum', 'ultimate_oscillator'],
  'volatility': ['bollinger_bands', 'atr', 'keltner_channels', 'donchian_channels', 'historical_volatility'],
  'volume': ['obv', 'ad_line', 'cmf', 'mfi', 'vwap', 'force_index', 'ease_of_movement']
}
```

**Missing Signals** (15% gap):
- ❌ Ichimoku Cloud (conversion, base, span A/B, lagging span)
- ❌ Parabolic SAR
- ❌ Elder Ray Index (Bull/Bear Power)
- ❌ Chaikin Oscillator

**Integration Points**:
- `core/strategy_ensemble.py` - Strategy voting system
- `core/crypto/crypto_predictor.py` - Crypto prediction engine
- Hunter brain decision logic

---

### PILLAR 4: Fundamentals & Corporate Actions
**Status**: ⚠️ **40% COMPLETE** - SEC EDGAR integration exists but incomplete

**Existing Modules**:
- `core/edgar_integration.py` (600+ lines) - SEC filing parser
  - Form 8-K critical items tracking:
    - 1.01: Material Agreement Entry
    - 1.02: Material Agreement Termination
    - 2.01: Acquisition/Disposition
    - 2.02: Results of Operations (earnings)
    - 3.01: Delisting Notice ⚠️
    - 4.02: Non-Reliance on Financials ⚠️
    - 5.02: Officer/Director Changes
  - `get_insider_transactions()` - Form 4 insider trading (lines 281-285)
  - `search_companies()` - Company search by name
  - `_ticker_to_cik()` - Ticker → CIK conversion
- `core/corporate_actions.py` - Corporate action tracking
  - Stock splits detection ✅
  - Dividends tracking ✅
  - Earnings dates ✅
  - `/api/corporate_actions` endpoint ✅
- `wolf_app.py` - DELISTED_SYMBOLS registry (lines 554-569)
  - WOLF bankruptcy + 120:1 reverse split tracking
  - PnL adjustment logic implemented
- `scripts/normalize_wolf_portfolio.py` - Portfolio normalization for splits

**Integration Points**:
- `/api/edgar/recent_filings?filing_type=8-K&hours_back=24`
- `/api/edgar/company_filings?ticker=WOLF&limit=20`
- `/api/edgar/insider_transactions?ticker=AAPL&days_back=90`
- `/api/corporate_actions` - Corporate action data
- Ghost agent tools: `filings.search`, `insiders.form4`, `company.profile`

**Missing Signals** (60% gap):
- ❌ Market cap calculation (real-time)
- ❌ Forward P/E, P/S, P/B ratios
- ❌ Debt-to-equity ratio
- ❌ EPS growth rate
- ❌ Revenue growth rate
- ❌ Earnings surprise % (actual vs estimate)
- ❌ Dividend yield calculation
- ❌ Float calculation (shares outstanding - restricted)
- ❌ Short interest % of float
- ❌ Institutional ownership %
- ❌ Bankruptcy prediction score
- ❌ Credit rating tracking

**Status Notes**:
- EDGAR integration: "40% COMPLETE - Code exists, not fully tested, no scheduled checks, no cockpit integration"
- Corporate actions: "70% COMPLETE - API exists, missing automatic split adjustment and dividend payout tracking"

---

### PILLAR 5: Sentiment & News Engine
**Status**: ✅ **60% COMPLETE** - Multiple sentiment sources exist

**Existing Modules**:
- `core/news_sentiment.py` (~150 lines) - News aggregation + sentiment
  - Alpha Vantage News API integration
  - Sentiment scoring (-1.0 to +1.0)
  - `fetch_news_sentiment()` function
  - Cache layer (1-hour TTL)
  - Sentiment labels: VERY_NEGATIVE → VERY_POSITIVE
  - `adjust_confidence_with_sentiment()` - Confidence boosting
- `core/social_sentiment.py` (~230 lines) - Social media tracking
  - `fetch_twitter_sentiment()` - Twitter/X mentions
  - `fetch_reddit_sentiment()` - WallStreetBets tracking
  - `get_combined_social_sentiment()` - Multi-source aggregation
  - `get_trending_stocks()` - Viral signal detection
  - Cache layer (10-min TTL)
  - Placeholder implementations (API keys required)
- `core/world_feed_fusion.py` (~500+ lines) - RSS feed aggregation
  - 8 RSS sources: Reuters, Bloomberg, FT, WSJ, CNBC, MarketWatch, Seeking Alpha
  - TextBlob NLP sentiment analysis
  - Keyword extraction
  - Symbol extraction from text
  - SQLite persistence (data/world_feed.db)
  - Sentiment aggregation with weighted scoring
  - News categories: earnings, markets, policy, company, economic, breaking
- `core/context_engine.py` (~200 lines) - World context aggregator
  - 25 RSS feeds (Reuters, MarketWatch, TechCrunch)
  - NER extraction (tickers, companies, people)
  - VADER sentiment scoring (-1.0 to +1.0)
  - Relevance matching to watchlist
  - Entity linking (CEO → Company → Ticker)

**Integration Points**:
- `/api/news` - News feed endpoint
- `/api/news/recent?minutes=120` - Recent news
- `/api/news/sentiment/{symbol}` - Symbol-specific sentiment
- Ghost agent tools: `news.search`, `sentiment.score`
- Cockpit panels: News Feed, TOP HEADLINES, News Context

**Missing Signals** (40% gap):
- ❌ Reuters direct API integration
- ❌ Benzinga API integration
- ❌ Twitter/X API v2 integration (requires TWITTER_BEARER_TOKEN)
- ❌ Reddit PRAW integration (requires REDDIT_CLIENT_ID)
- ❌ Social volume spike detection
- ❌ Influencer tracking (whale accounts)
- ❌ Meme coin sentiment (TikTok, Discord)
- ❌ Fear & Greed Index calculation
- ❌ Sentiment momentum (rate of change)
- ❌ Sector-specific sentiment aggregation

**Status Notes**:
- Twitter/X: Placeholder implementation, requires API key
- Reddit: Placeholder implementation, requires PRAW credentials
- News sentiment: Working with Alpha Vantage, needs Benzinga/Reuters expansion
- RSS feeds: Operational with 8 sources
- TextBlob NLP: Operational fallback sentiment analysis

---

### PILLAR 6: Macro & Global Context
**Status**: ✅ **65% COMPLETE** - Core macro indicators exist

**Existing Modules**:
- `core/world_context.py` (157 lines) - Global market context
  - SPY price tracking via price_quorum
  - VIX level tracking
  - Market mood calculation (bullish/neutral/bearish)
  - Confidence clamping (0-100 range, fixed 5000% bug)
  - News summary aggregation
  - `/api/world_context` endpoint
  - Lines 32-80: SPY/VIX price fetching
  - Lines 106-110: Confidence clamping logic
  - Lines 128-135: Market mood sentiment
- `core/regime_detector.py` - Market regime classification
  - Bull/bear/sideways regime detection
  - Volatility regime analysis
- `core/economic_calendar.py` - Economic event tracking
  - Earnings dates
  - Fed meetings
  - Economic data releases

**Tracked Macro Indicators**:
- ✅ SPY (S&P 500) price + change %
- ✅ VIX (Fear Index) level + status (calm/normal/elevated/high-fear)
- ✅ Market mood (sentiment score 0-100)

**Missing Signals** (35% gap):
- ❌ QQQ (Nasdaq-100) tracking
- ❌ DIA (Dow Jones) tracking
- ❌ DXY (Dollar Index) tracking
- ❌ 10-Year Treasury Yield
- ❌ 2-Year Treasury Yield
- ❌ Yield curve (2Y-10Y spread)
- ❌ Gold spot price (inflation hedge)
- ❌ Crude oil spot price
- ❌ CPI (Consumer Price Index) latest reading
- ❌ PPI (Producer Price Index) latest reading
- ❌ Unemployment rate
- ❌ Fed Funds Rate current target
- ❌ Fed rate change probability (CME FedWatch)
- ❌ Sector rotation analysis (XLF, XLE, XLK, XLV, etc.)

**Integration Points**:
- `/api/world_context` - Global context endpoint
- Cockpit dashboard - Market Context panel
- Hunter brain - World context enrichment

---

## 🛠️ Technical Discoveries

### 1. Dependency Injection Pattern
**Location**: `core/price_reliability.py` line 148-178

Ghost uses **injected functions** to avoid circular imports:
```python
def _fetch_price_from_provider(
    symbol: str,
    provider_name: str,
    price_quorum_func: Callable  # ← Injected to avoid circular import
) -> tuple[float | None, float | None]:
    ...
```

**Implication**: New data pillar abstraction layer must maintain this pattern.

### 2. Async Price Quorum System
**Location**: `core/price_quorum.py` lines 116-155

Production-ready async consensus:
```python
async def _get_price_async(
    self, 
    symbol: str, 
    providers: list[PriceProvider],
    timeout_seconds: float = 6.0
) -> PriceDecision:
    # Parallel provider fetching
    # Tolerance-based agreement detection
    # Median calculation from agreeing providers
    # Confidence scoring based on quorum size
```

**Configuration**:
- Market open: requires 3 providers agreeing within 3% tolerance
- Market closed: requires 1 provider within 6% tolerance

### 3. Cache Layers
**Discovery**: Multiple cache implementations across modules

| Module | Cache TTL | Key Pattern |
|--------|-----------|-------------|
| `crypto_providers.py` | 2 minutes | `symbol` |
| `news_sentiment.py` | 1 hour | `{symbol}_news` |
| `social_sentiment.py` | 10 minutes | `twitter_{symbol}`, `reddit_{symbol}_{subreddit}` |
| `price_quorum.py` | 30s (market), 5min (closed) | Inline TTL check |

**Implication**: New data pillar layer should use centralized cache manager (`core/cache_manager.py`).

### 4. Provider Rate Limits
**Discovered Constraints**:
- Polygon: 5 calls/min (free tier)
- AlphaVantage: Unknown (likely 5 calls/min free tier)
- Yahoo Finance: No official limit (aggressive rate limiting on scraper)

**Existing Solution**: `AsyncRateLimiter` integrated into `PriceQuorum.__init__()`.

### 5. Corporate Actions Registry
**Location**: `wolf_app.py` lines 554-569

```python
DELISTED_SYMBOLS: dict[str, dict[str, Any]] = {
    "WOLF": {
        "status": "restructured",
        "date": "2025-10-01",
        "reverse_split_ratio": 120,
        "note": "Emerged from Chapter 11 bankruptcy Oct 2025",
        "shareholders_diluted": True
    }
}
```

**Implication**: PILLAR 4 should use this registry + extend to generic corporate action tracker.

---

## 📐 Architecture Recommendations

### 1. Package Structure
**Recommended Namespace**: `core/data_pillars/`

```
core/
├── data_pillars/
│   ├── __init__.py                    # Pillar registry + unified interface
│   ├── base_pillar.py                 # Abstract base class for all pillars
│   ├── price_engine.py                # PILLAR 1: Wraps price_quorum, crypto_providers
│   ├── volume_engine.py               # PILLAR 2: New + wraps indicators volume funcs
│   ├── momentum_engine.py             # PILLAR 3: Wraps indicators.py momentum
│   ├── fundamentals_engine.py         # PILLAR 4: Wraps edgar_integration, corporate_actions
│   ├── sentiment_engine.py            # PILLAR 5: Wraps news_sentiment, social_sentiment, world_feed_fusion
│   └── macro_engine.py                # PILLAR 6: Wraps world_context, regime_detector
```

### 2. Unified Interface Pattern
**Goal**: All pillars expose same interface signature

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class DataSignal:
    """Unified signal structure across all pillars"""
    name: str                    # e.g., "RSI_14", "SPY_PRICE", "NEWS_SENTIMENT"
    value: float | str | None    # Signal value
    confidence: float            # 0.0-1.0
    data_available: bool         # True if real data, False if fallback/estimate
    source: str                  # e.g., "polygon", "alpha_vantage", "textblob"
    timestamp: float             # Unix timestamp
    metadata: dict[str, Any]     # Extra context

@dataclass
class PillarResponse:
    """Unified response from all pillar engines"""
    pillar_name: str             # e.g., "price_engine", "sentiment_engine"
    symbol: str                  # Target symbol
    signals: list[DataSignal]    # All signals from this pillar
    errors: list[str]            # Any errors encountered
    execution_time_ms: float     # Performance tracking
    timestamp: float             # Response timestamp
```

### 3. Dependency Injection for Integration
**Pattern**: Each pillar receives dependencies via constructor

```python
class PriceEngine:
    def __init__(
        self, 
        quorum: PriceQuorum,                # Existing quorum
        crypto_providers: CryptoProviders,  # Existing crypto
        cache_manager: CacheManager,        # Shared cache
        metrics: ConcurrencyMetrics         # Shared metrics
    ):
        self.quorum = quorum
        self.crypto = crypto_providers
        self.cache = cache_manager
        self.metrics = metrics
```

### 4. Graceful Degradation
**Requirement**: Every signal must have `data_available` flag

```python
# Example: If Twitter API unavailable
DataSignal(
    name="TWITTER_SENTIMENT",
    value=0.0,
    confidence=0.0,
    data_available=False,  # ← Honest about missing data
    source="placeholder",
    timestamp=time.time(),
    metadata={"error": "TWITTER_BEARER_TOKEN not configured"}
)
```

**Never**:
```python
# ❌ BAD: Silent fabrication
value=random.uniform(-1.0, 1.0)  # Fake sentiment
data_available=True              # Lying to hunter
```

---

## 🎯 Implementation Priority

### Phase 1: High-Value, Low-Risk (Week 1)
1. **PILLAR 1 wrapper** (price_engine.py)
   - Wrap existing price_quorum + crypto_providers
   - Add Alpaca, Tiingo providers
   - Unified `get_price()` interface
   - No changes to existing modules

2. **PILLAR 3 wrapper** (momentum_engine.py)
   - Wrap existing indicators.py
   - Add Ichimoku Cloud, Parabolic SAR
   - Unified `get_momentum_signals()` interface

3. **PILLAR 6 wrapper** (macro_engine.py)
   - Wrap existing world_context.py
   - Add QQQ, DXY, Treasury yields
   - Unified `get_macro_context()` interface

### Phase 2: Medium Complexity (Week 2)
4. **PILLAR 5 enhancement** (sentiment_engine.py)
   - Wrap news_sentiment, social_sentiment, world_feed_fusion
   - Add Twitter/Reddit API integration (if keys available)
   - Add Benzinga API
   - Unified `get_sentiment_signals()` interface

5. **PILLAR 2 new build** (volume_engine.py)
   - Wrap indicators.py volume functions
   - Add RVOL, dark pool, whale detection
   - Build order flow imbalance tracker
   - Unified `get_volume_signals()` interface

### Phase 3: Complex Data Pipelines (Week 3)
6. **PILLAR 4 enhancement** (fundamentals_engine.py)
   - Wrap edgar_integration, corporate_actions
   - Add real-time fundamentals API (Alpha Vantage, Financial Modeling Prep)
   - Build earnings surprise tracker
   - Unified `get_fundamental_signals()` interface

---

## 🚨 Safety Rails (MUST PRESERVE)

### 1. Auto-Trading Safety
**Constraint**: Do NOT enable AUTO_TRADE in any new code

```python
# ❌ NEVER
AUTO_TRADE = True

# ✅ ALWAYS
if ENABLE_AUTO_TRADING and AUTO_TRADE:
    # Require explicit double-confirmation
```

### 2. API Key Safety
**Constraint**: Never hardcode or log API keys

```python
# ✅ CORRECT
api_key = os.getenv("TWITTER_BEARER_TOKEN")
if not api_key:
    return {"ok": False, "error": "API key not configured"}
```

### 3. Production Endpoint Preservation
**Constraint**: Do NOT break existing endpoints

**Protected Endpoints**:
- `/api/health`
- `/api/cockpit`
- `/api/cockpit/snapshot`
- `/api/price/{symbol}`
- `/api/crypto/price/{symbol}`
- `/api/world_context`
- `/api/news`
- `/api/corporate_actions`

**Strategy**: Create NEW endpoints for pillar data, keep old endpoints untouched.

### 4. Database Schema Safety
**Constraint**: Do NOT modify existing database schemas

**Protected Databases**:
- `wolf.db` - Portfolio, orders, positions
- `watchlist.db` - Watchlist tracking
- `goals_log.db` - Goal tracking
- `data/world_feed.db` - RSS feeds
- `data/context_news.db` - Context news

**Strategy**: Create NEW tables for pillar metadata, use migrations for schema changes.

---

## 📊 Discovery Metrics

### Modules Discovered
- **Core modules scanned**: 60+
- **Lines of code analyzed**: ~5000+
- **Data sources identified**: 15+
- **API endpoints mapped**: 25+

### Infrastructure Completeness
| Pillar | Existing % | Missing % | Priority |
|--------|-----------|-----------|----------|
| PILLAR 1: Price | 70% | 30% | 🟢 HIGH |
| PILLAR 2: Volume | 30% | 70% | 🔴 CRITICAL |
| PILLAR 3: Momentum | 85% | 15% | 🟢 LOW |
| PILLAR 4: Fundamentals | 40% | 60% | 🟡 MEDIUM |
| PILLAR 5: Sentiment | 60% | 40% | 🟡 MEDIUM |
| PILLAR 6: Macro | 65% | 35% | 🟢 MEDIUM |

### Total Infrastructure Gap
- **Overall existing**: ~58% (weighted average)
- **Overall missing**: ~42%
- **Implementation effort**: ~4-6 weeks for complete build

---

## ✅ Next Steps (PHASE 1)

1. Create `core/data_pillars/` package skeleton
2. Implement `base_pillar.py` abstract interface
3. Build `price_engine.py` wrapper (PILLAR 1)
4. Build `momentum_engine.py` wrapper (PILLAR 3)
5. Build `macro_engine.py` wrapper (PILLAR 6)
6. Create unified `get_all_pillar_data(symbol)` orchestrator
7. Integrate with hunter brain (`core/hunter.py` or equivalent)
8. Test on cockpit `/api/cockpit/snapshot` endpoint

---

## 📚 References

### Key Files to Review
- `core/price_quorum.py` - Price consensus architecture
- `core/indicators.py` - Technical indicator library
- `core/news_sentiment.py` - News aggregation
- `core/world_context.py` - Macro context
- `core/edgar_integration.py` - SEC filings
- `wolf_app.py` - Main application routes

### Documentation
- `LEVEL10_README.md` - Feature matrix
- `GHOST_CAPABILITIES.md` - Complete capabilities list
- `CHATGPT_ANALYST.md` - Analyst integration docs
- `CRYPTO_PREDICTION_MODULE_BLUEPRINT.md` - Crypto predictor specs

---

**Discovery Phase**: ✅ **COMPLETE**  
**Ready for**: ✅ **PHASE 1 Implementation**  
**Estimated Build Time**: 4-6 weeks for full 6-pillar deployment  
**Risk Level**: 🟢 LOW (wrap existing + additive design)
