# GHOST PROVIDER ARCHITECTURE - AFTER (HARDENED STACK)

**Date**: November 25, 2025
**Architect**: Ghost Provider Surgeon

---

## DESIGN PRINCIPLES

1. **Paid-First**: Premium providers (Polygon paid tier, Binance) as primary
2. **Graceful Degradation**: Never fail completely, always fallback
3. **Cache-Heavy**: 80% API call reduction via Redis
4. **Provider Isolation**: One provider failure doesn't kill entire pillar
5. **Observable**: Full diagnostics, latency tracking, health endpoints

---

## NEW PROVIDER STACK

### Tier 1: STOCK/ETF/INDEX (Paid Polygon + Free Fallbacks)

**Spot Price Priority**:

```text

1. Polygon (PAID tier) - unlimited calls, 1ms latency
2. Yahoo Finance - free, rate-limited fallback
3. yfinance - free, library fallback
4. Redis Cache - last known good price (15min TTL)


```text

**Historical OHLCV Priority**:

```text

1. Polygon (PAID tier) - 1min/5min/1hour bars, unlimited
2. yfinance - free, rate-limited (backup)
3. Redis Cache - cached OHLCV (5min TTL)


```text

**Configuration**:

```bash

# Required

POLYGON_API_KEY=<paid-tier-key>  # $49/month unlimited
REDIS_URL=<upstash-redis-url>

# Optional

ALPHAVANTAGE_API_KEY=<key>  # Emergency fallback

```text

### Tier 2: CRYPTO (Binance + CoinGecko)

**Spot Price Priority**:

```text

1. Binance Public API - unlimited, 50ms latency
2. CoinGecko - free 50/min, good coverage
3. Coinbase - free unlimited, limited coins
4. Redis Cache - last known good price (30s TTL)


```text

**Historical OHLCV Priority**(NEW!):

```text

1. Binance Klines API - FREE unlimited, 1min/5min/1hour/1day bars
2. CoinGecko Market Chart - paid tier only (skip for now)
3. Redis Cache - cached OHLCV (10min TTL)


```text**Symbol Mapping**:

```python

GHOST_TO_BINANCE = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT",
    "AVAX": "AVAXUSDT",
    "DOT": "DOTUSDT",
    "MATIC": "MATICUSDT",

    # Add as needed

}

```text

**Configuration**:

```bash

# Optional (public API works without key)

BINANCE_API_KEY=<key>  # For higher rate limits if needed

```text

---

## CACHING STRATEGY (80% REDUCTION)

### Redis Key Schema

**Spot Prices**:

```text

ghost:price:spot:{SYMBOL}:v1
TTL: 90 seconds (stocks), 30 seconds (crypto)

Value: {"price": float, "prev_close": float, "provider": str, "ts": int}

```text

**OHLCV Bars**:

```text

ghost:ohlcv:{SYMBOL}:{INTERVAL}:v1
TTL: 5 minutes (1min/5min bars), 60 minutes (1hour/1day bars)

Value: [
  {"t": timestamp, "o": open, "h": high, "l": low, "c": close, "v": volume},
  ...
]

```text

**Technical Indicators**(Pre-computed):

```text

ghost:indicators:{SYMBOL}:v1
TTL: 60 seconds

Value: {
  "RSI_14": float,
  "MACD_HISTOGRAM": float,
  "BB_UPPER": float,
  "BB_LOWER": float,
  "BOLLINGER_POSITION": float,
  ...
}

```text**Provider Health**:

```text

ghost:provider:health:{PROVIDER}:v1
TTL: 30 seconds

Value: {
  "success_rate": float,  # 0.0-1.0
  "p95_latency_ms": float,
  "last_error": str,
  "last_ok_ts": int
}

```text

### Cache Hit Rate Targets

| Data Type | Target Hit Rate | TTL | Impact |
|-----------|-----------------|-----|--------|
| Spot Price | 70-80% | 90s stock, 30s crypto | -75% price API calls |
| OHLCV | 85-90% | 5min | -85% OHLCV API calls |
| Indicators | 80-85% | 60s | -80% calculation overhead |
| **Overall**|**80%+**| - |**-80% API load** |

---

## NEW MODULE STRUCTURE

### core/providers/unified_provider.py (NEW)

```python

"""
Unified Provider Interface
==========================
Single entry point for all price/OHLCV data with caching + fallbacks.
"""

from dataclasses import dataclass
from typing import Optional, List
import time

@dataclass
class SpotPrice:
    symbol: str
    price: float
    prev_close: Optional[float]
    provider: str
    timestamp: int
    cache_hit: bool = False

@dataclass
class OHLCVBar:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class OHLCVData:
    symbol: str
    interval: str  # "1m", "5m", "1h", "1d"
    bars: List[OHLCVBar]
    provider: str
    cache_hit: bool = False


class UnifiedProvider:
    """
    Hardened provider with:

    - Redis caching
    - Multi-provider fallbacks
    - Rate limit protection
    - Health tracking


    """

    def __init__(self, redis_client):
        self.redis = redis_client
        self.stock_providers = StockProviderChain(redis_client)
        self.crypto_providers = CryptoProviderChain(redis_client)

    def get_spot_price(self, symbol: str) -> Optional[SpotPrice]:
        """Get spot price with caching + fallbacks"""

        # 1. Check cache

        cached = self._get_cached_price(symbol)
        if cached:
            return cached

        # 2. Determine asset type

        is_crypto = self._is_crypto(symbol)

        # 3. Fetch from providers

        if is_crypto:
            price = self.crypto_providers.get_price(symbol)
        else:
            price = self.stock_providers.get_price(symbol)

        # 4. Cache result

        if price:
            self._cache_price(price)

        return price

    def get_ohlcv(self, symbol: str, interval: str, lookback: int) -> Optional[OHLCVData]:
        """Get OHLCV bars with caching + fallbacks"""

        # 1. Check cache

        cached = self._get_cached_ohlcv(symbol, interval)
        if cached and len(cached.bars) >= lookback:
            return cached

        # 2. Determine asset type

        is_crypto = self._is_crypto(symbol)

        # 3. Fetch from providers

        if is_crypto:
            ohlcv = self.crypto_providers.get_ohlcv(symbol, interval, lookback)
        else:
            ohlcv = self.stock_providers.get_ohlcv(symbol, interval, lookback)

        # 4. Cache result

        if ohlcv:
            self._cache_ohlcv(ohlcv)

        return ohlcv

```text

### core/providers/stock_provider_chain.py (REFACTOR)

```python

"""
Stock Provider Chain
====================
Polygon → Yahoo → yfinance → Cache fallback
"""

class StockProviderChain:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.polygon = PolygonProvider(os.getenv("POLYGON_API_KEY"))
        self.yahoo = YahooProvider()
        self.yfinance = YFinanceProvider()

    def get_price(self, symbol: str) -> Optional[SpotPrice]:
        """Try providers in order with 3x retry on Polygon"""
        providers = [
            ("polygon", self.polygon),
            ("yahoo", self.yahoo),
            ("yfinance", self.yfinance),
        ]

        for name, provider in providers:
            try:

                # Polygon gets 3 retries with backoff

                retries = 3 if name == "polygon" else 1

                for attempt in range(retries):
                    try:
                        price_data = provider.get_price(symbol)
                        if price_data and price_data > 0:
                            return SpotPrice(
                                symbol=symbol,
                                price=price_data,
                                provider=f"{name}:retry{attempt+1}" if attempt > 0 else name,
                                timestamp=int(time.time())
                            )
                    except Exception as e:
                        if attempt < retries - 1:
                            time.sleep(0.5 * (attempt + 1))  # Backoff
                        else:
                            raise
            except Exception as e:
                LOGGER.warning(f"{name} failed for {symbol}: {e}")
                continue

        # Final fallback: Redis cache (stale OK, better than nothing)

        return self._get_last_cached(symbol)

    def get_ohlcv(self, symbol: str, interval: str, lookback: int) -> Optional[OHLCVData]:
        """Try providers for OHLCV"""

        # Polygon first (paid tier = best)

        try:
            bars = self.polygon.get_ohlcv(symbol, interval, lookback)
            if bars and len(bars) >= 20:
                return OHLCVData(symbol=symbol, interval=interval, bars=bars, provider="polygon")
        except Exception as e:
            LOGGER.warning(f"Polygon OHLCV failed for {symbol}: {e}")

        # yfinance fallback

        try:
            bars = self.yfinance.get_ohlcv(symbol, interval, lookback)
            if bars and len(bars) >= 20:
                return OHLCVData(symbol=symbol, interval=interval, bars=bars, provider="yfinance")
        except Exception as e:
            LOGGER.warning(f"yfinance OHLCV failed for {symbol}: {e}")

        # Cache fallback (stale but usable)

        return self._get_cached_ohlcv(symbol, interval)

```text

### core/providers/crypto_provider_chain.py (NEW)

```python

"""
Crypto Provider Chain
=====================
Binance → CoinGecko → Coinbase → Cache fallback
"""

class CryptoProviderChain:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.binance = BinanceProvider()
        self.coingecko = CoinGeckoProvider()
        self.coinbase = CoinbaseProvider()

    def get_price(self, symbol: str) -> Optional[SpotPrice]:
        """Try crypto providers in order"""
        providers = [
            ("binance", self.binance),
            ("coingecko", self.coingecko),
            ("coinbase", self.coinbase),
        ]

        for name, provider in providers:
            try:
                price_data = provider.get_price(symbol)
                if price_data and price_data > 0:
                    return SpotPrice(
                        symbol=symbol,
                        price=price_data,
                        provider=name,
                        timestamp=int(time.time())
                    )
            except Exception as e:
                LOGGER.warning(f"{name} failed for {symbol}: {e}")
                continue

        return self._get_last_cached(symbol)

    def get_ohlcv(self, symbol: str, interval: str, lookback: int) -> Optional[OHLCVData]:
        """
        CRITICAL: Get crypto OHLCV from Binance
        This was MISSING - now crypto predictions will work!
        """

        # Binance Klines (primary, FREE unlimited)

        try:
            binance_symbol = self._ghost_to_binance(symbol)
            bars = self.binance.get_klines(binance_symbol, interval, lookback)
            if bars and len(bars) >= 20:
                return OHLCVData(symbol=symbol, interval=interval, bars=bars, provider="binance")
        except Exception as e:
            LOGGER.error(f"Binance OHLCV failed for {symbol}: {e}")

        # No good fallback for crypto OHLCV yet (CoinGecko requires paid tier)

        # Cache fallback

        return self._get_cached_ohlcv(symbol, interval)

    def _ghost_to_binance(self, symbol: str) -> str:
        """Map Ghost symbol to Binance ticker"""
        mapping = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "BNB": "BNBUSDT",
            "XRP": "XRPUSDT",
            "ADA": "ADAUSDT",
            "DOGE": "DOGEUSDT",
            "AVAX": "AVAXUSDT",
            "DOT": "DOTUSDT",
            "MATIC": "MATICUSDT",
        }
        return mapping.get(symbol.upper(), f"{symbol.upper()}USDT")

```text

### core/providers/binance_ohlcv.py (NEW)

```python

"""
Binance OHLCV Provider
======================
FREE unlimited access to crypto historical bars.
"""

import requests
from typing import List, Optional

class BinanceProvider:
    BASE_URL = "<<<<<https://api.binance.com/api/v3">>>>>

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> Optional[List[OHLCVBar]]:
        """
        Get Binance klines (candlestick data)

        Args:
            symbol: Binance ticker (e.g., "BTCUSDT")
            interval: "1m", "5m", "15m", "1h", "4h", "1d"
            limit: Number of bars (max 1000)

        Returns:
            List of OHLCV bars or None
        """
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "1d": "1d",
        }

        binance_interval = interval_map.get(interval, "1h")

        try:
            url = f"{self.BASE_URL}/klines"
            params = {
                "symbol": symbol,
                "interval": binance_interval,
                "limit": min(limit, 1000)
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Parse Binance kline format

            bars = []
            for kline in data:
                bars.append(OHLCVBar(
                    timestamp=int(kline[0]) // 1000,  # ms to seconds
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                ))

            return bars
        except Exception as e:
            LOGGER.error(f"Binance klines failed for {symbol}: {e}")
            return None

```text

---

## WIRING TO ENGINES

### Update technical_engine.py

```python

# OLD (Direct yfinance calls)

df = self._fetch_yfinance(symbol, days)

# NEW (Unified provider)

from core.providers.unified_provider import get_unified_provider

provider = get_unified_provider()
ohlcv = provider.get_ohlcv(symbol, interval="1d", lookback=90)
if ohlcv:
    df = self._ohlcv_to_dataframe(ohlcv.bars)

```text

### Update volume_engine.py

```python

# Same pattern as technical_engine

ohlcv = provider.get_ohlcv(symbol, interval="1d", lookback=90)

```text

### Update price_engine.py

```python

# OLD (_get_price_quorum)

quorum = get_price_quorum()
price = quorum.get_price(symbol, providers, ...)

# NEW (Unified provider)

provider = get_unified_provider()
spot = provider.get_spot_price(symbol)

```text

---

## HEALTH & DIAGNOSTICS

### New Endpoint: /api/v3/providers/health

```python

@APP.get("/api/v3/providers/health")
async def provider_health():
    """
    Provider health dashboard

    Returns:
        {
            "stocks": {
                "polygon": {"success_rate": 0.98, "p95_latency_ms": 45, "last_error": None},
                "yahoo": {"success_rate": 0.45, "p95_latency_ms": 890, "last_error": "429 Rate Limit"},
                "yfinance": {"success_rate": 0.32, "p95_latency_ms": 1200, "last_error": "Timeout"}
            },
            "crypto": {
                "binance": {"success_rate": 1.0, "p95_latency_ms": 52, "last_error": None},
                "coingecko": {"success_rate": 0.92, "p95_latency_ms": 234, "last_error": None},
                "coinbase": {"success_rate": 0.88, "p95_latency_ms": 198, "last_error": None}
            },
            "cache": {
                "hit_rate": 0.82,
                "total_requests": 45234,
                "cache_hits": 37092,
                "cache_misses": 8142
            }
        }
    """
    return get_unified_provider().get_health_stats()

```text

---

## DEPLOYMENT STEPS

1. **Install Binance OHLCV module**✅


2.**Add Redis caching layer**✅
3.**Refactor technical/volume engines**✅
4.**Update wolf_app price quorum**✅
5.**Add health endpoint**✅
6.**Run tests (local + production)**⏳
7.**Upgrade Polygon to paid tier**($49/month) ⏳
8.**Monitor and tune cache TTLs**⏳


---

## EXPECTED RESULTS

### Feature Extraction (Target)**Stocks**(with Polygon paid)

```text

MSFT: 25/26 features (96.2%)
AAPL: 25/26 features (96.2%)
SPY: 25/26 features (96.2%)
TSLA: 25/26 features (96.2%)

```text**Crypto**(with Binance OHLCV):

```text

BTC: 23/25 features (92%)
ETH: 23/25 features (92%)
SOL: 23/25 features (92%)
DOGE: 23/25 features (92%)

```text

### Prediction Quality**Confidence Range**: 42-75% (varied, not stuck at 40%)

**Direction Mix**: 40% UP, 35% DOWN, 25% FLAT (realistic distribution)

### Performance

**API Call Reduction**: 80% (via caching)
**Latency**: 150-300ms per prediction (down from 800-1200ms)
**Provider Uptime**: 98%+ (paid Polygon + Binance)

---

## COST ANALYSIS

| Item | Cost | Benefit |
|------|------|---------|
| Polygon Paid Tier | $49/month | Unlimited calls, 1ms latency |
| Binance Public API | FREE | Unlimited crypto OHLCV |
| CoinGecko Free | FREE | 50 calls/min crypto prices |
| Upstash Redis | $10/month | 80% API call reduction |
| **Total**|**$59/month**|**100% operational Ghost**|

ROI: $59/month eliminates:

- 429 rate limit errors
- 40% FLAT stuck predictions
- Missing crypto features
- User complaints**Worth it**: ✅ **ABSOLUTELY**


---

## NEXT: IMPLEMENTATION

See implementation commits for:

- `unified_provider.py`
- `binance_ohlcv.py`
- `crypto_provider_chain.py`
- Engine wiring updates
- Health endpoint
- Test suite
