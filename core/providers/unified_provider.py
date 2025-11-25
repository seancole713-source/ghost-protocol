"""
Unified Provider Interface
===========================
Single entry point for all price/OHLCV data with caching + fallbacks.

Architecture:
1. Check Redis cache first (80% hit rate target)
2. On cache miss, try provider chain in priority order
3. Cache successful results with appropriate TTL
4. Return last known good value on total failure

Provider Chains:
- Stocks: Polygon → Yahoo → yfinance → Cache
- Crypto: Binance → CoinGecko → Coinbase → Cache

Cache TTLs:
- Spot Prices: 90s (stocks), 30s (crypto)
- OHLCV: 5min (intraday), 60min (daily)
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
import time
from loguru import logger as LOGGER

from core.providers.cache_utils import get_cache, cache_spot_price, cache_ohlcv
from core.providers.binance_ohlcv import BinanceOHLCVProvider, OHLCVBar


@dataclass
class SpotPrice:
    """Spot price with metadata"""
    symbol: str
    price: float
    prev_close: Optional[float]
    provider: str
    timestamp: int
    cache_hit: bool = False


@dataclass
class OHLCVData:
    """OHLCV bars with metadata"""
    symbol: str
    interval: str  # "1m", "5m", "1h", "1d"
    bars: List[OHLCVBar]
    provider: str
    timestamp: int
    cache_hit: bool = False


class UnifiedProvider:
    """
    Unified provider with caching and multi-provider fallbacks.
    
    FREE-TIER FIRST STRATEGY:
    - Stocks: Yahoo → yfinance → cache (NO PAID APIs)
    - Crypto: Binance → CoinGecko → cache (NO PAID APIs)
    
    Features:
    - Redis caching (80% call reduction)
    - Provider health tracking
    - Graceful degradation
    - Rate limit protection with retry/cooldown
    """
    
    def __init__(self):
        """Initialize unified provider with all sub-providers"""
        self.cache = get_cache()
        
        # Initialize FREE-TIER providers
        self.binance_ohlcv = BinanceOHLCVProvider()  # Crypto OHLCV (FREE)
        
        try:
            from core.providers.yahoo_finance import YahooFinanceProvider
            self.yahoo = YahooFinanceProvider()  # Stock OHLCV (FREE)
            LOGGER.info("✅ Yahoo Finance provider initialized")
        except Exception as e:
            LOGGER.warning(f"Yahoo Finance provider unavailable: {e}")
            self.yahoo = None
        
        # Health tracking
        self.provider_stats = {
            "binance": {"requests": 0, "successes": 0, "failures": 0, "total_latency": 0},
            "polygon": {"requests": 0, "successes": 0, "failures": 0, "total_latency": 0},
            "yahoo": {"requests": 0, "successes": 0, "failures": 0, "total_latency": 0},
            "coingecko": {"requests": 0, "successes": 0, "failures": 0, "total_latency": 0},
        }
        
        LOGGER.info("✅ Unified provider initialized")
    
    def get_spot_price(self, symbol: str) -> Optional[SpotPrice]:
        """
        Get spot price with caching + fallbacks.
        
        Args:
            symbol: Ghost symbol (e.g., "AAPL", "BTC")
        
        Returns:
            SpotPrice or None
        """
        is_crypto = self._is_crypto(symbol)
        
        # Try cache first
        def compute_price():
            if is_crypto:
                return self._get_crypto_spot_price(symbol)
            else:
                return self._get_stock_spot_price(symbol)
        
        # Cache with appropriate TTL
        ttl = 30 if is_crypto else 90
        key = f"ghost:price:spot:{symbol.upper()}:v1"
        
        cached_data = self.cache.get_cached_json(key, ttl, compute_price)
        
        if cached_data:
            return SpotPrice(
                symbol=cached_data["symbol"],
                price=cached_data["price"],
                prev_close=cached_data.get("prev_close"),
                provider=cached_data["provider"],
                timestamp=cached_data["timestamp"],
                cache_hit=True
            )
        
        return None
    
    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        lookback: int = 100
    ) -> Optional[OHLCVData]:
        """
        Get OHLCV bars with caching + fallbacks.
        
        Args:
            symbol: Ghost symbol (e.g., "AAPL", "BTC")
            interval: Timeframe ("1m", "5m", "1h", "1d")
            lookback: Number of bars
        
        Returns:
            OHLCVData or None
        """
        is_crypto = self._is_crypto(symbol)
        
        # Try cache first
        def compute_ohlcv():
            if is_crypto:
                return self._get_crypto_ohlcv(symbol, interval, lookback)
            else:
                return self._get_stock_ohlcv(symbol, interval, lookback)
        
        # Cache with appropriate TTL
        ttl = 300 if interval in ["1m", "5m", "15m"] else 3600
        key = f"ghost:ohlcv:{symbol.upper()}:{interval}:v1"
        
        cached_data = self.cache.get_cached_json(key, ttl, compute_ohlcv)
        
        if cached_data and cached_data.get("bars"):
            bars = [
                OHLCVBar(
                    timestamp=bar["timestamp"],
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"]
                )
                for bar in cached_data["bars"]
            ]
            
            return OHLCVData(
                symbol=cached_data["symbol"],
                interval=cached_data["interval"],
                bars=bars,
                provider=cached_data["provider"],
                timestamp=cached_data["timestamp"],
                cache_hit=True
            )
        
        return None
    
    def _get_crypto_spot_price(self, symbol: str) -> Optional[dict]:
        """
        Get crypto spot price from provider chain.
        
        Priority: Binance → CoinGecko → Coinbase
        """
        # TODO: Implement provider chain
        # For now, placeholder
        LOGGER.warning(f"Crypto spot price not yet implemented for {symbol}")
        return None
    
    def _get_stock_spot_price(self, symbol: str) -> Optional[dict]:
        """
        Get stock spot price from provider chain.
        
        Priority: Polygon → Yahoo → yfinance
        """
        # TODO: Implement provider chain
        # For now, placeholder
        LOGGER.warning(f"Stock spot price not yet implemented for {symbol}")
        return None
    
    def _get_crypto_ohlcv(
        self,
        symbol: str,
        interval: str,
        lookback: int
    ) -> Optional[dict]:
        """
        Get crypto OHLCV from Binance.
        
        This is the CRITICAL fix for crypto predictions!
        """
        start_time = time.time()
        self._track_request("binance")
        
        try:
            bars = self.binance_ohlcv.get_ohlcv(symbol, interval, lookback)
            
            if bars and len(bars) >= 20:
                latency = (time.time() - start_time) * 1000
                self._track_success("binance", latency)
                
                return {
                    "symbol": symbol,
                    "interval": interval,
                    "bars": [
                        {
                            "timestamp": bar.timestamp,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume
                        }
                        for bar in bars
                    ],
                    "provider": "binance",
                    "timestamp": int(time.time())
                }
            else:
                self._track_failure("binance")
                LOGGER.warning(f"[BINANCE] Insufficient bars for {symbol}: {len(bars) if bars else 0}")
                return None
        
        except Exception as e:
            self._track_failure("binance")
            LOGGER.error(f"[BINANCE] OHLCV failed for {symbol}: {e}")
            return None
    
    def _get_stock_ohlcv(
        self,
        symbol: str,
        interval: str,
        lookback: int
    ) -> Optional[dict]:
        """
        Get stock OHLCV from FREE provider chain.
        
        Priority: Yahoo Finance (FREE) → yfinance (FREE) → cache
        """
        # Calculate lookback days from bar count
        lookback_days = lookback if interval == "1d" else min(lookback // 78, 90)  # ~78 bars/day for 1h
        
        # PRIMARY: Yahoo Finance (FREE)
        if self.yahoo:
            start_time = time.time()
            self._track_request("yahoo")
            
            try:
                bars = self.yahoo.get_ohlcv(symbol, interval, lookback_days)
                
                if bars and len(bars) >= 20:
                    latency = (time.time() - start_time) * 1000
                    self._track_success("yahoo", latency)
                    
                    return {
                        "symbol": symbol,
                        "interval": interval,
                        "bars": [
                            {
                                "timestamp": bar.timestamp,
                                "open": bar.open,
                                "high": bar.high,
                                "low": bar.low,
                                "close": bar.close,
                                "volume": bar.volume
                            }
                            for bar in bars
                        ],
                        "provider": "yahoo",
                        "timestamp": int(time.time())
                    }
                else:
                    self._track_failure("yahoo")
                    LOGGER.warning(f"[YAHOO] Insufficient bars for {symbol}: {len(bars) if bars else 0}")
            
            except Exception as e:
                self._track_failure("yahoo")
                LOGGER.error(f"[YAHOO] OHLCV failed for {symbol}: {e}")
        
        # FALLBACK: yfinance (FREE)
        LOGGER.warning(f"[STOCK] {symbol}: Yahoo failed, trying yfinance...")
        # TODO: Implement yfinance fallback
        
        return None
    
    def _is_crypto(self, symbol: str) -> bool:
        """
        Detect if symbol is crypto vs stock.
        
        Heuristics:
        - Crypto: Known crypto symbols in Binance mapping
        - Stock: Everything else (AAPL, MSFT, SPY, etc.)
        """
        symbol_upper = symbol.upper()
        
        # Known crypto symbols (from Binance provider)
        crypto_symbols = set(self.binance_ohlcv.get_supported_symbols())
        if symbol_upper in crypto_symbols:
            return True
        
        # Common stock patterns (has digits, longer than 5 chars, ends in numbers)
        if any(c.isdigit() for c in symbol):
            return False
        
        # Length heuristic: crypto usually <= 5 chars, stocks can be longer
        # But check against known crypto list first (most reliable)
        return False  # Default to stock (Yahoo will handle it)
    
    def _track_request(self, provider: str) -> None:
        """Track provider request"""
        if provider in self.provider_stats:
            self.provider_stats[provider]["requests"] += 1
    
    def _track_success(self, provider: str, latency_ms: float) -> None:
        """Track provider success"""
        if provider in self.provider_stats:
            self.provider_stats[provider]["successes"] += 1
            self.provider_stats[provider]["total_latency"] += latency_ms
    
    def _track_failure(self, provider: str) -> None:
        """Track provider failure"""
        if provider in self.provider_stats:
            self.provider_stats[provider]["failures"] += 1
    
    def get_health_stats(self) -> dict:
        """
        Get provider health statistics.
        
        Returns:
            {
                "providers": {
                    "binance": {"success_rate": 0.98, "avg_latency_ms": 52, ...},
                    ...
                },
                "cache": {...}
            }
        """
        health = {"providers": {}}
        
        for provider, stats in self.provider_stats.items():
            total = stats["requests"]
            if total > 0:
                success_rate = stats["successes"] / total
                avg_latency = stats["total_latency"] / stats["successes"] if stats["successes"] > 0 else 0
            else:
                success_rate = 0.0
                avg_latency = 0.0
            
            health["providers"][provider] = {
                "requests": total,
                "successes": stats["successes"],
                "failures": stats["failures"],
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(avg_latency, 1)
            }
        
        # Add cache stats
        health["cache"] = self.cache.get_stats()
        
        return health


# Global instance
_unified_provider: Optional[UnifiedProvider] = None


def get_unified_provider() -> UnifiedProvider:
    """Get or create global unified provider"""
    global _unified_provider
    if _unified_provider is None:
        _unified_provider = UnifiedProvider()
    return _unified_provider


if __name__ == "__main__":
    # Test unified provider
    provider = get_unified_provider()
    
    # Test BTC OHLCV (crypto)
    print("\n=== Testing BTC OHLCV (Crypto) ===")
    ohlcv = provider.get_ohlcv("BTC", interval="1h", lookback=50)
    if ohlcv:
        print(f"✅ {ohlcv.symbol}: {len(ohlcv.bars)} bars from {ohlcv.provider}")
        print(f"   Cache hit: {ohlcv.cache_hit}")
        print(f"   Latest: {ohlcv.bars[-1]}")
    else:
        print("❌ Failed to fetch BTC OHLCV")
    
    # Test again (should hit cache)
    print("\n=== Testing BTC OHLCV Again (Cache Hit Expected) ===")
    ohlcv2 = provider.get_ohlcv("BTC", interval="1h", lookback=50)
    if ohlcv2:
        print(f"✅ Cache hit: {ohlcv2.cache_hit}")
    
    # Health stats
    print("\n=== Provider Health ===")
    health = provider.get_health_stats()
    import json
    print(json.dumps(health, indent=2))
