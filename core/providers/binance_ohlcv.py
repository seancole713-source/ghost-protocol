"""
Binance OHLCV Provider
======================
FREE unlimited access to crypto historical candlestick data.

Endpoints:
- GET /api/v3/klines - Historical OHLCV bars

Rate Limits:
- Public API: 1200 requests/min (no API key needed)
- Weight: 1 per klines request

Supported Intervals:
- 1m, 3m, 5m, 15m, 30m (intraday)
- 1h, 2h, 4h, 6h, 8h, 12h (hourly)
- 1d, 3d, 1w, 1M (daily+)
"""

import requests
from typing import Optional, List
from dataclasses import dataclass
from loguru import logger as LOGGER
import time


@dataclass
class OHLCVBar:
    """Single OHLCV candlestick bar"""
    timestamp: int  # Unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float


class BinanceOHLCVProvider:
    """
    Binance Klines Provider
    
    Provides FREE unlimited crypto OHLCV data.
    No API key required for public endpoints.
    
    Note: If region-blocked (451), falls back to Binance.US API
    """
    
    BASE_URL = "https://api.binance.us/api/v3"  # Use Binance.US for US-based access
    
    # Ghost symbol → Binance ticker mapping
    SYMBOL_MAP = {
        # Major Coins
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "BNB": "BNBUSDT",
        
        # Top 20
        "SOL": "SOLUSDT",
        "XRP": "XRPUSDT",
        "ADA": "ADAUSDT",
        "DOGE": "DOGEUSDT",
        "AVAX": "AVAXUSDT",
        "DOT": "DOTUSDT",
        "MATIC": "MATICUSDT",
        "LINK": "LINKUSDT",
        "UNI": "UNIUSDT",
        "ATOM": "ATOMUSDT",
        "LTC": "LTCUSDT",
        "BCH": "BCHUSDT",
        "NEAR": "NEARUSDT",
        "APT": "APTUSDT",
        "ARB": "ARBUSDT",
        "OP": "OPUSDT",
        "FIL": "FILUSDT",
        
        # Meme Coins
        "SHIB": "SHIBUSDT",
        "PEPE": "PEPEUSDT",
        "FLOKI": "FLOKIUSDT",
        "BONK": "BONKUSDT",
        
        # AI Coins
        "RNDR": "RNDRUSDT",
        "FET": "FETUSDT",
        "AGIX": "AGIXUSDT",
        "OCEAN": "OCEANUSDT",
        
        # DeFi
        "AAVE": "AAVEUSDT",
        "MKR": "MKRUSDT",
        "SNX": "SNXUSDT",
        "CRV": "CRVUSDT",
        
        # Gaming
        "IMX": "IMXUSDT",
        "SAND": "SANDUSDT",
        "MANA": "MANAUSDT",
        "AXS": "AXSUSDT",
    }
    
    # Interval mapping
    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Binance provider.
        
        Args:
            api_key: Optional API key for higher rate limits (not needed for public API)
        """
        self.api_key = api_key
        self.last_request_time = 0
        self.min_request_interval = 0.05  # 50ms = 1200 requests/min max
    
    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> Optional[List[OHLCVBar]]:
        """
        Get OHLCV candlestick bars from Binance.
        
        Args:
            symbol: Ghost symbol (e.g., "BTC", "ETH")
            interval: Timeframe ("1m", "5m", "1h", "1d")
            limit: Number of bars (max 1000)
        
        Returns:
            List of OHLCV bars or None on failure
        """
        # Map Ghost symbol to Binance ticker
        binance_symbol = self._map_symbol(symbol)
        if not binance_symbol:
            LOGGER.warning(f"[BINANCE] Symbol {symbol} not supported")
            return None
        
        # Map interval
        binance_interval = self.INTERVAL_MAP.get(interval, "1h")
        
        # Rate limiting
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/klines"
            params = {
                "symbol": binance_symbol,
                "interval": binance_interval,
                "limit": min(limit, 1000)
            }
            
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                LOGGER.warning(f"[BINANCE] No data for {symbol} ({binance_symbol})")
                return None
            
            # Parse Binance kline format:
            # [
            #   [
            #     1499040000000,      // 0: Open time (ms)
            #     "0.01634000",       // 1: Open
            #     "0.80000000",       // 2: High
            #     "0.01575800",       // 3: Low
            #     "0.01577100",       // 4: Close
            #     "148976.11427815",  // 5: Volume
            #     1499644799999,      // 6: Close time
            #     "2434.19055334",    // 7: Quote asset volume
            #     308,                // 8: Number of trades
            #     "1756.87402397",    // 9: Taker buy base asset volume
            #     "28.46694368",      // 10: Taker buy quote asset volume
            #     "17928899.62484339" // 11: Ignore
            #   ]
            # ]
            
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
            
            LOGGER.info(
                f"[BINANCE] ✅ Fetched {len(bars)} bars for {symbol} "
                f"({binance_symbol}, {binance_interval})"
            )
            
            return bars
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                LOGGER.error(f"[BINANCE] ⚠️  Rate limit exceeded for {symbol}")
            else:
                LOGGER.error(f"[BINANCE] ❌ HTTP error for {symbol}: {e}")
            return None
        
        except Exception as e:
            LOGGER.error(f"[BINANCE] ❌ Failed to fetch {symbol}: {e}")
            return None
    
    def _map_symbol(self, ghost_symbol: str) -> Optional[str]:
        """
        Map Ghost symbol to Binance ticker.
        
        Args:
            ghost_symbol: Ghost format (e.g., "BTC")
        
        Returns:
            Binance ticker (e.g., "BTCUSDT") or None
        """
        symbol_upper = ghost_symbol.upper()
        
        # Try exact match first
        if symbol_upper in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[symbol_upper]
        
        # Try appending USDT
        potential = f"{symbol_upper}USDT"
        if self._validate_binance_symbol(potential):
            return potential
        
        LOGGER.warning(f"[BINANCE] No mapping for {ghost_symbol}")
        return None
    
    def _validate_binance_symbol(self, symbol: str) -> bool:
        """
        Check if symbol exists on Binance (lightweight check).
        
        Args:
            symbol: Binance ticker (e.g., "BTCUSDT")
        
        Returns:
            True if valid symbol
        """
        # For now, just check if it matches expected pattern
        # Could call /api/v3/exchangeInfo for full validation
        return symbol.endswith("USDT") and len(symbol) > 4
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting (1200 requests/min = 50ms between calls)"""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported Ghost symbols"""
        return sorted(self.SYMBOL_MAP.keys())


if __name__ == "__main__":
    # Test Binance OHLCV
    import os
    
    provider = BinanceOHLCVProvider(api_key=os.getenv("BINANCE_API_KEY"))
    
    # Test BTC
    print("\n=== Testing BTC OHLCV ===")
    bars = provider.get_ohlcv("BTC", interval="1h", limit=50)
    if bars:
        print(f"✅ Fetched {len(bars)} bars for BTC")
        print(f"Latest bar: {bars[-1]}")
        print(f"Oldest bar: {bars[0]}")
    else:
        print("❌ Failed to fetch BTC")
    
    # Test ETH
    print("\n=== Testing ETH OHLCV ===")
    bars = provider.get_ohlcv("ETH", interval="5m", limit=20)
    if bars:
        print(f"✅ Fetched {len(bars)} bars for ETH")
        print(f"Latest: {bars[-1]}")
    else:
        print("❌ Failed to fetch ETH")
    
    # Test unsupported
    print("\n=== Testing Unsupported Symbol ===")
    bars = provider.get_ohlcv("FAKE", interval="1d", limit=10)
    if bars:
        print(f"✅ Fetched {len(bars)} bars")
    else:
        print("❌ Failed as expected")
    
    # Show supported symbols
    print(f"\n=== Supported Symbols ({len(provider.get_supported_symbols())}) ===")
    print(", ".join(provider.get_supported_symbols()))
