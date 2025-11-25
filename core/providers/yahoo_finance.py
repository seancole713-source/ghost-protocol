"""
Yahoo Finance Provider
======================
FREE stock/ETF OHLCV data (rate-limited).

Endpoints:
- GET /v8/finance/chart/{symbol} - Historical OHLCV

Rate Limits:
- FREE: ~2000 requests/hour (~33/min)
- Must implement: caching, cooldown, retry with backoff

NO API KEY REQUIRED - 100% FREE
"""

import requests
import time
from typing import Optional, List
from dataclasses import dataclass
from loguru import logger as LOGGER
from datetime import datetime, timedelta


@dataclass
class OHLCVBar:
    """Single OHLCV candlestick bar"""
    timestamp: int  # Unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float


class YahooFinanceProvider:
    """
    Yahoo Finance FREE provider
    
    Features:
    - FREE unlimited access (rate-limited)
    - Stocks, ETFs, indices
    - Historical OHLCV
    - NO API KEY needed
    
    Caveats:
    - Rate limits: ~33 requests/min
    - Must use cooldown + caching
    - 429 errors require exponential backoff
    """
    
    BASE_URL = "https://query1.finance.yahoo.com"
    
    def __init__(self):
        """Initialize Yahoo Finance provider with rate limiting"""
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds = 30/min max (safe)
        self.retry_count = 0
        self.max_retries = 3
        self.backoff_base = 5.0  # Start with 5 second backoff
    
    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        lookback_days: int = 90
    ) -> Optional[List[OHLCVBar]]:
        """
        Get OHLCV bars from Yahoo Finance.
        
        Args:
            symbol: Stock ticker (e.g., "AAPL", "MSFT", "SPY")
            interval: Timeframe ("1m", "5m", "1h", "1d")
            lookback_days: Days of historical data
        
        Returns:
            List of OHLCV bars or None on failure
        """
        # Map interval to Yahoo format
        yahoo_interval = self._map_interval(interval)
        
        # Calculate time range
        end_time = int(time.time())
        start_time = end_time - (lookback_days * 86400)
        
        # Rate limiting with cooldown
        self._rate_limit()
        
        # Build URL
        url = f"{self.BASE_URL}/v8/finance/chart/{symbol}"
        params = {
            "interval": yahoo_interval,
            "period1": start_time,
            "period2": end_time,
            "includePrePost": "false"
        }
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=10
                )
                
                # Handle 429 rate limit
                if response.status_code == 429:
                    backoff = self.backoff_base * (2 ** attempt)
                    LOGGER.warning(
                        f"[YAHOO] ⚠️  Rate limit (429) for {symbol}, "
                        f"retry {attempt+1}/{self.max_retries} after {backoff}s"
                    )
                    time.sleep(backoff)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Parse Yahoo Finance response
                bars = self._parse_response(data, symbol)
                
                if bars and len(bars) >= 20:
                    LOGGER.info(
                        f"[YAHOO] ✅ Fetched {len(bars)} bars for {symbol} ({yahoo_interval})"
                    )
                    return bars
                else:
                    LOGGER.warning(
                        f"[YAHOO] Insufficient bars for {symbol}: {len(bars) if bars else 0}"
                    )
                    return None
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Already handled above
                    continue
                else:
                    LOGGER.error(f"[YAHOO] ❌ HTTP error for {symbol}: {e}")
                    return None
            
            except Exception as e:
                LOGGER.error(f"[YAHOO] ❌ Failed to fetch {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    backoff = self.backoff_base * (2 ** attempt)
                    LOGGER.warning(f"[YAHOO] Retry {attempt+1}/{self.max_retries} after {backoff}s")
                    time.sleep(backoff)
                else:
                    return None
        
        # All retries exhausted
        LOGGER.error(f"[YAHOO] ❌ All retries exhausted for {symbol}")
        return None
    
    def _parse_response(self, data: dict, symbol: str) -> Optional[List[OHLCVBar]]:
        """Parse Yahoo Finance JSON response"""
        try:
            result = data.get("chart", {}).get("result", [])
            if not result:
                LOGGER.warning(f"[YAHOO] No results in response for {symbol}")
                return None
            
            quote = result[0]
            timestamps = quote.get("timestamp", [])
            indicators = quote.get("indicators", {}).get("quote", [])
            
            if not indicators:
                LOGGER.warning(f"[YAHOO] No indicators in response for {symbol}")
                return None
            
            ohlcv_data = indicators[0]
            opens = ohlcv_data.get("open", [])
            highs = ohlcv_data.get("high", [])
            lows = ohlcv_data.get("low", [])
            closes = ohlcv_data.get("close", [])
            volumes = ohlcv_data.get("volume", [])
            
            # Build bars
            bars = []
            for i in range(len(timestamps)):
                # Skip bars with None values
                if (opens[i] is None or highs[i] is None or 
                    lows[i] is None or closes[i] is None):
                    continue
                
                bars.append(OHLCVBar(
                    timestamp=timestamps[i],
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=float(volumes[i]) if volumes[i] is not None else 0.0
                ))
            
            return bars
        
        except Exception as e:
            LOGGER.error(f"[YAHOO] Failed to parse response for {symbol}: {e}")
            return None
    
    def _map_interval(self, interval: str) -> str:
        """Map Ghost interval to Yahoo interval"""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
        }
        return mapping.get(interval, "1d")
    
    def _get_headers(self) -> dict:
        """Get request headers"""
        return {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting (2 seconds = 30/min safe)"""
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            LOGGER.debug(f"[YAHOO] Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


if __name__ == "__main__":
    # Test Yahoo Finance provider
    provider = YahooFinanceProvider()
    
    # Test AAPL
    print("\n=== Testing AAPL (Stock) ===")
    bars = provider.get_ohlcv("AAPL", interval="1d", lookback_days=60)
    if bars:
        print(f"✅ Fetched {len(bars)} bars for AAPL")
        print(f"Latest bar: {bars[-1]}")
        print(f"Oldest bar: {bars[0]}")
    else:
        print("❌ Failed to fetch AAPL")
    
    # Test SPY
    print("\n=== Testing SPY (ETF) ===")
    bars = provider.get_ohlcv("SPY", interval="1d", lookback_days=60)
    if bars:
        print(f"✅ Fetched {len(bars)} bars for SPY")
        print(f"Latest: {bars[-1]}")
    else:
        print("❌ Failed to fetch SPY")
    
    # Test MSFT
    print("\n=== Testing MSFT (Stock) ===")
    bars = provider.get_ohlcv("MSFT", interval="1d", lookback_days=60)
    if bars:
        print(f"✅ Fetched {len(bars)} bars for MSFT")
        print(f"Latest: {bars[-1]}")
    else:
        print("❌ Failed to fetch MSFT")
