"""
Phase 2.4: Additional Crypto Provider - Coinbase

Adds Coinbase as additional crypto price provider for redundancy.
Coinbase provides free public API for spot prices (no API key required).
"""

import httpx
from typing import Optional, Dict, Any
from core.logger import get_logger

LOGGER = get_logger(__name__)


class CoinbaseProvider:
    """
    Coinbase public API price provider.
    
    API Documentation: https://docs.cloud.coinbase.com/exchange/reference
    No API key required for public endpoints
    Rate limit: 10 requests/second (public)
    """
    
    def __init__(self):
        self.base_url = "https://api.coinbase.com/v2"
        self.enabled = True
    
    async def get_spot_price(self, symbol: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """
        Get spot price for crypto symbol.
        
        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            timeout: Request timeout in seconds
        
        Returns:
            {
                "price": 68432.15,
                "currency": "USD",
                "timestamp": "2026-03-21T14:32:15Z",
                "provider": "coinbase"
            }
        """
        try:
            # Coinbase uses symbol-USD format
            pair = f"{symbol.upper()}-USD"
            url = f"{self.base_url}/prices/{pair}/spot"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                
                if "data" not in data:
                    LOGGER.warning(f"[Coinbase] No data returned for {symbol}")
                    return None
                
                spot = data["data"]
                price = float(spot.get("amount", 0))
                
                if price <= 0:
                    return None
                
                return {
                    "price": price,
                    "currency": spot.get("currency", "USD"),
                    "timestamp": data.get("timestamp"),
                    "provider": "coinbase"
                }
        
        except httpx.TimeoutException:
            LOGGER.warning(f"[Coinbase] Timeout fetching {symbol}")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                LOGGER.warning("[Coinbase] Rate limit exceeded")
            elif e.response.status_code == 404:
                LOGGER.debug(f"[Coinbase] Symbol {symbol} not found")
            else:
                LOGGER.error(f"[Coinbase] HTTP error {e.response.status_code}")
            return None
        except Exception as e:
            LOGGER.error(f"[Coinbase] Error fetching {symbol}: {e}")
            return None
    
    async def get_batch_prices(self, symbols: list[str], timeout: float = 5.0) -> Dict[str, Dict[str, Any]]:
        """
        Get spot prices for multiple symbols.
        Note: Coinbase doesn't support batch requests, so we make concurrent calls.
        
        Args:
            symbols: List of crypto symbols
            timeout: Request timeout per symbol
        
        Returns:
            {"BTC": {...price...}, "ETH": {...price...}}
        """
        import asyncio
        
        if not symbols:
            return {}
        
        # Limit concurrent requests to avoid rate limiting
        results = {}
        batch_size = 5
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [self.get_spot_price(symbol, timeout) for symbol in batch]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, response in zip(batch, responses):
                if isinstance(response, dict) and response:
                    results[symbol] = response
        
        return results


# Singleton instance
_coinbase_provider: Optional[CoinbaseProvider] = None


def get_coinbase_provider() -> CoinbaseProvider:
    """Get or create Coinbase provider singleton."""
    global _coinbase_provider
    if _coinbase_provider is None:
        _coinbase_provider = CoinbaseProvider()
    return _coinbase_provider
