"""
Phase 2.4: Additional Price Provider - FMP (Financial Modeling Prep)

Adds Financial Modeling Prep as a third stock price provider for redundancy.
FMP provides real-time stock quotes with generous free tier (250 calls/day).
"""

import os
import httpx
from typing import Optional, Dict, Any
from datetime import datetime
from core.logger import get_logger

LOGGER = get_logger(__name__)


class FMPProvider:
    """
    Financial Modeling Prep price provider.
    
    API Documentation: https://site.financialmodelingprep.com/developer/docs
    Free tier: 250 API calls/day
    """
    
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY", "")
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            LOGGER.info("[FMP] API key not configured - provider disabled")
    
    async def get_quote(self, symbol: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """
        Get real-time stock quote from FMP.
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            timeout: Request timeout in seconds
        
        Returns:
            {
                "price": 175.32,
                "change": 1.25,
                "change_pct": 0.72,
                "volume": 52183900,
                "timestamp": "2026-03-21T14:32:15"
            }
        """
        if not self.enabled:
            return None
        
        try:
            url = f"{self.base_url}/quote/{symbol}"
            params = {"apikey": self.api_key}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                
                if not data or len(data) == 0:
                    LOGGER.warning(f"[FMP] No data returned for {symbol}")
                    return None
                
                quote = data[0]  # FMP returns array with single quote
                
                return {
                    "price": quote.get("price"),
                    "change": quote.get("change"),
                    "change_pct": quote.get("changesPercentage"),
                    "volume": quote.get("volume"),
                    "timestamp": quote.get("timestamp"),
                    "provider": "fmp"
                }
        
        except httpx.TimeoutException:
            LOGGER.warning(f"[FMP] Timeout fetching {symbol}")
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                LOGGER.warning("[FMP] Rate limit exceeded")
            else:
                LOGGER.error(f"[FMP] HTTP error {e.response.status_code}")
            return None
        except Exception as e:
            LOGGER.error(f"[FMP] Error fetching {symbol}: {e}")
            return None
    
    async def get_batch_quotes(self, symbols: list[str], timeout: float = 5.0) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for multiple symbols (up to 20 at once).
        
        Args:
            symbols: List of stock tickers
            timeout: Request timeout in seconds
        
        Returns:
            {"AAPL": {...quote...}, "NVDA": {...quote...}}
        """
        if not self.enabled or not symbols:
            return {}
        
        try:
            # FMP supports batch quotes with comma-separated symbols
            symbols_str = ",".join(symbols[:20])  # Max 20 symbols
            url = f"{self.base_url}/quote/{symbols_str}"
            params = {"apikey": self.api_key}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                
                result = {}
                for quote in data:
                    symbol = quote.get("symbol")
                    if symbol:
                        result[symbol] = {
                            "price": quote.get("price"),
                            "change": quote.get("change"),
                            "change_pct": quote.get("changesPercentage"),
                            "volume": quote.get("volume"),
                            "timestamp": quote.get("timestamp"),
                            "provider": "fmp"
                        }
                
                return result
        
        except Exception as e:
            LOGGER.error(f"[FMP] Batch quote error: {e}")
            return {}


# Singleton instance
_fmp_provider: Optional[FMPProvider] = None


def get_fmp_provider() -> FMPProvider:
    """Get or create FMP provider singleton."""
    global _fmp_provider
    if _fmp_provider is None:
        _fmp_provider = FMPProvider()
    return _fmp_provider
