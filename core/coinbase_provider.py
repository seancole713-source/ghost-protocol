"""
Coinbase Price Provider - Reliable, no API key needed, generous rate limits

This is the PRIMARY crypto price provider for Ghost Protocol.
- No API key required
- Rate limit: ~10,000 requests/hour (very generous)
- Returns ALL prices in ONE call (efficient)
- Never rate limited in production
"""
import requests
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CoinbaseProvider:
    """
    Uses Coinbase exchange rates API - no API key required
    Rate limit: ~10,000 requests/hour (very generous)
    """
    
    BASE_URL = "https://api.coinbase.com/v2/exchange-rates"
    
    # Map common symbols to Coinbase currency codes
    SYMBOL_MAP = {
        'BTC': 'BTC',
        'ETH': 'ETH',
        'SOL': 'SOL',
        'BNB': 'BNB',
        'XRP': 'XRP',
        'ADA': 'ADA',
        'AVAX': 'AVAX',
        'DOT': 'DOT',
        'MATIC': 'MATIC',
        'LINK': 'LINK',
        'UNI': 'UNI',
        'ATOM': 'ATOM',
        'LTC': 'LTC',
        'DOGE': 'DOGE',
        'SHIB': 'SHIB',
        'XLM': 'XLM',
        'ALGO': 'ALGO',
        'VET': 'VET',
        'FIL': 'FIL',
        'AAVE': 'AAVE',
        'EOS': 'EOS',
        'XTZ': 'XTZ',
        'THETA': 'THETA',
        'XMR': 'XMR',
        'NEO': 'NEO',
        'MKR': 'MKR',
        'COMP': 'COMP',
        'SNX': 'SNX',
        'GRT': 'GRT',
        'SUSHI': 'SUSHI',
        'YFI': 'YFI',
        'CRV': 'CRV',
        'SAND': 'SAND',
        'MANA': 'MANA',
        'AXS': 'AXS',
        'ENJ': 'ENJ',
        'CHZ': 'CHZ',
        'BAT': 'BAT',
        'ZRX': 'ZRX',
        'ANKR': 'ANKR',
        'LRC': 'LRC',
        'STORJ': 'STORJ',
        '1INCH': '1INCH',
        'APE': 'APE',
        'OP': 'OP',
        'ARB': 'ARB',
        'SUI': 'SUI',
        'SEI': 'SEI',
        'TIA': 'TIA',
        'INJ': 'INJ',
        'NEAR': 'NEAR',
        'ROSE': 'ROSE',
        'FTM': 'FTM',
        'EGLD': 'EGLD',
        'HBAR': 'HBAR',
        'QNT': 'QNT',
        'METIS': 'METIS',
        'TRX': 'TRX',
        'TON': 'TON',
        'PEPE': 'PEPE',
        'BONK': 'BONK',
        'WIF': 'WIF',
        'FLOKI': 'FLOKI',
    }
    
    def __init__(self):
        self._cache: Dict[str, float] = {}
        self._cache_timestamp: float = 0
        self._cache_ttl: int = 60  # Cache for 60 seconds
    
    def _fetch_all_rates(self) -> Dict[str, float]:
        """Fetch all exchange rates from Coinbase in ONE call"""
        now = time.time()
        
        # Return cached if fresh
        if self._cache and (now - self._cache_timestamp) < self._cache_ttl:
            return self._cache
        
        try:
            response = requests.get(
                f"{self.BASE_URL}?currency=USD",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            rates = data.get('data', {}).get('rates', {})
            
            # Convert rates (they're USD -> crypto, we need crypto -> USD)
            prices = {}
            for symbol, rate in rates.items():
                try:
                    # Rate is "how many crypto per 1 USD", we want "USD per 1 crypto"
                    prices[symbol] = 1.0 / float(rate)
                except (ValueError, ZeroDivisionError):
                    continue
            
            self._cache = prices
            self._cache_timestamp = now
            logger.info(f"[CoinbaseProvider] Fetched {len(prices)} prices")
            return prices
            
        except Exception as e:
            logger.error(f"[CoinbaseProvider] Error fetching rates: {e}")
            # Return stale cache if available
            if self._cache:
                logger.warning("[CoinbaseProvider] Returning stale cache")
                return self._cache
            return {}
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price for a single symbol"""
        symbol = symbol.upper().replace('-USD', '').replace('USDT', '')
        
        # Map to Coinbase symbol if needed
        cb_symbol = self.SYMBOL_MAP.get(symbol, symbol)
        
        rates = self._fetch_all_rates()
        price = rates.get(cb_symbol)
        
        if price:
            logger.debug(f"[CoinbaseProvider] {symbol}: ${price:,.2f}")
        else:
            logger.debug(f"[CoinbaseProvider] No price for {symbol}")
        
        return price
    
    def get_prices_batch(self, symbols: list) -> Dict[str, float]:
        """Get prices for multiple symbols (efficient - one API call)"""
        rates = self._fetch_all_rates()
        
        result = {}
        for symbol in symbols:
            symbol_clean = symbol.upper().replace('-USD', '').replace('USDT', '')
            cb_symbol = self.SYMBOL_MAP.get(symbol_clean, symbol_clean)
            
            if cb_symbol in rates:
                result[symbol_clean] = rates[cb_symbol]
        
        logger.info(f"[CoinbaseProvider] Batch: {len(result)}/{len(symbols)} prices found")
        return result
    
    def is_available(self) -> bool:
        """Check if provider is working"""
        try:
            response = requests.get(
                f"{self.BASE_URL}?currency=USD",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_cache_info(self) -> dict:
        """Get cache statistics"""
        now = time.time()
        age = now - self._cache_timestamp if self._cache_timestamp > 0 else None
        return {
            "cache_size": len(self._cache),
            "cache_age_seconds": round(age, 1) if age else None,
            "cache_ttl_seconds": self._cache_ttl,
            "is_fresh": age is not None and age < self._cache_ttl,
        }


# Singleton instance
_provider = None


def get_coinbase_provider() -> CoinbaseProvider:
    """Get the singleton CoinbaseProvider instance"""
    global _provider
    if _provider is None:
        _provider = CoinbaseProvider()
    return _provider


def get_crypto_price(symbol: str) -> Optional[float]:
    """Convenience function to get a single crypto price"""
    return get_coinbase_provider().get_price(symbol)


def get_crypto_prices_batch(symbols: list) -> Dict[str, float]:
    """Convenience function to get multiple crypto prices"""
    return get_coinbase_provider().get_prices_batch(symbols)


def is_coinbase_available() -> bool:
    """Check if Coinbase API is available"""
    return get_coinbase_provider().is_available()
