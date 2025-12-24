"""
Price Validator - Reject stale or unrealistic prices

This module provides sanity checks for price data to prevent
predictions based on incorrect entry prices.
"""
import time
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Known approximate price ranges (updated Dec 2025)
# Format: symbol -> (min_price, max_price)
# Allow 50% outside range for volatility
PRICE_SANITY_RANGES = {
    # Major Crypto
    'BTC': (50000, 150000),
    'ETH': (1500, 5000),
    'SOL': (50, 300),
    'BNB': (400, 1000),
    'XRP': (0.3, 5),
    'ADA': (0.2, 2),
    'AVAX': (8, 100),
    'DOT': (3, 50),
    'LINK': (5, 50),
    'MATIC': (0.05, 3),
    'DOGE': (0.05, 0.5),
    'SHIB': (0.000005, 0.0001),
    'TRX': (0.05, 0.5),
    'TON': (2, 10),
    'LTC': (50, 200),
    'XLM': (0.05, 0.5),
    
    # Meme coins (volatile)
    'PEPE': (0.000001, 0.0001),
    'BONK': (0.000001, 0.0001),
    'WIF': (0.5, 10),
    'FLOKI': (0.00005, 0.001),
    
    # Major Stocks
    'AAPL': (150, 300),
    'MSFT': (350, 550),
    'GOOGL': (150, 250),
    'GOOG': (150, 250),
    'AMZN': (150, 300),
    'META': (400, 800),
    'TSLA': (150, 600),
    'NVDA': (400, 1500),
    
    # Healthcare (from Dec 23 TOP 10)
    'LLY': (700, 1200),
    'UNH': (300, 650),
    'TMO': (450, 700),
    'ISRG': (450, 700),
    
    # Other large caps
    'JPM': (150, 300),
    'V': (250, 400),
    'MA': (400, 600),
    'HD': (300, 500),
    'PG': (140, 200),
    'JNJ': (140, 200),
}


class PriceValidator:
    """
    Validates prices are reasonable and fresh before using them.
    """
    
    def __init__(self, max_age_seconds: int = 300):  # 5 min default
        self.max_age = max_age_seconds
        self._price_timestamps: dict = {}
        self._last_known_prices: dict = {}
    
    def validate_price(
        self, 
        symbol: str, 
        price: float, 
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Validate a price is reasonable and fresh.
        
        Args:
            symbol: Asset symbol (BTC, AAPL, etc.)
            price: Price to validate
            timestamp: When the price was fetched (defaults to now)
            
        Returns:
            (is_valid, reason) - True if valid, False with explanation if not
        """
        symbol = symbol.upper()
        now = time.time()
        timestamp = timestamp or now
        
        # Check for zero/negative
        if price is None or price <= 0:
            return False, f"Invalid price: {price}"
        
        # Check freshness
        age = now - timestamp
        if age > self.max_age:
            return False, f"Price too old: {age:.0f}s > {self.max_age}s max"
        
        # Check sanity range
        if symbol in PRICE_SANITY_RANGES:
            min_price, max_price = PRICE_SANITY_RANGES[symbol]
            
            # Allow 50% outside range for extreme volatility
            lower_bound = min_price * 0.5
            upper_bound = max_price * 1.5
            
            if price < lower_bound:
                return False, f"Price ${price:,.4f} suspiciously low (expected min: ${min_price:,.2f})"
            if price > upper_bound:
                return False, f"Price ${price:,.4f} suspiciously high (expected max: ${max_price:,.2f})"
        
        # Check for sudden jumps (if we have history)
        if symbol in self._last_known_prices:
            last_price = self._last_known_prices[symbol]
            if last_price > 0:
                change_pct = abs(price - last_price) / last_price * 100
                # Flag >50% changes as suspicious (but don't reject - crypto is volatile)
                if change_pct > 50:
                    logger.warning(
                        f"[PriceValidator] {symbol} price changed {change_pct:.1f}%: "
                        f"${last_price:,.2f} -> ${price:,.2f}"
                    )
        
        # Update last known price
        self._last_known_prices[symbol] = price
        
        return True, "ok"
    
    def record_price(self, symbol: str, price: float):
        """Record a price fetch timestamp"""
        symbol = symbol.upper()
        self._price_timestamps[symbol] = time.time()
        self._last_known_prices[symbol] = price
    
    def get_price_age(self, symbol: str) -> Optional[float]:
        """Get age of last recorded price in seconds"""
        ts = self._price_timestamps.get(symbol.upper())
        if ts:
            return time.time() - ts
        return None
    
    def get_last_known_price(self, symbol: str) -> Optional[float]:
        """Get last validated price for symbol"""
        return self._last_known_prices.get(symbol.upper())
    
    def is_price_fresh(self, symbol: str) -> bool:
        """Check if we have a fresh price for this symbol"""
        age = self.get_price_age(symbol)
        return age is not None and age < self.max_age
    
    def get_status(self) -> dict:
        """Get validator status for debugging"""
        now = time.time()
        fresh_count = sum(
            1 for sym in self._price_timestamps 
            if now - self._price_timestamps[sym] < self.max_age
        )
        return {
            "tracked_symbols": len(self._price_timestamps),
            "fresh_prices": fresh_count,
            "stale_prices": len(self._price_timestamps) - fresh_count,
            "max_age_seconds": self.max_age,
            "sanity_ranges_defined": len(PRICE_SANITY_RANGES),
        }


# Singleton instance
_validator = None


def get_validator() -> PriceValidator:
    """Get the singleton PriceValidator instance"""
    global _validator
    if _validator is None:
        _validator = PriceValidator()
    return _validator


def validate_price(symbol: str, price: float, timestamp: Optional[float] = None) -> Tuple[bool, str]:
    """
    Convenience function to validate a price.
    
    Returns:
        (is_valid, reason)
    """
    return get_validator().validate_price(symbol, price, timestamp)


def is_price_valid(symbol: str, price: float) -> bool:
    """Simple boolean check if price is valid"""
    valid, _ = validate_price(symbol, price)
    return valid


def record_validated_price(symbol: str, price: float):
    """Record a price that passed validation"""
    get_validator().record_price(symbol, price)
