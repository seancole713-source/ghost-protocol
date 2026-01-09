"""
BTC Correlation Features for Ghost Protocol
============================================

Calculates correlation and lead/lag features relative to Bitcoin.
BTC often leads the crypto market - when BTC moves, altcoins follow.

Key Insights:
- BTC up + altcoin prediction = more confident UP
- BTC down + altcoin prediction = more confident DOWN
- BTC/altcoin correlation > 0.7 = strong relationship
- BTC lead time is typically 1-6 hours for major moves
"""

import os
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

LOGGER = logging.getLogger(__name__)

# Cache for BTC data (avoid fetching repeatedly)
_BTC_CACHE: Dict = {
    "price": None,
    "rsi": None,
    "macd": None,
    "momentum_1d": None,
    "momentum_7d": None,
    "last_update": None
}


def get_btc_features(price_fetcher=None) -> Dict:
    """
    Get current BTC features for correlation analysis.
    
    Args:
        price_fetcher: Optional function to fetch BTC price/indicators
    
    Returns:
        Dictionary of BTC features
    """
    global _BTC_CACHE
    
    # Check cache (refresh every 5 minutes)
    now = datetime.now()
    if _BTC_CACHE["last_update"]:
        age = (now - _BTC_CACHE["last_update"]).total_seconds()
        if age < 300:  # 5 minutes
            LOGGER.debug(f"Using cached BTC features (age: {age:.0f}s)")
            return {
                "BTC_PRICE": _BTC_CACHE["price"],
                "BTC_RSI": _BTC_CACHE["rsi"],
                "BTC_MACD_HISTOGRAM": _BTC_CACHE["macd"],
                "BTC_MOMENTUM_1D": _BTC_CACHE["momentum_1d"],
                "BTC_MOMENTUM_7D": _BTC_CACHE["momentum_7d"],
            }
    
    # Fetch fresh BTC data
    try:
        if price_fetcher:
            btc_data = price_fetcher("BTC")
        else:
            # Use default fetcher (if available)
            from wolf_app import turbo_crypto_price
            btc_data = turbo_crypto_price("BTC", max_budget_s=2.0)
        
        if not btc_data.get("ok") or not btc_data.get("price"):
            LOGGER.warning("Failed to fetch BTC data for correlation features")
            return {}
        
        price = float(btc_data["price"])
        
        # Calculate technical indicators (if we have historical data)
        indicators = btc_data.get("indicators", {})
        rsi = indicators.get("RSI_14", 50.0)  # Default neutral
        macd = indicators.get("MACD_HISTOGRAM", 0.0)
        
        # Calculate momentum (% change)
        historical = btc_data.get("historical", [])
        momentum_1d = 0.0
        momentum_7d = 0.0
        
        if historical and len(historical) >= 2:
            # 1-day momentum
            try:
                price_1d_ago = historical[-24] if len(historical) >= 24 else historical[0]
                momentum_1d = ((price - price_1d_ago) / price_1d_ago) * 100
            except (IndexError, ZeroDivisionError):
                pass
            
            # 7-day momentum
            try:
                price_7d_ago = historical[-168] if len(historical) >= 168 else historical[0]
                momentum_7d = ((price - price_7d_ago) / price_7d_ago) * 100
            except (IndexError, ZeroDivisionError):
                pass
        
        # Update cache
        _BTC_CACHE = {
            "price": price,
            "rsi": rsi,
            "macd": macd,
            "momentum_1d": momentum_1d,
            "momentum_7d": momentum_7d,
            "last_update": now
        }
        
        LOGGER.info(
            f"📊 BTC Features: ${price:,.2f}, RSI={rsi:.1f}, "
            f"1D={momentum_1d:+.2f}%, 7D={momentum_7d:+.2f}%"
        )
        
        return {
            "BTC_PRICE": price,
            "BTC_RSI": rsi,
            "BTC_MACD_HISTOGRAM": macd,
            "BTC_MOMENTUM_1D": momentum_1d,
            "BTC_MOMENTUM_7D": momentum_7d,
        }
    
    except Exception as e:
        LOGGER.error(f"Failed to get BTC features: {e}")
        return {}


def calculate_btc_correlation(symbol: str, symbol_prices: list, btc_prices: list) -> float:
    """
    Calculate correlation between symbol and BTC.
    
    Args:
        symbol: Trading symbol
        symbol_prices: List of symbol prices (recent to old)
        btc_prices: List of BTC prices (recent to old)
    
    Returns:
        Correlation coefficient (-1 to +1)
    """
    if not symbol_prices or not btc_prices:
        return 0.0
    
    # Align lengths
    min_len = min(len(symbol_prices), len(btc_prices))
    if min_len < 10:  # Need at least 10 points
        return 0.0
    
    symbol_prices = symbol_prices[:min_len]
    btc_prices = btc_prices[:min_len]
    
    # Calculate returns
    try:
        symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
        btc_returns = np.diff(btc_prices) / btc_prices[:-1]
        
        # Pearson correlation
        correlation = np.corrcoef(symbol_returns, btc_returns)[0, 1]
        
        if np.isnan(correlation):
            return 0.0
        
        return float(correlation)
    
    except Exception as e:
        LOGGER.debug(f"Correlation calculation failed for {symbol}: {e}")
        return 0.0


def detect_btc_lead(
    symbol: str,
    symbol_prices: list,
    btc_prices: list,
    max_lag_hours: int = 6
) -> Tuple[int, float]:
    """
    Detect if BTC leads the symbol movement.
    
    Args:
        symbol: Trading symbol
        symbol_prices: Hourly prices (recent to old)
        btc_prices: Hourly BTC prices (recent to old)
        max_lag_hours: Maximum lag to check
    
    Returns:
        (lead_hours, correlation) where lead_hours is how many hours BTC leads
    """
    if not symbol_prices or not btc_prices or len(symbol_prices) < 24:
        return 0, 0.0
    
    # Calculate returns
    try:
        symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
        btc_returns = np.diff(btc_prices) / btc_prices[:-1]
        
        best_lag = 0
        best_corr = 0.0
        
        # Test different lags
        for lag in range(1, min(max_lag_hours + 1, len(btc_returns) // 2)):
            # Shift BTC returns by lag hours
            if lag >= len(btc_returns):
                break
            
            btc_lagged = btc_returns[:-lag] if lag > 0 else btc_returns
            symbol_aligned = symbol_returns[lag:] if lag > 0 else symbol_returns
            
            min_len = min(len(btc_lagged), len(symbol_aligned))
            if min_len < 10:
                continue
            
            corr = np.corrcoef(
                btc_lagged[:min_len],
                symbol_aligned[:min_len]
            )[0, 1]
            
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_lag = lag
                best_corr = float(corr)
        
        return best_lag, best_corr
    
    except Exception as e:
        LOGGER.debug(f"Lead detection failed for {symbol}: {e}")
        return 0, 0.0


def calculate_btc_correlation_features(
    symbol: str,
    symbol_data: Dict,
    btc_features: Optional[Dict] = None
) -> Dict:
    """
    Calculate all BTC correlation features for a symbol.
    
    Args:
        symbol: Trading symbol
        symbol_data: Symbol price/indicator data
        btc_features: Optional pre-fetched BTC features
    
    Returns:
        Dictionary of correlation features
    """
    # Don't calculate for BTC itself
    if symbol.upper() == "BTC":
        return {}
    
    # Don't calculate for stocks (BTC correlation not relevant)
    if not _is_crypto_symbol(symbol):
        return {}
    
    # Get BTC features
    if not btc_features:
        btc_features = get_btc_features()
    
    if not btc_features:
        # BTC data unavailable - return neutral defaults
        return {
            "BTC_CORRELATION": 0.0,
            "BTC_LEAD_HOURS": 0,
            "BTC_MACD_BULLISH": 0.5,
            "BTC_LEADS": 0
        }
    
    features = {}
    
    # Add raw BTC features
    features.update(btc_features)
    
    # Calculate correlation (if we have historical data)
    symbol_prices = symbol_data.get("historical_prices", [])
    btc_prices = btc_features.get("historical_prices", [])
    
    if symbol_prices and btc_prices:
        correlation = calculate_btc_correlation(symbol, symbol_prices, btc_prices)
        features["BTC_CORRELATION"] = correlation
        
        # Detect lead/lag
        if abs(correlation) > 0.5:  # Only check lead if correlated
            lead_hours, lead_corr = detect_btc_lead(symbol, symbol_prices, btc_prices)
            features["BTC_LEAD_HOURS"] = lead_hours
            features["BTC_LEADS"] = 1 if lead_hours > 0 and lead_corr > 0.6 else 0
            
            if lead_hours > 0:
                LOGGER.info(
                    f"📊 {symbol} BTC correlation: {correlation:.2f}, "
                    f"BTC leads by {lead_hours}h (r={lead_corr:.2f})"
                )
    
    # Binary features based on BTC signals
    btc_rsi = btc_features.get("BTC_RSI", 50.0)
    btc_macd = btc_features.get("BTC_MACD_HISTOGRAM", 0.0)
    btc_momentum_1d = btc_features.get("BTC_MOMENTUM_1D", 0.0)
    
    # BTC bullish/bearish signals
    features["BTC_MACD_BULLISH"] = 1 if btc_macd > 0 else 0
    features["BTC_RSI_OVERSOLD"] = 1 if btc_rsi < 30 else 0
    features["BTC_RSI_OVERBOUGHT"] = 1 if btc_rsi > 70 else 0
    features["BTC_MOMENTUM_POSITIVE"] = 1 if btc_momentum_1d > 0 else 0
    
    return features


def _is_crypto_symbol(symbol: str) -> bool:
    """
    Check if symbol is a cryptocurrency.
    
    Args:
        symbol: Trading symbol
    
    Returns:
        True if crypto, False if stock
    """
    # Common crypto symbols
    crypto_symbols = {
        "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOT", "AVAX", "MATIC",
        "LINK", "UNI", "AAVE", "DOGE", "LTC", "BCH", "XLM", "ALGO", "VET",
        "ICP", "FIL", "ATOM", "NEAR", "APT", "OP", "ARB", "SUI", "TIA",
        "CHZ", "ZEC", "ILV", "RNDR", "RLC", "EGLD", "TURBO", "DASH", "FLOW",
        "OCEAN", "LRC", "CELO", "NMR", "SAND", "MANA", "ENJ", "GALA", "IMX"
    }
    
    symbol_upper = symbol.upper().strip()
    
    # Direct match
    if symbol_upper in crypto_symbols:
        return True
    
    # Heuristics: Most stocks are 1-4 letters, cryptos often longer
    # But this is imperfect (e.g., T, C, F are stocks)
    if len(symbol_upper) >= 4 and not symbol_upper.isdigit():
        return True
    
    # If in doubt, assume stock (safer default)
    return False


def enhance_features_with_btc_correlation(
    symbol: str,
    features: Dict,
    btc_features: Optional[Dict] = None
) -> Dict:
    """
    Add BTC correlation features to existing feature dictionary.
    
    Args:
        symbol: Trading symbol
        features: Existing feature dictionary
        btc_features: Optional pre-fetched BTC features
    
    Returns:
        Enhanced feature dictionary
    """
    # Skip if not crypto
    if not _is_crypto_symbol(symbol):
        return features
    
    # Calculate correlation features
    correlation_features = calculate_btc_correlation_features(
        symbol,
        symbol_data=features,  # Pass existing features (may contain historical prices)
        btc_features=btc_features
    )
    
    # Merge with existing features
    enhanced = {**features, **correlation_features}
    
    return enhanced


# Convenience function for direct import
def add_btc_features(symbol: str, features: Dict) -> Dict:
    """Add BTC correlation features to feature dictionary."""
    return enhance_features_with_btc_correlation(symbol, features)
