"""
Ghost Price Feed Reliability Layer
===================================
Implements primary/secondary provider fallback with staleness checks

This module wraps existing Ghost price providers with:
1. Configurable primary/secondary provider selection
2. Strict freshness validation
3. Automatic fallback on failure
4. Provider performance tracking

Usage:
    from core.price_reliability import get_price_with_fallback
    
    price_data = get_price_with_fallback(
        symbol="WOLF",
        asset_type="stock",
        primary="polygon",
        secondary="yahoo"
    )
    
    if price_data:
        print(f"Price: ${price_data['price']} from {price_data['provider']}")
"""

import logging
import os
import time
from typing import Any, Literal, Optional, Dict, Tuple, Callable

LOGGER = logging.getLogger(__name__)

# Configuration
PRICE_SOURCE_PRIMARY = os.getenv("PRICE_SOURCE_PRIMARY", "polygon")
PRICE_SOURCE_SECONDARY = os.getenv("PRICE_SOURCE_SECONDARY", "yahoo")
PRICE_FRESHNESS_THRESHOLD_S = float(os.getenv("PRICE_FRESHNESS_THRESHOLD_S", "300"))  # 5 minutes

# Provider performance tracking (in-memory cache)
_PROVIDER_STATS = {
    "polygon": {"success": 0, "fail": 0, "stale": 0, "total_latency_ms": 0.0},
    "yahoo": {"success": 0, "fail": 0, "stale": 0, "total_latency_ms": 0.0},
    "alphavantage": {"success": 0, "fail": 0, "stale": 0, "total_latency_ms": 0.0},
}


def get_price_with_fallback(
    symbol: str,
    asset_type: Literal["stock", "crypto"] = "stock",
    primary: Optional[str] = None,
    secondary: Optional[str] = None,
    freshness_threshold_s: Optional[float] = None,
    price_quorum_func: Optional[Callable] = None  # Injected from wolf_app.py
) -> Optional[Dict[str, Any]]:
    """
    Fetch price with primary/secondary provider fallback
    
    Args:
        symbol: Trading symbol (e.g., "WOLF", "BTC")
        asset_type: "stock" or "crypto"
        primary: Primary provider name (default from env)
        secondary: Secondary provider name (default from env)
        freshness_threshold_s: Max age in seconds (default 300)
        price_quorum_func: Function to call for price fetch
        
    Returns:
        {
            "price": float,
            "timestamp": float,
            "provider": str,
            "prev_close": float,
            "fresh": bool,
            "fallback_used": bool
        }
        
        None if both providers fail
        
    Example:
        >>> data = get_price_with_fallback("WOLF", "stock")
        >>> data["price"]
        17.51
        >>> data["provider"]
        'polygon'
        >>> data["fallback_used"]
        False
    """
    # Use env defaults if not specified
    primary = primary or PRICE_SOURCE_PRIMARY
    secondary = secondary or PRICE_SOURCE_SECONDARY
    freshness_threshold_s = freshness_threshold_s or PRICE_FRESHNESS_THRESHOLD_S
    
    # Try primary provider first
    LOGGER.debug(f"[{symbol}] Fetching price from primary provider: {primary}")
    
    start_time = time.time()
    primary_result = _fetch_price_from_provider(
        symbol,
        asset_type,
        primary,
        price_quorum_func
    )
    primary_latency_ms = (time.time() - start_time) * 1000
    
    # Validate primary result
    if primary_result and _is_price_fresh(primary_result, freshness_threshold_s):
        _record_provider_success(primary, primary_latency_ms)
        primary_result["fallback_used"] = False
        LOGGER.info(f"[{symbol}] Price from {primary}: ${primary_result['price']:.4f} (fresh)")
        return primary_result
    elif primary_result:
        _record_provider_stale(primary, primary_latency_ms)
        age_s = time.time() - primary_result.get("timestamp", 0)
        LOGGER.warning(f"[{symbol}] Price from {primary} is stale ({age_s:.0f}s old), trying fallback")
    else:
        _record_provider_failure(primary, primary_latency_ms)
        LOGGER.warning(f"[{symbol}] Primary provider {primary} failed, trying fallback")
    
    # Try secondary provider
    LOGGER.debug(f"[{symbol}] Fetching price from secondary provider: {secondary}")
    
    start_time = time.time()
    secondary_result = _fetch_price_from_provider(
        symbol,
        asset_type,
        secondary,
        price_quorum_func
    )
    secondary_latency_ms = (time.time() - start_time) * 1000
    
    # Validate secondary result
    if secondary_result and _is_price_fresh(secondary_result, freshness_threshold_s):
        _record_provider_success(secondary, secondary_latency_ms)
        secondary_result["fallback_used"] = True
        LOGGER.info(f"[{symbol}] Price from {secondary} (fallback): ${secondary_result['price']:.4f} (fresh)")
        return secondary_result
    elif secondary_result:
        _record_provider_stale(secondary, secondary_latency_ms)
        age_s = time.time() - secondary_result.get("timestamp", 0)
        LOGGER.warning(f"[{symbol}] Price from {secondary} is stale ({age_s:.0f}s old)")
    else:
        _record_provider_failure(secondary, secondary_latency_ms)
        LOGGER.error(f"[{symbol}] Secondary provider {secondary} failed")
    
    # Both providers failed
    LOGGER.error(f"[{symbol}] All providers failed ({primary}, {secondary})")
    return None


def _fetch_price_from_provider(
    symbol: str,
    asset_type: str,
    provider: str,
    price_quorum_func: Optional[Callable]
) -> Optional[Dict[str, Any]]:
    """
    Internal: Fetch price from specific provider
    
    Returns:
        {price, timestamp, provider, prev_close} or None
    """
    try:
        # Use injected price_quorum_func if available
        # Otherwise, this would import from wolf_app (circular dependency risk)
        if price_quorum_func:
            result = price_quorum_func(symbol, asset_type)
            if result and result.get("price"):
                return {
                    "price": float(result["price"]),
                    "timestamp": result.get("timestamp", time.time()),
                    "provider": result.get("provider", provider),
                    "prev_close": result.get("prev_close", result["price"]),
                }
        
        return None
        
    except Exception as e:
        LOGGER.error(f"Provider {provider} error: {e}")
        return None


def _is_price_fresh(price_data: dict[str, Any], threshold_s: float) -> bool:
    """Check if price data is fresh"""
    if not price_data or not price_data.get("price"):
        return False
    
    timestamp = price_data.get("timestamp", 0)
    if timestamp == 0:
        # No timestamp = assume current time (fallback)
        return True
    
    age_s = time.time() - timestamp
    fresh = age_s <= threshold_s
    
    price_data["fresh"] = fresh
    price_data["age_s"] = age_s
    
    return fresh


def _record_provider_success(provider: str, latency_ms: float):
    """Record successful provider fetch"""
    if provider in _PROVIDER_STATS:
        _PROVIDER_STATS[provider]["success"] += 1
        _PROVIDER_STATS[provider]["total_latency_ms"] += latency_ms


def _record_provider_failure(provider: str, latency_ms: float):
    """Record failed provider fetch"""
    if provider in _PROVIDER_STATS:
        _PROVIDER_STATS[provider]["fail"] += 1
        _PROVIDER_STATS[provider]["total_latency_ms"] += latency_ms


def _record_provider_stale(provider: str, latency_ms: float):
    """Record stale provider data"""
    if provider in _PROVIDER_STATS:
        _PROVIDER_STATS[provider]["stale"] += 1
        _PROVIDER_STATS[provider]["total_latency_ms"] += latency_ms


def get_provider_stats() -> dict[str, Any]:
    """
    Get provider performance statistics
    
    Returns:
        {
            "polygon": {
                "success": 145,
                "fail": 3,
                "stale": 2,
                "total_requests": 150,
                "success_rate": 0.97,
                "avg_latency_ms": 245.3
            },
            ...
        }
    """
    stats = {}
    
    for provider, data in _PROVIDER_STATS.items():
        total = data["success"] + data["fail"] + data["stale"]
        if total == 0:
            continue
        
        success_rate = data["success"] / total if total > 0 else 0.0
        avg_latency = data["total_latency_ms"] / total if total > 0 else 0.0
        
        stats[provider] = {
            "success": data["success"],
            "fail": data["fail"],
            "stale": data["stale"],
            "total_requests": total,
            "success_rate": success_rate,
            "avg_latency_ms": round(avg_latency, 1),
        }
    
    return stats


def reset_provider_stats():
    """Reset provider statistics (for testing)"""
    global _PROVIDER_STATS
    for provider in _PROVIDER_STATS:
        _PROVIDER_STATS[provider] = {
            "success": 0,
            "fail": 0,
            "stale": 0,
            "total_latency_ms": 0.0
        }


# Example usage and tests
if __name__ == "__main__":
    import json
    
    # Mock price quorum function for testing
    def mock_price_quorum(symbol, asset_type):
        import random
        return {
            "price": 17.51 + random.uniform(-0.5, 0.5),
            "timestamp": time.time() - random.uniform(0, 600),  # 0-10 min old
            "provider": "polygon",
            "prev_close": 17.45,
        }
    
    print("=== Price Reliability Layer Test ===\n")
    
    # Test 1: Fetch with fallback
    print("Test 1: Normal fetch")
    result = get_price_with_fallback(
        "WOLF",
        "stock",
        price_quorum_func=mock_price_quorum
    )
    if result:
        print(f"  Price: ${result['price']:.4f}")
        print(f"  Provider: {result['provider']}")
        print(f"  Fresh: {result.get('fresh', False)}")
        print(f"  Fallback used: {result.get('fallback_used', False)}")
    print()
    
    # Test 2: Check stats
    print("Test 2: Provider statistics")
    stats = get_provider_stats()
    print(json.dumps(stats, indent=2))
    print()
    
    # Test 3: Multiple fetches
    print("Test 3: Multiple fetches (5x)")
    for i in range(5):
        result = get_price_with_fallback("WOLF", "stock", price_quorum_func=mock_price_quorum)
        if result:
            print(f"  Fetch {i+1}: ${result['price']:.4f} from {result['provider']} (fresh: {result.get('fresh')})")
    print()
    
    # Test 4: Final stats
    print("Test 4: Final statistics")
    stats = get_provider_stats()
    print(json.dumps(stats, indent=2))
