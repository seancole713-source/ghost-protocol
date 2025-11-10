"""
GHOST In-Memory Caching Module
Provides LRU caching with TTL support for frequently accessed data.
Zero external dependencies - completely free!
"""

import functools
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from threading import RLock
from typing import Any, TypeVar

T = TypeVar("T")


class TTLCache:
    """Thread-safe TTL (Time-To-Live) cache with LRU eviction."""

    def __init__(self, maxsize: int = 128, ttl: float = 300.0):
        """
        Initialize TTL cache.

        Args:
            maxsize: Maximum number of entries (default 128)
            ttl: Time-to-live in seconds (default 300 = 5 minutes)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[Any, float, float | None]] = OrderedDict()
        self.lock = RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            value, timestamp, ttl_override = self.cache[key]

            # Check if expired (use per-entry TTL if provided)
            effective_ttl = ttl_override if ttl_override is not None else self.ttl
            if time.time() - timestamp > effective_ttl:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set value in cache with current timestamp and optional per-entry TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional per-entry TTL override (default: use cache instance TTL)
        """
        with self.lock:
            # Remove oldest if at capacity
            if key not in self.cache and len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)

            self.cache[key] = (value, time.time(), ttl)
            self.cache.move_to_end(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def info(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "ttl": self.ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }


# Global cache instances for different data types
PRICE_CACHE = TTLCache(maxsize=256, ttl=60.0)  # 1 minute for prices
MARKET_DATA_CACHE = TTLCache(maxsize=128, ttl=300.0)  # 5 minutes for market data
API_RESPONSE_CACHE = TTLCache(maxsize=512, ttl=180.0)  # 3 minutes for API responses
FORECAST_CACHE = TTLCache(maxsize=64, ttl=900.0)  # 15 minutes for forecasts


def cached(
    cache: TTLCache | None = None, ttl: float | None = None, key_func: Callable | None = None
):
    """
    Decorator to cache function results with TTL support.

    Args:
        cache: TTLCache instance to use (default: API_RESPONSE_CACHE)
        ttl: Override TTL for this function (optional)
        key_func: Custom function to generate cache key from args/kwargs

    Example:
        @cached(cache=PRICE_CACHE, ttl=30.0)
        def get_price(symbol: str) -> float:
            return expensive_api_call(symbol)
    """
    if cache is None:
        cache = API_RESPONSE_CACHE

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _default_key(func.__name__, args, kwargs)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result

        # Expose cache control methods
        wrapper.cache_clear = cache.clear  # type: ignore
        wrapper.cache_info = cache.info  # type: ignore

        return wrapper

    return decorator


def async_cached(
    cache: TTLCache | None = None, ttl: float | None = None, key_func: Callable | None = None
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to cache async function results with TTL support.

    Example:
        @async_cached(cache=PRICE_CACHE, ttl=30.0)
        async def fetch_price(symbol: str) -> float:
            return await expensive_async_api_call(symbol)
    """
    if cache is None:
        cache = API_RESPONSE_CACHE

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _default_key(func.__name__, args, kwargs)

            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result

            # Call async function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result

        # Expose cache control methods
        wrapper.cache_clear = cache.clear  # type: ignore
        wrapper.cache_info = cache.info  # type: ignore

        return wrapper

    return decorator


def _default_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate default cache key from function name and arguments."""
    key_parts = [func_name]

    # Add positional args
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(str(hash(str(arg))))

    # Add keyword args (sorted for consistency)
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, (str, int, float, bool)):
            key_parts.append(f"{k}={v}")
        else:
            key_parts.append(f"{k}={hash(str(v))}")

    return ":".join(key_parts)


def get_all_cache_stats() -> dict:
    """Get statistics for all cache instances."""
    return {
        "price_cache": PRICE_CACHE.info(),
        "market_data_cache": MARKET_DATA_CACHE.info(),
        "api_response_cache": API_RESPONSE_CACHE.info(),
        "forecast_cache": FORECAST_CACHE.info(),
    }


def clear_all_caches() -> None:
    """Clear all cache instances."""
    PRICE_CACHE.clear()
    MARKET_DATA_CACHE.clear()
    API_RESPONSE_CACHE.clear()
    FORECAST_CACHE.clear()


# Convenience function for manual caching
def cache_get(key: str, cache_type: str = "api") -> Any | None:
    """
    Get value from specified cache.

    Args:
        key: Cache key
        cache_type: 'price', 'market', 'api', or 'forecast'
    """
    cache_map = {
        "price": PRICE_CACHE,
        "market": MARKET_DATA_CACHE,
        "api": API_RESPONSE_CACHE,
        "forecast": FORECAST_CACHE,
    }
    cache = cache_map.get(cache_type, API_RESPONSE_CACHE)
    return cache.get(key)


def cache_set(key: str, value: Any, cache_type: str = "api") -> None:
    """
    Set value in specified cache.

    Args:
        key: Cache key
        value: Value to cache
        cache_type: 'price', 'market', 'api', or 'forecast'
    """
    cache_map = {
        "price": PRICE_CACHE,
        "market": MARKET_DATA_CACHE,
        "api": API_RESPONSE_CACHE,
        "forecast": FORECAST_CACHE,
    }
    cache = cache_map.get(cache_type, API_RESPONSE_CACHE)
    cache.set(key, value)
