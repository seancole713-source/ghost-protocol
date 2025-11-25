"""
Redis Caching Utilities
========================
80% API call reduction via intelligent caching.

TTL Strategy:
- Spot Prices: 90s (stocks), 30s (crypto) - high frequency updates
- OHLCV Bars: 5min (intraday), 60min (daily) - stable historical data
- Indicators: 60s - pre-computed technical signals
- Provider Health: 30s - status monitoring

Cache Keys:
- ghost:price:spot:{SYMBOL}:v1
- ghost:ohlcv:{SYMBOL}:{INTERVAL}:v1
- ghost:indicators:{SYMBOL}:v1
- ghost:provider:health:{PROVIDER}:v1
"""

import json
import time
from typing import Any, Callable, Optional
import redis
import os
import logging

LOGGER = logging.getLogger(__name__)


class CacheUtils:
    """Redis caching with TTL and JSON serialization"""
    
    def __init__(self):
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            LOGGER.warning("REDIS_URL not set - caching disabled")
            self.redis_client = None
        else:
            try:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                LOGGER.info("✅ Redis connected for caching")
            except Exception as e:
                LOGGER.error(f"❌ Redis connection failed: {e}")
                self.redis_client = None
    
    def get_cached_json(
        self,
        key: str,
        ttl: int,
        compute_fn: Callable[[], Any],
        force_refresh: bool = False
    ) -> Any:
        """
        Get cached JSON or compute + cache if missing.
        
        Args:
            key: Redis key (e.g., "ghost:price:spot:AAPL:v1")
            ttl: Time-to-live in seconds
            compute_fn: Function to compute value if cache miss
            force_refresh: Bypass cache and recompute
        
        Returns:
            Cached or computed value
        """
        # No Redis = direct compute
        if not self.redis_client:
            return compute_fn()
        
        # Force refresh bypasses cache
        if force_refresh:
            value = compute_fn()
            self._set_cached_json(key, value, ttl)
            return value
        
        # Try cache first
        try:
            cached = self.redis_client.get(key)
            if cached:
                LOGGER.debug(f"✅ Cache HIT: {key}")
                return json.loads(cached)
        except Exception as e:
            LOGGER.warning(f"Cache read failed for {key}: {e}")
        
        # Cache miss - compute and store
        LOGGER.debug(f"❌ Cache MISS: {key}")
        value = compute_fn()
        
        if value is not None:
            self._set_cached_json(key, value, ttl)
        
        return value
    
    def _set_cached_json(self, key: str, value: Any, ttl: int) -> None:
        """Store JSON value in Redis with TTL"""
        if not self.redis_client:
            return
        
        try:
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            LOGGER.debug(f"💾 Cached {key} (TTL={ttl}s)")
        except Exception as e:
            LOGGER.warning(f"Cache write failed for {key}: {e}")
    
    def invalidate(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "ghost:price:*")
        
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                LOGGER.info(f"🗑️  Invalidated {deleted} keys matching {pattern}")
                return deleted
            return 0
        except Exception as e:
            LOGGER.error(f"Cache invalidation failed for {pattern}: {e}")
            return 0
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            {
                "total_keys": int,
                "memory_used_mb": float,
                "hit_rate": float,
                "uptime_seconds": int
            }
        """
        if not self.redis_client:
            return {"error": "Redis not connected"}
        
        try:
            info = self.redis_client.info()
            stats = self.redis_client.info("stats")
            
            total_keys = self.redis_client.dbsize()
            memory_used = info.get("used_memory", 0) / 1024 / 1024  # MB
            
            hits = stats.get("keyspace_hits", 0)
            misses = stats.get("keyspace_misses", 0)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
            
            return {
                "total_keys": total_keys,
                "memory_used_mb": round(memory_used, 2),
                "hit_rate": round(hit_rate, 3),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
                "keyspace_hits": hits,
                "keyspace_misses": misses
            }
        except Exception as e:
            LOGGER.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}


# Global cache instance
_cache_utils: Optional[CacheUtils] = None


def get_cache() -> CacheUtils:
    """Get or create global cache instance"""
    global _cache_utils
    if _cache_utils is None:
        _cache_utils = CacheUtils()
    return _cache_utils


# Convenience functions
def cache_spot_price(symbol: str, compute_fn: Callable, ttl: Optional[int] = None) -> Any:
    """Cache spot price with appropriate TTL (90s stocks, 30s crypto)"""
    # Detect crypto vs stock by symbol pattern
    is_crypto = len(symbol) <= 5 and not any(c.isdigit() for c in symbol)
    default_ttl = 30 if is_crypto else 90
    
    key = f"ghost:price:spot:{symbol.upper()}:v1"
    return get_cache().get_cached_json(key, ttl or default_ttl, compute_fn)


def cache_ohlcv(symbol: str, interval: str, compute_fn: Callable) -> Any:
    """Cache OHLCV with appropriate TTL (5min intraday, 60min daily)"""
    # Intraday = 5min cache, Daily = 60min cache
    ttl = 300 if interval in ["1m", "5m", "15m"] else 3600
    
    key = f"ghost:ohlcv:{symbol.upper()}:{interval}:v1"
    return get_cache().get_cached_json(key, ttl, compute_fn)


def cache_indicators(symbol: str, compute_fn: Callable) -> Any:
    """Cache pre-computed technical indicators (60s TTL)"""
    key = f"ghost:indicators:{symbol.upper()}:v1"
    return get_cache().get_cached_json(key, 60, compute_fn)


def invalidate_symbol(symbol: str) -> int:
    """Invalidate all cached data for a symbol"""
    return get_cache().invalidate(f"ghost:*:{symbol.upper()}:*")


def get_cache_stats() -> dict[str, Any]:
    """Get cache statistics for monitoring"""
    return get_cache().get_stats()


if __name__ == "__main__":
    # Test cache
    cache = get_cache()
    
    def expensive_compute():
        time.sleep(1)
        return {"price": 150.25, "timestamp": int(time.time())}
    
    # First call - cache miss
    start = time.time()
    result1 = cache.get_cached_json("test:key:v1", 30, expensive_compute)
    miss_time = time.time() - start
    print(f"Cache MISS: {miss_time:.3f}s - {result1}")
    
    # Second call - cache hit
    start = time.time()
    result2 = cache.get_cached_json("test:key:v1", 30, expensive_compute)
    hit_time = time.time() - start
    print(f"Cache HIT: {hit_time:.3f}s - {result2}")
    
    # Stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
