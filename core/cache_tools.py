"""
Ghost Cache Management Tools
Namespace-safe cache operations without FLUSHDB
"""

from datetime import datetime
from typing import Any

# Redis client (will be set by wolf_app.py)
REDIS_CLIENT = None
LOGGER = None


def purge_prices(symbols: list[str] | None = None) -> int:
    """
    Purge price caches for specific symbols

    Args:
        symbols: List of symbols to purge (None = all symbols)

    Returns:
        Number of keys deleted
    """
    if not REDIS_CLIENT:
        return 0

    try:
        deleted = 0

        if symbols:
            # Purge specific symbols
            for symbol in symbols:
                pattern = f"price:{symbol}:*"
                cursor = 0
                while True:
                    cursor, keys = REDIS_CLIENT.scan(cursor, match=pattern, count=100)
                    if keys:
                        REDIS_CLIENT.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
        else:
            # Purge all price keys
            pattern = "price:*"
            cursor = 0
            while True:
                cursor, keys = REDIS_CLIENT.scan(cursor, match=pattern, count=100)
                if keys:
                    REDIS_CLIENT.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

        if LOGGER:
            LOGGER.info(f"Purged {deleted} price cache keys")

        return deleted

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Price purge error: {e}")
        return 0


def purge_alert_dedup(older_than_days: int = 7) -> int:
    """
    Clean up old alert deduplication keys

    Args:
        older_than_days: Remove keys older than this many days

    Returns:
        Number of keys deleted
    """
    if not REDIS_CLIENT:
        return 0

    try:
        deleted = 0
        today = datetime.now()

        # Scan for alert dedup keys
        pattern = "alerts:sent:*"
        cursor = 0

        while True:
            cursor, keys = REDIS_CLIENT.scan(cursor, match=pattern, count=100)
            for key in keys:
                # Parse key format: alerts:sent:{market}:{symbol}:{horizon}:{date}
                parts = key.decode() if isinstance(key, bytes) else key
                parts = parts.split(":")

                if len(parts) >= 6:
                    date_str = parts[5]
                    try:
                        key_date = datetime.strptime(date_str, "%Y-%m-%d")
                        age_days = (today - key_date).days

                        if age_days > older_than_days:
                            REDIS_CLIENT.delete(key)
                            deleted += 1
                    except ValueError:
                        # Invalid date format - delete it
                        REDIS_CLIENT.delete(key)
                        deleted += 1

            if cursor == 0:
                break

        if LOGGER:
            LOGGER.info(f"Purged {deleted} old alert dedup keys (>{older_than_days}d)")

        return deleted

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Alert dedup purge error: {e}")
        return 0


def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics

    Returns:
        Dict with counts of different key types
    """
    if not REDIS_CLIENT:
        return {"error": "Redis not available"}

    try:
        stats = {
            "total_keys": REDIS_CLIENT.dbsize(),
            "price_keys": 0,
            "alert_keys": 0,
            "other_keys": 0,
        }

        # Count price keys
        cursor = 0
        while True:
            cursor, keys = REDIS_CLIENT.scan(cursor, match="price:*", count=100)
            stats["price_keys"] += len(keys)
            if cursor == 0:
                break

        # Count alert keys
        cursor = 0
        while True:
            cursor, keys = REDIS_CLIENT.scan(cursor, match="alerts:*", count=100)
            stats["alert_keys"] += len(keys)
            if cursor == 0:
                break

        # Calculate other keys
        stats["other_keys"] = stats["total_keys"] - stats["price_keys"] - stats["alert_keys"]

        return stats

    except Exception as e:
        if LOGGER:
            LOGGER.error(f"Cache stats error: {e}")
        return {"error": str(e)}
