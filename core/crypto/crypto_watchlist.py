"""
Crypto Watchlist Manager
Handles dynamic crypto watchlist with persistence
"""

import json
from pathlib import Path

# Default watchlist (hardcoded fallback)
DEFAULT_WATCHLIST = [
    "BTC",
    "ETH",
    "SOL",
    "DOGE",
    "SHIB",
    "PEPE",
    "ADA",
    "DOT",
    "MATIC",
    "AVAX",
    "LINK",
    "UNI",
    "ATOM",
    "XRP",
    "LTC",
]

# Watchlist file path
WATCHLIST_FILE = Path(__file__).parent.parent.parent / "data" / "crypto_watchlist.json"

# In-memory cache (session-level persistence)
_WATCHLIST_CACHE: list[str] = None


def get_crypto_watchlist() -> list[str]:
    """
    Get current crypto watchlist.

    Priority:
    1. In-memory cache (session)
    2. Persistent file (if exists)
    3. Default hardcoded list

    Returns:
        List of crypto symbols (uppercase)
    """
    global _WATCHLIST_CACHE

    # Return cached watchlist if available
    if _WATCHLIST_CACHE is not None:
        return _WATCHLIST_CACHE

    # Try to load from persistent file
    if WATCHLIST_FILE.exists():
        try:
            with open(WATCHLIST_FILE) as f:
                data = json.load(f)
                watchlist = data.get("watchlist", [])
                if watchlist:
                    _WATCHLIST_CACHE = [s.upper() for s in watchlist]
                    print(
                        f"[WATCHLIST] Loaded {len(_WATCHLIST_CACHE)} symbols from {WATCHLIST_FILE}"
                    )
                    return _WATCHLIST_CACHE
        except Exception as e:
            print(f"[WATCHLIST] Error loading from file: {e}")

    # Fall back to default
    _WATCHLIST_CACHE = DEFAULT_WATCHLIST.copy()
    print(f"[WATCHLIST] Using default watchlist ({len(_WATCHLIST_CACHE)} symbols)")

    # Try to save default to file for future use
    _save_watchlist(_WATCHLIST_CACHE)

    return _WATCHLIST_CACHE


def add_to_watchlist(symbol: str) -> bool:
    """
    Add symbol to watchlist.

    Args:
        symbol: Crypto symbol (e.g., "BTC", "PEPE")

    Returns:
        True if added, False if already exists
    """
    symbol = symbol.upper()
    watchlist = get_crypto_watchlist()

    if symbol in watchlist:
        return False

    watchlist.append(symbol)
    _save_watchlist(watchlist)
    print(f"[WATCHLIST] Added {symbol} (total: {len(watchlist)})")

    return True


def remove_from_watchlist(symbol: str) -> bool:
    """
    Remove symbol from watchlist.

    Args:
        symbol: Crypto symbol (e.g., "BTC")

    Returns:
        True if removed, False if not found
    """
    symbol = symbol.upper()
    watchlist = get_crypto_watchlist()

    if symbol not in watchlist:
        return False

    watchlist.remove(symbol)
    _save_watchlist(watchlist)
    print(f"[WATCHLIST] Removed {symbol} (total: {len(watchlist)})")

    return True


def _save_watchlist(watchlist: list[str]):
    """
    Save watchlist to persistent file.

    Args:
        watchlist: List of symbols to save
    """
    global _WATCHLIST_CACHE

    # Update cache
    _WATCHLIST_CACHE = watchlist

    # Save to file
    try:
        WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "watchlist": watchlist,
            "count": len(watchlist),
            "last_updated": __import__("time").time(),
        }

        with open(WATCHLIST_FILE, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[WATCHLIST] Saved {len(watchlist)} symbols to {WATCHLIST_FILE}")
    except Exception as e:
        print(f"[WATCHLIST] Error saving to file: {e}")


def reset_watchlist():
    """Reset watchlist to default (for testing/debugging)"""
    global _WATCHLIST_CACHE
    _WATCHLIST_CACHE = DEFAULT_WATCHLIST.copy()
    _save_watchlist(_WATCHLIST_CACHE)
    print(f"[WATCHLIST] Reset to default ({len(_WATCHLIST_CACHE)} symbols)")


def is_in_watchlist(symbol: str) -> bool:
    """Check if symbol is in watchlist"""
    return symbol.upper() in get_crypto_watchlist()


# Auto-initialize on import
get_crypto_watchlist()
