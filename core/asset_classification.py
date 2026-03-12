"""Asset classification helpers.

Keep this minimal and dependency-free.

Design goals:
- Single source of truth for crypto-vs-stock routing.
- Runtime extensibility so higher-level apps (e.g., `wolf_app.py`) can register
    their active crypto universe without introducing circular imports.
- FIX (Mar 12, 2026): Merge with config/symbols.py CRYPTO_SYMBOLS at import time
    so there's ONE authoritative crypto list. Previously 33 crypto symbols (including
    CHZ, ILV, TURBO, ZEC) were missing here, causing them to be classified as "stocks"
    in the Telegram picks pipeline and stealing slots from actual stocks.
"""

from __future__ import annotations

import os
from typing import Iterable


# Curated crypto universe used by free-tier providers in this repo.
# This is intentionally conservative: we only mark symbols as crypto when
# we expect our crypto providers (Binance/CoinGecko/Coinbase) to support them.
CRYPTO_SYMBOLS: set[str] = {
    # Majors
    "BTC",
    "ETH",
    "BNB",
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "DOT",
    "MATIC",
    "LINK",
    "UNI",
    "LTC",
    "BCH",
    "ATOM",
    "NEAR",
    "APT",
    "ARB",
    "OP",
    "FIL",
    # Common alts seen in production logs
    "TRX",
    "TON",
    "XLM",
    "ETC",
    "XMR",
    # Meme
    "SHIB",
    "PEPE",
    "FLOKI",
    "BONK",
    "WIF",
    "BABYDOGE",
    "ELON",
    # AI / DeFi / Gaming
    "RNDR",
    "FET",
    "AGIX",
    "OCEAN",
    "AAVE",
    "MKR",
    "SNX",
    "CRV",
    "SUSHI",
    "COMP",
    "IMX",
    "SAND",
    "MANA",
    "AXS",
    "GALA",
}


def register_crypto_symbols(symbols: Iterable[str] | None) -> int:
    """Register additional crypto symbols at runtime.

    Returns the number of new symbols added.
    """
    if not symbols:
        return 0
    before = len(CRYPTO_SYMBOLS)
    for sym in symbols:
        if sym is None:
            continue
        s = str(sym).strip().upper()
        if s:
            CRYPTO_SYMBOLS.add(s)
    return len(CRYPTO_SYMBOLS) - before


def _load_env_extras() -> None:
    extras = os.getenv("CRYPTO_SYMBOLS_EXTRA", "")
    if not extras.strip():
        return
    register_crypto_symbols([s for s in extras.split(",")])


_load_env_extras()


def _merge_config_crypto() -> None:
    """Merge crypto symbols from config/symbols.py into this module's set.

    This ensures a single authoritative crypto universe — any symbol marked
    as crypto in config/symbols.py is also recognized here.  Without this,
    symbols like CHZ, ILV, TURBO etc. were classified as 'stock' in the
    Telegram picks pipeline, crowding out actual stocks.
    """
    try:
        from config.symbols import CRYPTO_SYMBOLS as _config_crypto
        added = register_crypto_symbols(_config_crypto)
        if added:
            import logging
            logging.getLogger("ghost").info(
                f"[ASSET_CLASSIFICATION] Merged {added} crypto symbols from config/symbols.py "
                f"(total: {len(CRYPTO_SYMBOLS)})"
            )
    except ImportError:
        pass  # config.symbols not available — keep local list


_merge_config_crypto()


def is_crypto_symbol(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_SYMBOLS
