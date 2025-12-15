"""Asset classification helpers.

Keep this minimal and dependency-free.

Design goals:
- Single source of truth for crypto-vs-stock routing.
- Runtime extensibility so higher-level apps (e.g., `wolf_app.py`) can register
    their active crypto universe without introducing circular imports.
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


def is_crypto_symbol(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_SYMBOLS
