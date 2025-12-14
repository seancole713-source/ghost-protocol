"""Asset classification helpers.

Keep this minimal and dependency-free.
"""

from __future__ import annotations


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


def is_crypto_symbol(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_SYMBOLS
