"""
GHOST Crypto Prediction Module
Parallel to Stock Module, Shared AI Core
"""

__version__ = "0.1.0"
__author__ = "GHOST AI"

from .crypto_predictor import CryptoPredictionEngine
from .crypto_providers import (
    BinanceProvider,
    CoinbaseProvider,
    CoinGeckoProvider,
    get_crypto_price_quorum,
)

__all__ = [
    "CoinGeckoProvider",
    "BinanceProvider",
    "CoinbaseProvider",
    "get_crypto_price_quorum",
    "CryptoPredictionEngine",
]
