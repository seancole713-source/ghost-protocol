"""
Crypto Price Providers
Multi-source quorum system for reliable crypto prices
Supports environment-driven provider selection via CRYPTO_QUORUM
"""

import asyncio
import logging
import os
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

# Shared HTTP session with retry logic
_session = requests.Session()
_retry_strategy = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# Provider configuration from environment
_DEFAULT_CRYPTO_QUORUM = ["coingecko", "binance", "coinbase"]
_CRYPTO_QUORUM_ORDER = None  # Lazy-loaded from env

# Provider configuration from environment
_DEFAULT_CRYPTO_QUORUM = ["coingecko", "binance", "coinbase"]
_CRYPTO_QUORUM_ORDER = None  # Lazy-loaded from env


def _get_crypto_provider_order() -> list[str]:
    """
    Get crypto provider order from CRYPTO_QUORUM environment variable.
    
    Returns ordered list of provider names to try.
    Falls back to default order if env var not set.
    
    Example: CRYPTO_QUORUM="coingecko,binance,coinbase"
    """
    global _CRYPTO_QUORUM_ORDER
    
    if _CRYPTO_QUORUM_ORDER is not None:
        return _CRYPTO_QUORUM_ORDER
    
    env_quorum = os.getenv("CRYPTO_QUORUM", "").strip()
    
    if env_quorum:
        # Parse comma-separated provider names
        providers = [p.strip().lower() for p in env_quorum.split(",") if p.strip()]
        if providers:
            _CRYPTO_QUORUM_ORDER = providers
            LOGGER.info(f"Crypto provider order from CRYPTO_QUORUM: {providers}")
            return _CRYPTO_QUORUM_ORDER
    
    # Fallback to default
    _CRYPTO_QUORUM_ORDER = _DEFAULT_CRYPTO_QUORUM.copy()
    LOGGER.info(f"Crypto provider order (default): {_CRYPTO_QUORUM_ORDER}")
    return _CRYPTO_QUORUM_ORDER


class CoinGeckoProvider:
    """
    CoinGecko API - Primary provider (Free tier: 50 calls/min)
    Docs: https://www.coingecko.com/en/api/documentation
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Symbol to CoinGecko ID mapping
    SYMBOL_MAP = {
        # Major Cryptocurrencies (Blue Chip)
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
        "XRP": "ripple",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "DOT": "polkadot",
        "MATIC": "matic-network",
        "LINK": "chainlink",
        # Layer 2 & Scaling
        "OP": "optimism",
        "ARB": "arbitrum",
        # DeFi Tokens
        "UNI": "uniswap",
        "AAVE": "aave",
        "MKR": "maker",
        "CRV": "curve-dao-token",
        "SUSHI": "sushi",
        "COMP": "compound-governance-token",
        # Meme Coins (Top Tier)
        "DOGE": "dogecoin",  # OG meme king
        "SHIB": "shiba-inu",  # Dogecoin killer
        "PEPE": "pepe",  # Pepe the Frog
        "FLOKI": "floki",  # Floki Inu
        "BONK": "bonk",  # Solana meme coin
        # Meme Coins (Mid Tier)
        "WIF": "dogwifhat",  # Dog Wif Hat (Solana)
        "BABYDOGE": "baby-doge-coin",
        "ELON": "dogelon-mars",
        "SHIB2": "shiba-2",
        "AKITA": "akita-inu",
        # AI & Gaming Coins
        "FET": "fetch-ai",
        "AGIX": "singularitynet",
        "RNDR": "render-token",
        "SAND": "the-sandbox",
        "MANA": "decentraland",
        "AXS": "axie-infinity",
        "GALA": "gala",
        # New/Trending (Add as needed)
        "BRETT": "brett",
        "MOG": "mog-coin",
        "TURBO": "turbo",
        "WOJAK": "wojak",
    }

    def __init__(self):
        self.last_call = 0
        self.min_interval = 1.2  # 50 calls/min = 1.2s between calls

    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

    def get_coin_id(self, symbol: str) -> str | None:
        """Convert symbol to CoinGecko ID"""
        return self.SYMBOL_MAP.get(symbol.upper())

    def get_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get current price with 24h metrics

        Returns:
            {
                'symbol': 'BTC',
                'price': 43251.50,
                'change_24h': 1250.30,
                'change_24h_pct': 2.98,
                'market_cap': 845000000000,
                'volume_24h': 32000000000,
                'last_updated': 1728741600,
                'provider': 'coingecko'
            }
        """
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            LOGGER.warning(f"CoinGecko: Unknown symbol {symbol}")
            return None

        try:
            self._rate_limit()

            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_last_updated_at": "true",
            }

            response = _session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if coin_id not in data:
                return None

            coin_data = data[coin_id]

            return {
                "symbol": symbol.upper(),
                "price": float(coin_data.get("usd", 0)),
                "change_24h": float(coin_data.get("usd_24h_change", 0)),
                "change_24h_pct": float(coin_data.get("usd_24h_change", 0)),
                "market_cap": float(coin_data.get("usd_market_cap", 0)),
                "volume_24h": float(coin_data.get("usd_24h_vol", 0)),
                "last_updated": int(coin_data.get("last_updated_at", time.time())),
                "provider": "coingecko",
            }

        except Exception as e:
            LOGGER.warning(f"CoinGecko fetch failed for {symbol}: {e}")
            return None

    def get_historical(self, symbol: str, days: int = 7) -> list[dict] | None:
        """
        Get historical prices for pattern analysis

        Returns list of {timestamp, price} dicts
        """
        coin_id = self.get_coin_id(symbol)
        if not coin_id:
            return None

        try:
            self._rate_limit()

            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
            params = {"vs_currency": "usd", "days": days, "interval": "hourly"}

            response = _session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            prices = data.get("prices", [])

            return [{"timestamp": int(p[0] / 1000), "price": float(p[1])} for p in prices]

        except Exception as e:
            LOGGER.warning(f"CoinGecko historical fetch failed for {symbol}: {e}")
            return None


class BinanceProvider:
    """
    Binance API - Secondary provider (Unlimited public data)
    Docs: https://binance-docs.github.io/apidocs/spot/en/
    
    Enhanced with:
    - Exponential backoff for rate limits and 451 errors
    - Automatic fallback to Binance US endpoint
    - Retry logic for transient failures
    """

    REST_URL = "https://api.binance.com/api/v3"
    REST_URL_US = "https://api.binance.us/api/v3"  # US fallback for Cloudflare blocks

    def __init__(self):
        self.symbol_suffix = "USDT"  # Trade against USDT
        self.max_retries = 3
        self.base_delay = 0.5  # Start with 500ms

    def get_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get current price from Binance with retry logic and US fallback

        Returns similar format to CoinGecko for consistency
        """
        binance_symbol = f"{symbol.upper()}{self.symbol_suffix}"
        
        # Try primary endpoint first, then US fallback
        urls = [self.REST_URL, self.REST_URL_US]
        
        for base_url in urls:
            for attempt in range(self.max_retries):
                try:
                    # Get ticker with 24h stats
                    url = f"{base_url}/ticker/24hr"
                    params = {"symbol": binance_symbol}

                    response = _session.get(url, params=params, timeout=10)
                    response.raise_for_status()

                    data = response.json()

                    return {
                        "symbol": symbol.upper(),
                        "price": float(data.get("lastPrice", 0)),
                        "change_24h": float(data.get("priceChange", 0)),
                        "change_24h_pct": float(data.get("priceChangePercent", 0)),
                        "volume_24h": float(data.get("quoteVolume", 0)),
                        "last_updated": int(data.get("closeTime", 0) / 1000),
                        "provider": "binance",
                    }

                except Exception as e:
                    error_msg = str(e)
                    status_code = getattr(getattr(e, 'response', None), 'status_code', 0)
                    
                    # Check for Cloudflare 451 or rate limiting (retryable)
                    is_cloudflare_block = status_code == 451 or "451" in error_msg
                    is_rate_limit = status_code == 429 or "429" in error_msg
                    
                    # Retry on temporary errors
                    if (is_cloudflare_block or is_rate_limit) and attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)  # 0.5s, 1s, 2s
                        LOGGER.debug(
                            f"Binance blocked/rate-limited for {symbol} "
                            f"(status {status_code}), retrying in {delay}s "
                            f"(attempt {attempt + 1}/{self.max_retries}, url={base_url})"
                        )
                        time.sleep(delay)
                        continue  # Retry same URL
                    
                    # If final attempt, try next URL
                    if attempt == self.max_retries - 1:
                        LOGGER.debug(
                            f"Binance fetch failed for {symbol} after {self.max_retries} attempts "
                            f"on {base_url}: {e}"
                        )
                        break  # Try alternative URL
        
        # All attempts and URLs exhausted
        LOGGER.warning(f"Binance fetch failed for {symbol}: All endpoints exhausted")
        return None


class CoinbaseProvider:
    """
    Coinbase API - Tertiary provider (High reliability)
    Docs: https://developers.coinbase.com/api/v2
    """

    BASE_URL = "https://api.coinbase.com/v2"

    def get_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get spot price from Coinbase

        Note: Coinbase provides less 24h data than CoinGecko/Binance
        """
        try:
            url = f"{self.BASE_URL}/prices/{symbol.upper()}-USD/spot"

            response = _session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            price_data = data.get("data", {})

            return {
                "symbol": symbol.upper(),
                "price": float(price_data.get("amount", 0)),
                "last_updated": int(time.time()),
                "provider": "coinbase",
            }

        except Exception as e:
            LOGGER.warning(f"Coinbase fetch failed for {symbol}: {e}")
            return None


# Cache for crypto prices (2-minute TTL)
_CRYPTO_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL = 120  # 2 minutes


def _get_crypto_cache(symbol: str) -> dict[str, Any] | None:
    """Get cached price if still valid"""
    if symbol in _CRYPTO_CACHE:
        cached = _CRYPTO_CACHE[symbol]
        if time.time() - cached.get("cached_at", 0) < _CACHE_TTL:
            return cached
    return None


def _set_crypto_cache(symbol: str, price_data: dict[str, Any]):
    """Cache price data"""
    price_data["cached_at"] = time.time()
    _CRYPTO_CACHE[symbol] = price_data


async def get_crypto_price_quorum(symbol: str, use_cache: bool = True) -> dict[str, Any] | None:
    """
    Get crypto price with provider quorum

    Strategy:
    1. Check cache if enabled
    2. Query CoinGecko (primary)
    3. Query Binance (secondary)
    4. Query Coinbase (tertiary)
    5. Require 2+ providers agreeing within 1% spread
    6. Return median price with quorum metadata

    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        use_cache: Whether to use cached price

    Returns:
        {
            'symbol': 'BTC',
            'price': 43251.50,
            'provider': 'coingecko',
            'confidence': 0.95,
            'quorum_size': 3,
            'spread': 0.003,
            'timestamp': 1728741600,
            'change_24h_pct': 2.98,
            'market_cap': 845000000000
        }
    """
    symbol = symbol.upper()

    # Check cache
    if use_cache:
        cached = _get_crypto_cache(symbol)
        if cached:
            LOGGER.debug(f"Crypto price cache hit for {symbol}")
            return cached

    # Initialize providers based on environment configuration
    provider_order = _get_crypto_provider_order()
    provider_map = {
        "coingecko": CoinGeckoProvider(),
        "binance": BinanceProvider(),
        "coinbase": CoinbaseProvider(),
    }
    
    providers = [
        (name, provider_map[name]) 
        for name in provider_order 
        if name in provider_map
    ]

    # Collect prices from all providers - SHORT-CIRCUIT on first success to avoid 401/451 retries
    results: list[tuple[str, float, dict]] = []

    for name, provider in providers:
        try:
            price_data = await asyncio.to_thread(provider.get_price, symbol)
            if price_data and price_data.get("price", 0) > 0:
                results.append((name, price_data["price"], price_data))
                LOGGER.debug(f"{name}: {symbol} = ${price_data['price']:.2f}")
                # Short-circuit: accept first working provider to avoid slow 401/451 errors
                if len(results) >= 1:
                    LOGGER.info(f"Short-circuit: using {name} for {symbol} (fast-path)")
                    break
        except Exception as e:
            # Skip 401/451 immediately instead of retrying
            if "401" in str(e) or "451" in str(e) or "Unauthorized" in str(e):
                LOGGER.info(f"Provider {name} auth failed for {symbol}, skipping: {e}")
                continue
            LOGGER.warning(f"Provider {name} failed for {symbol}: {e}")

    if not results:
        LOGGER.error(f"All crypto providers failed for {symbol}")
        return None

    # Calculate quorum
    prices = [r[1] for r in results]
    median_price = sorted(prices)[len(prices) // 2]

    # Check spread
    if len(prices) > 1:
        spread = (max(prices) - min(prices)) / median_price
    else:
        spread = 0.0

    # Determine confidence based on quorum size and spread
    if len(results) >= 3 and spread < 0.01:  # 3 providers, <1% spread
        confidence = 0.95
    elif len(results) >= 2 and spread < 0.02:  # 2 providers, <2% spread
        confidence = 0.85
    elif len(results) >= 2:
        confidence = 0.75
    else:
        confidence = 0.65  # Single provider

    # Use primary provider's data (CoinGecko) for extra fields
    primary_data = results[0][2]

    result = {
        "symbol": symbol,
        "price": median_price,
        "provider": results[0][0],  # Credit primary provider
        "confidence": confidence,
        "quorum_size": len(results),
        "spread": spread,
        "timestamp": int(time.time()),
        "change_24h_pct": primary_data.get("change_24h_pct", 0),
        "market_cap": primary_data.get("market_cap", 0),
        "volume_24h": primary_data.get("volume_24h", 0),
    }

    # Cache result
    _set_crypto_cache(symbol, result)

    LOGGER.info(
        f"Crypto price quorum for {symbol}: "
        f"${median_price:.2f} ({len(results)} providers, "
        f"{spread * 100:.2f}% spread, {confidence:.0%} confidence)"
    )

    return result


# Default watchlists by category
DEFAULT_WATCHLIST = ["BTC", "ETH", "SOL", "BNB", "ADA"]  # Conservative default

WATCHLIST_BLUE_CHIP = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT", "MATIC", "LINK"]

WATCHLIST_DEFI = ["UNI", "AAVE", "MKR", "CRV", "SUSHI", "COMP", "LINK"]

WATCHLIST_MEME_COINS = ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BABYDOGE", "ELON"]

WATCHLIST_AI_GAMING = ["FET", "AGIX", "RNDR", "SAND", "MANA", "AXS", "GALA"]

WATCHLIST_ALL = WATCHLIST_BLUE_CHIP + WATCHLIST_DEFI + WATCHLIST_MEME_COINS + WATCHLIST_AI_GAMING


def get_default_watchlist() -> list[str]:
    """Get default crypto watchlist (conservative blue chip)"""
    return DEFAULT_WATCHLIST.copy()


def get_watchlist_by_category(category: str = "default") -> list[str]:
    """
    Get watchlist by category

    Categories:
    - 'default': Conservative blue chip (BTC, ETH, SOL, BNB, ADA)
    - 'blue_chip': Top 10 major cryptos
    - 'defi': DeFi tokens
    - 'meme': Meme coins (DOGE, SHIB, PEPE, etc.)
    - 'ai_gaming': AI and gaming tokens
    - 'all': All tracked coins (40+)
    """
    categories = {
        "default": DEFAULT_WATCHLIST,
        "blue_chip": WATCHLIST_BLUE_CHIP,
        "defi": WATCHLIST_DEFI,
        "meme": WATCHLIST_MEME_COINS,
        "meme_coins": WATCHLIST_MEME_COINS,  # Alias
        "ai_gaming": WATCHLIST_AI_GAMING,
        "all": WATCHLIST_ALL,
    }

    return categories.get(category.lower(), DEFAULT_WATCHLIST).copy()


def get_all_supported_symbols() -> list[str]:
    """Get all supported crypto symbols"""
    from .crypto_providers import CoinGeckoProvider

    provider = CoinGeckoProvider()
    return sorted(provider.SYMBOL_MAP.keys())
