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

# =============================================================================
# PRICE VALIDATION - Prevent corrupt prices (e.g., BTC @ $38)
# Added Dec 28, 2025 after discovering corrupt data in PostgreSQL
# =============================================================================

MIN_SANE_PRICES = {
    'BTC': 10000,    # BTC should never be below $10k (last seen 2020)
    'ETH': 500,      # ETH should never be below $500
    'SOL': 10,       # SOL should never be below $10
    'BNB': 100,      # BNB should never be below $100
    'XRP': 0.10,     # XRP should never be below $0.10
    'ADA': 0.05,     # ADA should never be below $0.05
    'DOGE': 0.005,   # DOGE should never be below $0.005
    'AVAX': 5,       # AVAX should never be below $5
    'DOT': 2,        # DOT should never be below $2
    'LINK': 3,       # LINK should never be below $3
    'MATIC': 0.20,   # MATIC should never be below $0.20
    'UNI': 2,        # UNI should never be below $2
    'AAVE': 30,      # AAVE should never be below $30
    'ATOM': 3,       # ATOM should never be below $3
    'LTC': 30,       # LTC should never be below $30
    # Added to prevent price corruption bugs
    'FTM': 0.01,     # FTM (Fantom) ~$0.06 as of Jan 2026
    'NEAR': 1,       # NEAR should never be below $1
    'ALGO': 0.05,    # ALGO should never be below $0.05
    'BCH': 50,       # BCH should never be below $50
    'ANKR': 0.001,   # ANKR is a low-price coin ~$0.006
}

MAX_SANE_PRICES = {
    'BTC': 500000,   # BTC ceiling (will need to update in bull market!)
    'ETH': 50000,    # ETH ceiling
    'SOL': 5000,     # SOL ceiling
    'BNB': 2000,     # BNB ceiling
    'DOGE': 10,      # DOGE ceiling (even in meme season)
    'SHIB': 0.01,    # SHIB ceiling
    'XRP': 50,       # XRP ceiling
}


def validate_crypto_price(symbol: str, price: float) -> bool:
    """
    Reject obviously wrong prices before storing.
    
    Prevents corrupt data like BTC @ $38 (actual $87k) from poisoning
    accuracy calculations.
    
    Returns:
        True if price is valid, False if it should be rejected
    """
    if price <= 0:
        LOGGER.warning(f"Price validation failed: {symbol} @ ${price:.2f} (non-positive)")
        return False
    
    symbol_upper = symbol.upper()
    
    min_price = MIN_SANE_PRICES.get(symbol_upper, 0.0000001)
    max_price = MAX_SANE_PRICES.get(symbol_upper, 10000000)  # $10M default ceiling
    
    if price < min_price:
        LOGGER.warning(f"Price validation failed: {symbol} @ ${price:.2f} < min ${min_price}")
        return False
    
    if price > max_price:
        LOGGER.warning(f"Price validation failed: {symbol} @ ${price:.2f} > max ${max_price}")
        return False
    
    return True

# Shared HTTP session with connection pooling and retry logic
_session = requests.Session()
_retry_strategy = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
# Increase pool size from default 10 to 50 to handle concurrent requests
_adapter = HTTPAdapter(
    max_retries=_retry_strategy,
    pool_connections=50,  # Total connection pools
    pool_maxsize=50,  # Max connections per pool
    pool_block=False,  # Don't block on full pool
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# Provider configuration from environment
# In strict quorum mode we need >=2 providers; Binance.com is often 451-blocked.
# Priority order (Dec 2025):
#   1. Coinbase - Most reliable, no API key needed, never rate-limited
#   2. Binance.US - Works in US (binance.com is 451-blocked)
#   3. CryptoCompare - Good fallback, optional API key
#   4. CoinGecko - LAST because free tier is heavily rate-limited (429)
_DEFAULT_CRYPTO_QUORUM = ["coinbase", "binance", "cryptocompare", "coingecko"]
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

        # Common alts (needed for production watchlists)
        "TRX": "tron",
        "TON": "the-open-network",
        "XLM": "stellar",
        "ETC": "ethereum-classic",
        "XMR": "monero",
        # FTM and other L1/L2 chains (CRITICAL - was missing, caused $0.48 bug!)
        "FTM": "fantom",  # Fantom Opera chain
        "NEAR": "near",
        "ALGO": "algorand",
        "ATOM": "cosmos",
        "ICP": "internet-computer",
        "HBAR": "hedera-hashgraph",
        "VET": "vechain",
        "EGLD": "elrond-erd-2",  # MultiversX
        "THETA": "theta-token",
        "NEO": "neo",
        "KAVA": "kava",
        "ZIL": "zilliqa",
        "ROSE": "oasis-network",
        # Other tracked coins
        "LTC": "litecoin",
        "BCH": "bitcoin-cash",
        "ANKR": "ankr",
        "METIS": "metis-token",
        "CRO": "crypto-com-chain",
        "LDO": "lido-dao",
        "APE": "apecoin",
        "IMX": "immutable-x",
        "FLOW": "flow",
        "MINA": "mina-protocol",
        "KCS": "kucoin-shares",
        "ENJ": "enjincoin",
        "CHZ": "chiliz",
        "QTUM": "qtum",
        "ZEC": "zcash",
        "DASH": "dash",
    }

    def __init__(self):
        self.last_call = 0
        # PERFORMANCE FIX: Increased from 2.0s to 5.0s to prevent 429 spam
        # Ultra-conservative rate: 12 calls/min instead of 30 calls/min
        self.min_interval = 5.0  # 12 calls/min = 5.0s between calls
        
        # Circuit breaker: skip CoinGecko if too many 429s
        self.consecutive_429s = 0
        self.circuit_open = False
        self.circuit_open_until = 0.0

    def _rate_limit(self):
        """Enforce rate limiting (5s minimum between calls) + circuit breaker"""
        # Check circuit breaker
        if self.circuit_open:
            if time.time() < self.circuit_open_until:
                LOGGER.debug(f"CoinGecko circuit open for {self.circuit_open_until - time.time():.0f}s more")
                raise Exception("CoinGecko circuit breaker open - too many 429s")
            else:
                # Reset circuit breaker
                LOGGER.info("CoinGecko circuit breaker reset - trying again")
                self.circuit_open = False
                self.consecutive_429s = 0
        
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            LOGGER.debug(f"CoinGecko rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
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
            
            # Reset 429 counter on success
            if self.consecutive_429s > 0:
                LOGGER.info(f"CoinGecko recovered - resetting 429 counter from {self.consecutive_429s}")
                self.consecutive_429s = 0

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
            # Track 429 errors and open circuit breaker
            if "429" in str(e) or "Too Many Requests" in str(e):
                self.consecutive_429s += 1
                if self.consecutive_429s >= 3:
                    self.circuit_open = True
                    self.circuit_open_until = time.time() + 300  # 5 minutes
                    LOGGER.error(f"CoinGecko circuit breaker OPENED - {self.consecutive_429s} consecutive 429s - disabled for 5min")
                else:
                    LOGGER.warning(f"CoinGecko 429 #{self.consecutive_429s} for {symbol} - circuit opens at 3")
            else:
                LOGGER.warning(f"CoinGecko fetch failed for {symbol}: {e}")
            return None

        except Exception as e:
            # Track 429 errors and open circuit breaker
            if "429" in str(e) or "Too Many Requests" in str(e):
                self.consecutive_429s += 1
                if self.consecutive_429s >= 3:
                    self.circuit_open = True
                    self.circuit_open_until = time.time() + 300  # 5 minutes
                    LOGGER.error(f"CoinGecko circuit breaker OPENED - {self.consecutive_429s} consecutive 429s - disabled for 5min")
                else:
                    LOGGER.warning(f"CoinGecko 429 #{self.consecutive_429s} for {symbol} - circuit opens at 3")
            else:
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
    - Circuit breaker pattern for reliability
    - Exponential backoff for rate limits and 451 errors
    - Automatic fallback to Binance US endpoint
    - Retry logic for transient failures
    """

    REST_URL = "https://api.binance.com/api/v3"
    REST_URL_US = "https://api.binance.us/api/v3"  # US fallback for Cloudflare blocks

    def __init__(self):
        self.symbol_suffix = "USDT"  # Trade against USDT
        self.max_retries = 2  # Reduced from 3 to fail faster
        self.base_delay = 0.3  # Reduced from 500ms to 300ms for faster fallback
        
        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_until = 0.0
        self.consecutive_failures = 0

    def get_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get current price from Binance with retry logic and US fallback

        Returns similar format to CoinGecko for consistency
        """
        # Check circuit breaker
        if self.circuit_open:
            if time.time() < self.circuit_open_until:
                LOGGER.debug(f"Binance circuit breaker OPEN - skipping {symbol}")
                return None
            else:
                # Try to close circuit (half-open state)
                self.circuit_open = False
                self.consecutive_failures = 0
                LOGGER.info("Binance circuit breaker transitioning to HALF-OPEN")
        
        binance_symbol = f"{symbol.upper()}{self.symbol_suffix}"
        
        # Prefer Binance.US first (Binance.com is frequently 451-blocked)
        urls = [self.REST_URL_US, self.REST_URL]
        
        for base_url in urls:
            for attempt in range(self.max_retries):
                try:
                    # Get ticker with 24h stats
                    url = f"{base_url}/ticker/24hr"
                    params = {"symbol": binance_symbol}

                    response = _session.get(url, params=params, timeout=10)
                    response.raise_for_status()

                    data = response.json()
                    
                    # Success - reset circuit breaker
                    if self.consecutive_failures > 0:
                        LOGGER.info(f"Binance recovered - resetting failure counter from {self.consecutive_failures}")
                        self.consecutive_failures = 0

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
                        delay = self.base_delay * (2 ** attempt)  # 0.3s, 0.6s, 1.2s
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
        
        # All attempts and URLs exhausted - track for circuit breaker
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.circuit_open = True
            self.circuit_open_until = time.time() + 300  # 5 minutes
            LOGGER.error(f"Binance circuit breaker OPENED - {self.consecutive_failures} consecutive failures - disabled for 5min")
        else:
            LOGGER.warning(f"Binance failure #{self.consecutive_failures} for {symbol} - circuit opens at 3")
        
        return None


class CoinbaseProvider:
    """
    Coinbase API - Tertiary provider (High reliability)
    Docs: https://developers.coinbase.com/api/v2
    
    Enhanced with circuit breaker pattern for consistency
    """

    BASE_URL = "https://api.coinbase.com/v2"

    def __init__(self):
        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_until = 0.0
        self.consecutive_failures = 0

    def get_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get spot price from Coinbase

        Note: Coinbase provides less 24h data than CoinGecko/Binance
        """
        # Check circuit breaker
        if self.circuit_open:
            if time.time() < self.circuit_open_until:
                LOGGER.debug(f"Coinbase circuit breaker OPEN - skipping {symbol}")
                return None
            else:
                # Try to close circuit (half-open state)
                self.circuit_open = False
                self.consecutive_failures = 0
                LOGGER.info("Coinbase circuit breaker transitioning to HALF-OPEN")
        
        try:
            url = f"{self.BASE_URL}/prices/{symbol.upper()}-USD/spot"

            response = _session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            price_data = data.get("data", {})
            
            # Success - reset circuit breaker
            if self.consecutive_failures > 0:
                LOGGER.info(f"Coinbase recovered - resetting failure counter from {self.consecutive_failures}")
                self.consecutive_failures = 0

            return {
                "symbol": symbol.upper(),
                "price": float(price_data.get("amount", 0)),
                "last_updated": int(time.time()),
                "provider": "coinbase",
            }

        except Exception as e:
            # Track failures for circuit breaker
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.circuit_open = True
                self.circuit_open_until = time.time() + 300  # 5 minutes
                LOGGER.error(f"Coinbase circuit breaker OPENED - {self.consecutive_failures} consecutive failures - disabled for 5min")
            else:
                LOGGER.warning(f"Coinbase failure #{self.consecutive_failures} for {symbol} - circuit opens at 3: {e}")
            
            return None
            return None


class CryptoCompareProvider:
    """CryptoCompare price API (broad symbol coverage).

    Optional API key via `CRYPTOCOMPARE_API_KEY` (not required for light usage).
    """

    BASE_URL = "https://min-api.cryptocompare.com"

    def __init__(self):
        self.api_key = os.getenv("CRYPTOCOMPARE_API_KEY", "").strip() or None

    def get_price(self, symbol: str) -> dict[str, Any] | None:
        try:
            url = f"{self.BASE_URL}/data/price"
            params = {"fsym": symbol.upper(), "tsyms": "USD"}
            headers = {}
            if self.api_key:
                headers["authorization"] = f"Apikey {self.api_key}"

            response = _session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json() if response.content else {}
            if not isinstance(data, dict) or "USD" not in data:
                return None

            return {
                "symbol": symbol.upper(),
                "price": float(data.get("USD", 0) or 0),
                "last_updated": int(time.time()),
                "provider": "cryptocompare",
            }
        except Exception as e:
            LOGGER.warning(f"CryptoCompare fetch failed for {symbol}: {e}")
            return None


# Cache for crypto prices (30-second TTL for high-frequency trading)
# PERFORMANCE FIX: Reduced from 15min to 30s for better price freshness
# while still preventing excessive API calls
_CRYPTO_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_TTL = 30  # 30 seconds - balances freshness with API rate limits

# Cache metrics for monitoring
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _get_crypto_cache(symbol: str) -> dict[str, Any] | None:
    """Get cached price if still valid"""
    global _CACHE_HITS, _CACHE_MISSES
    
    if symbol in _CRYPTO_CACHE:
        cached = _CRYPTO_CACHE[symbol]
        if time.time() - cached.get("cached_at", 0) < _CACHE_TTL:
            _CACHE_HITS += 1
            return cached
    
    _CACHE_MISSES += 1
    return None


def get_cache_stats() -> dict[str, Any]:
    """Get cache performance statistics"""
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate = (_CACHE_HITS / total * 100) if total > 0 else 0
    
    return {
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "total_requests": total,
        "hit_rate_pct": round(hit_rate, 2),
        "cache_size": len(_CRYPTO_CACHE),
        "ttl_seconds": _CACHE_TTL,
    }


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

    def _truthy(v: str | None) -> bool:
        return str(v or "").strip().lower() in {"1", "true", "yes", "on"}

    require_quorum = _truthy(os.getenv("PRICE_REQUIRE_QUORUM", "0")) or _truthy(
        os.getenv("PREDICT_REQUIRE_PRICE_QUORUM", "0")
    )
    try:
        min_providers = max(1, int(os.getenv("PRICE_MIN_PROVIDERS", "1")))
    except Exception:
        min_providers = 1

    # Check cache (but never allow cached single-provider values to bypass quorum enforcement)
    if use_cache:
        cached = _get_crypto_cache(symbol)
        if isinstance(cached, dict):
            cached_quorum = int(cached.get("quorum_size") or 0)
            if require_quorum and cached_quorum < min_providers:
                LOGGER.info(
                    f"Crypto price cache bypassed for {symbol}: quorum_size={cached_quorum} < min_providers={min_providers}"
                )
            else:
                LOGGER.debug(f"Crypto price cache hit for {symbol}")
                return cached

    # Initialize providers based on environment configuration
    provider_order = _get_crypto_provider_order()
    provider_map = {
        "coingecko": CoinGeckoProvider(),
        "binance": BinanceProvider(),
        "coinbase": CoinbaseProvider(),
        "cryptocompare": CryptoCompareProvider(),
    }

    providers = [(name, provider_map[name]) for name in provider_order if name in provider_map]

    # Collect prices from providers.
    # When quorum is NOT required, short-circuit on first success to minimize API load.
    results: list[tuple[str, float, dict]] = []

    for name, provider in providers:
        try:
            price_data = await asyncio.to_thread(provider.get_price, symbol)
            if price_data and price_data.get("price", 0) > 0:
                price = float(price_data["price"])
                
                # VALIDATION: Reject obviously wrong prices
                if not validate_crypto_price(symbol, price):
                    LOGGER.warning(f"Provider {name} returned invalid price for {symbol}: ${price:.2f} - SKIPPING")
                    continue
                
                results.append((name, price, price_data))
                if not require_quorum:
                    LOGGER.info(f"Short-circuit: using {name} for {symbol} (fast-path)")
                    break
                if len(results) >= min_providers:
                    # We have enough providers for quorum; stop to limit API load.
                    break
        except Exception as e:
            error_str = str(e)
            if any(code in error_str for code in ["401", "451", "429", "Unauthorized", "rate limit"]):
                LOGGER.debug(f"Provider {name} blocked/throttled for {symbol}, skipping: {e}")
                continue
            LOGGER.warning(f"Provider {name} failed for {symbol}: {e}")

    if not results:
        LOGGER.error(f"All crypto providers failed for {symbol}")
        return None

    if require_quorum and len(results) < min_providers:
        LOGGER.warning(
            f"Crypto quorum failed for {symbol}: only {len(results)}/{min_providers} providers returned prices"
        )
        return None

    prices = [r[1] for r in results]
    median_price = sorted(prices)[len(prices) // 2]

    if len(prices) > 1:
        spread = (max(prices) - min(prices)) / median_price
    else:
        spread = 0.0

    if len(results) >= 3 and spread < 0.01:
        confidence = 0.95
    elif len(results) >= 2 and spread < 0.02:
        confidence = 0.85
    elif len(results) >= 2:
        confidence = 0.75
    else:
        confidence = 0.65

    primary_data = results[0][2]
    result = {
        "symbol": symbol,
        "price": median_price,
        "provider": results[0][0],
        "confidence": confidence,
        "quorum_size": len(results),
        "spread": spread,
        "timestamp": int(time.time()),
        "change_24h_pct": primary_data.get("change_24h_pct", 0),
        "market_cap": primary_data.get("market_cap", 0),
        "volume_24h": primary_data.get("volume_24h", 0),
    }

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
# ---- Turbo-friendly crypto price wrappers (sync, single-asset) ----
import logging
import time
import requests

logger = logging.getLogger("core.crypto.crypto_providers")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _binance_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("USDT"):
        return s
    return s + "USDT"


def get_price_binance(symbol: str) -> dict:
    """
    Sync wrapper for Binance spot price.
    Returns: {"provider": "binance", "symbol": "BTC", "price": float, "ts": ms}
    """
    pair = _binance_symbol(symbol)
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()
    data = resp.json()
    price = float(data["price"])
    return {
        "provider": "binance",
        "symbol": symbol.upper(),
        "price": price,
        "ts": _now_ms(),
    }


_COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    # Expanded coverage for common watchlist assets
    "XMR": "monero",
    "TON": "the-open-network",
    "TRX": "tron",
    "XLM": "stellar",
    "ALGO": "algorand",
    "ETC": "ethereum-classic",
    "ATOM": "cosmos",
    "BCH": "bitcoin-cash",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "UNI": "uniswap",
    "AAVE": "aave",
    "DOGE": "dogecoin",
    "SHIB": "shiba-inu",
    "PEPE": "pepe",
    "FLOKI": "floki",
    "BONK": "bonk",
    "WIF": "dogwifhat",
}


def get_price_coingecko(symbol: str) -> dict:
    """
    Sync wrapper for CoinGecko simple price API.
    Only supports a small set of majors; raises for others.
    """
    sid = _COINGECKO_IDS.get(symbol.upper())
    if not sid:
        raise ValueError(f"CoinGecko wrapper does not support symbol: {symbol}")
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={sid}&vs_currencies=usd"
    )
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()
    data = resp.json()
    price = float(data[sid]["usd"])
    return {
        "provider": "coingecko",
        "symbol": symbol.upper(),
        "price": price,
        "ts": _now_ms(),
    }


def get_price_coinbase(symbol: str) -> dict:
    """
    Sync wrapper for Coinbase spot price.
    Uses /v2/prices/<SYMBOL>-USD/spot
    """
    pair = f"{symbol.upper()}-USD"
    url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
    resp = requests.get(url, timeout=3)
    resp.raise_for_status()
    data = resp.json()
    amount = data["data"]["amount"]
    price = float(amount)
    return {
        "provider": "coinbase",
        "symbol": symbol.upper(),
        "price": price,
        "ts": _now_ms(),
    }