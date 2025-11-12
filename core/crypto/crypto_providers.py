"""
Crypto Price Providers
Multi-source quorum system for reliable crypto prices
"""

import json
import logging
import os
import time
from pathlib import Path
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

# VIP Contract Map - Load once at module init with env override support
_CONTRACT_MAP: dict[str, dict] = {}
_CONTRACT_MAP_FILE = Path(os.getenv("VIP_CONTRACT_MAP_PATH", str(Path(__file__).parent / "contract_map.json")))
_CONTRACT_MAP_MTIME = 0.0


def _load_contract_map():
    """Load contract map with file mtime watcher for hot-reload"""
    global _CONTRACT_MAP, _CONTRACT_MAP_MTIME

    try:
        if not _CONTRACT_MAP_FILE.exists():
            LOGGER.warning(f"Contract map not found: {_CONTRACT_MAP_FILE}")
            return

        current_mtime = _CONTRACT_MAP_FILE.stat().st_mtime

        # Only reload if file changed
        if current_mtime <= _CONTRACT_MAP_MTIME and _CONTRACT_MAP:
            return

        with open(_CONTRACT_MAP_FILE) as f:
            raw_map = json.load(f)

        # Normalize: uppercase keys, validate addresses
        _CONTRACT_MAP = {}
        for symbol, data in raw_map.items():
            symbol_upper = symbol.upper().strip()
            address = data.get("address", "").strip()

            # Fail closed on invalid addresses
            if not address or address.startswith("0x...") or len(address) < 20:
                LOGGER.error(f"Invalid contract address for {symbol_upper}: {address}")
                continue

            _CONTRACT_MAP[symbol_upper] = {
                "chain": data.get("chain", "ethereum").lower(),
                "address": address.lower() if address.startswith("0x") else address,
                "decimals": data.get("decimals", 18),
                "name": data.get("name", symbol_upper),
                "coingecko_id": data.get("coingecko_id", "")
            }

        _CONTRACT_MAP_MTIME = current_mtime
        LOGGER.info(f"✅ Loaded contract map: {len(_CONTRACT_MAP)} VIP tokens: {list(_CONTRACT_MAP.keys())}")

    except Exception as e:
        LOGGER.error(f"Failed to load contract map: {e}")
        _CONTRACT_MAP = {}


# Load contract map on module import
_load_contract_map()


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

    def get_price_by_contract(self, chain: str, address: str, symbol: str) -> dict[str, Any] | None:
        """
        Get price using contract address (VIP path for mapped tokens)

        Args:
            chain: Chain name (ethereum, solana, etc.)
            address: Contract address
            symbol: Token symbol for response

        Returns:
            Same format as get_price() but via contract endpoint
        """
        try:
            self._rate_limit()

            # Map chain names to CoinGecko platform IDs
            chain_map = {
                "ethereum": "ethereum",
                "bsc": "binance-smart-chain",
                "polygon": "polygon-pos",
                "solana": "solana",
                "avalanche": "avalanche",
                "arbitrum": "arbitrum-one",
                "optimism": "optimistic-ethereum"
            }

            platform_id = chain_map.get(chain.lower(), "ethereum")
            url = f"{self.BASE_URL}/coins/{platform_id}/contract/{address}"

            response = _session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            market_data = data.get("market_data", {})

            if not market_data:
                LOGGER.warning(f"No market data for contract {address} on {chain}")
                return None

            current_price = market_data.get("current_price", {}).get("usd", 0)
            if current_price <= 0:
                return None

            price_change_24h = market_data.get("price_change_percentage_24h", 0)

            LOGGER.info(f"✅ Contract price for {symbol}: ${current_price:.6f} (source=contract)")

            return {
                "symbol": symbol.upper(),
                "price": float(current_price),
                "change_24h_pct": float(price_change_24h),
                "market_cap": float(market_data.get("market_cap", {}).get("usd", 0)),
                "volume_24h": float(market_data.get("total_volume", {}).get("usd", 0)),
                "last_updated": int(time.time()),
                "provider": "coingecko",
                "price_source": "contract"  # Debug marker
            }

        except Exception as e:
            LOGGER.warning(f"CoinGecko contract fetch failed for {symbol} ({address}): {e}")
            return None

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
    """

    REST_URL = "https://api.binance.com/api/v3"

    def __init__(self):
        self.symbol_suffix = "USDT"  # Trade against USDT

    def get_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Get current price from Binance

        Returns similar format to CoinGecko for consistency
        """
        try:
            binance_symbol = f"{symbol.upper()}{self.symbol_suffix}"

            # Get ticker with 24h stats
            url = f"{self.REST_URL}/ticker/24hr"
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
            LOGGER.warning(f"Binance fetch failed for {symbol}: {e}")
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
    2. Check if symbol has contract mapping (VIP path)
    3. Try contract-based lookup first (CoinGecko contract endpoint)
    4. Fallback to symbol-based lookups
    5. Short-circuit on first success

    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        use_cache: Whether to use cached price

    Returns:
        {
            'symbol': 'BTC',
            'price': 43251.50,
            'provider': 'coingecko',
            'confidence': 0.95,
            'quorum_size': 1,
            'spread': 0.0,
            'timestamp': 1728741600,
            'change_24h_pct': 2.98,
            'market_cap': 845000000000,
            'price_source': 'contract'  # Present for VIP tokens
        }
    """
    symbol = symbol.upper()

    # Check cache
    if use_cache:
        cached = _get_crypto_cache(symbol)
        if cached:
            LOGGER.debug(f"Crypto price cache hit for {symbol}")
            return cached

    # Reload contract map if file changed
    _load_contract_map()

    # VIP CONTRACT PATH: Use contract address if available
    if symbol in _CONTRACT_MAP:
        contract_info = _CONTRACT_MAP[symbol]
        chain = contract_info["chain"]
        address = contract_info["address"]

        LOGGER.info(f"🎯 VIP token {symbol}: using contract path ({chain}/{address[:8]}...)")

        coingecko = CoinGeckoProvider()
        price_data = coingecko.get_price_by_contract(chain, address, symbol)

        if price_data and price_data.get("price", 0) > 0:
            # Success via contract - package as single-provider quorum
            result = {
                "symbol": symbol,
                "price": price_data["price"],
                "provider": "coingecko",
                "confidence": 0.90,  # High confidence for contract-based
                "quorum_size": 1,
                "spread": 0.0,
                "timestamp": int(time.time()),
                "change_24h_pct": price_data.get("change_24h_pct", 0),
                "market_cap": price_data.get("market_cap", 0),
                "volume_24h": price_data.get("volume_24h", 0),
                "price_source": "contract"  # Debug marker
            }

            _set_crypto_cache(symbol, result)
            LOGGER.info(f"✅ VIP_CONTRACT_PRICE_OK: {symbol} = ${result['price']:.6f}")
            return result
        else:
            LOGGER.warning(f"Contract lookup failed for {symbol}, falling back to symbol search")

    # STANDARD PATH: Symbol-based lookup with quorum
    # Read CRYPTO_QUORUM env for provider order
    import os
    quorum_env = os.getenv("CRYPTO_QUORUM", "coingecko,binance,coinbase")
    provider_names = [p.strip() for p in quorum_env.split(",") if p.strip()]

    # Initialize providers in quorum order
    available_providers = {
        "coingecko": CoinGeckoProvider(),
        "binance": BinanceProvider(),
        "coinbase": CoinbaseProvider(),
    }

    # Collect prices from all providers - SHORT-CIRCUIT on first success to avoid 401/451 retries
    results: list[tuple[str, float, dict]] = []

    for name in provider_names:
        provider = available_providers.get(name)
        if not provider:
            continue

        try:
            price_data = provider.get_price(symbol)
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
