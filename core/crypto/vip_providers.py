"""
VIP Coin Price Providers
Handles meme/VIP coins: WEPE, LILPEPE, DORKL, SLOTH, APC

Strategy:
1. Try CoinGecko for known VIP tokens
2. Return structured "NO DATA" if unavailable
3. No simulation - only real prices or explicit NO DATA state
"""

import logging
import os
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

# Shared HTTP session
_session = requests.Session()
_retry_strategy = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# Cache for VIP prices (30 second TTL)
_VIP_CACHE: dict[str, dict[str, Any]] = {}
_VIP_CACHE_TTL_S = int(os.getenv("VIP_CACHE_TTL_S", "30"))

# VIP coin mapping to CoinGecko IDs (where available)
# Updated Nov 20, 2025 - Priority 2: VIP Microcap Scanner
VIP_COIN_MAP = {
    "WEPE": "wall-street-pepe",  # Wall Street Pepe (Verified on CoinGecko)
    "LILPEPE": "pepe-2",  # Lil Pepe (Alternative CoinGecko ID)
    "DORKL": "dork-lord",  # Dork Lord (Corrected ID)
    "SLOTH": "slothana",  # Slothana (Verified on CoinGecko)
    "APC": "ape-coin",  # ApeCoin (APE on CoinGecko)
}

# DEXScreener contract addresses for fallback (when CoinGecko unavailable)
VIP_DEX_CONTRACTS = {
    "WEPE": "0x",  # Ethereum mainnet contract (will search DEXScreener)
    "LILPEPE": "0x",  # Will use DEXScreener search
    "DORKL": "0x",  # Will use DEXScreener search
    "SLOTH": "0x",  # Solana address
    "APC": None,  # ApeCoin is on major CEXs, no DEX needed
}

# Last successful provider usage tracking
_LAST_VIP_PROVIDER_SUCCESS: dict[str, float] = {}


def _get_vip_cache(symbol: str) -> dict[str, Any] | None:
    """Get cached VIP price if still valid"""
    if symbol in _VIP_CACHE:
        age = time.time() - _VIP_CACHE[symbol].get("timestamp", 0)
        if age < _VIP_CACHE_TTL_S:
            return _VIP_CACHE[symbol]
    return None


def _set_vip_cache(symbol: str, data: dict[str, Any]) -> None:
    """Cache VIP price data"""
    _VIP_CACHE[symbol] = data


def get_vip_price(symbol: str, use_cache: bool = True) -> dict[str, Any]:
    """
    Get VIP coin price from available providers.
    
    Strategy:
    1. Check cache
    2. Try CoinGecko (primary)
    3. Try DEXScreener (fallback for tokens not on CG)
    4. Return structured NO DATA if all fail
    
    Returns:
        {
            'symbol': 'WEPE',
            'price': 0.00123,
            'provider': 'coingecko',
            'confidence': 0.70,
            'timestamp': 1731654000,
            'available': True
        }
        
        Or for NO DATA:
        {
            'symbol': 'LILPEPE',
            'price': None,
            'provider': 'none',
            'confidence': 0.0,
            'timestamp': 1731654000,
            'available': False,
            'reason': 'Not available on any provider'
        }
    """
    symbol = symbol.upper()
    
    # Check cache
    if use_cache:
        cached = _get_vip_cache(symbol)
        if cached:
            LOGGER.debug(f"VIP price cache hit for {symbol}")
            return cached
    
    # Try CoinGecko if symbol is mapped
    coingecko_id = VIP_COIN_MAP.get(symbol)
    
    if coingecko_id:
        try:
            result = _fetch_from_coingecko(symbol, coingecko_id)
            if result and result.get("available"):
                _LAST_VIP_PROVIDER_SUCCESS[symbol] = time.time()
                _set_vip_cache(symbol, result)
                return result
        except Exception as e:
            LOGGER.warning(f"CoinGecko failed for VIP coin {symbol}: {e}")
    
    # Try DEXScreener as fallback
    dex_contract = VIP_DEX_CONTRACTS.get(symbol)
    if dex_contract and dex_contract != "0x":  # Skip placeholder addresses
        try:
            result = _fetch_from_dexscreener(symbol, dex_contract)
            if result and result.get("available"):
                _LAST_VIP_PROVIDER_SUCCESS[symbol] = time.time()
                _set_vip_cache(symbol, result)
                return result
        except Exception as e:
            LOGGER.warning(f"DEXScreener failed for VIP coin {symbol}: {e}")
    
    # Return structured NO DATA result
    no_data_result = {
        "symbol": symbol,
        "price": None,
        "provider": "none",
        "confidence": 0.0,
        "timestamp": time.time(),
        "available": False,
        "reason": f"Not available on any provider (coingecko_id: {coingecko_id})"
    }
    
    _set_vip_cache(symbol, no_data_result)
    return no_data_result


def _fetch_from_coingecko(symbol: str, coingecko_id: str) -> dict[str, Any] | None:
    """Fetch VIP coin price from CoinGecko"""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coingecko_id,
            "vs_currencies": "usd",
            "include_24h_change": "true",
            "include_market_cap": "true"
        }
        
        response = _session.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if coingecko_id not in data:
            return None
        
        coin_data = data[coingecko_id]
        price = coin_data.get("usd")
        
        if not price or price <= 0:
            return None
        
        return {
            "symbol": symbol,
            "price": price,
            "provider": "coingecko",
            "confidence": 0.70,  # Lower confidence for VIP coins
            "timestamp": time.time(),
            "available": True,
            "change_24h_pct": coin_data.get("usd_24h_change", 0),
            "market_cap": coin_data.get("usd_market_cap", 0)
        }
        
    except Exception as e:
        LOGGER.debug(f"CoinGecko fetch failed for {symbol} ({coingecko_id}): {e}")
        return None


def _fetch_from_dexscreener(symbol: str, contract_address: str) -> dict[str, Any] | None:
    """
    Fetch VIP coin price from DEXScreener API
    
    DEXScreener API is free and provides real-time DEX data
    Docs: https://docs.dexscreener.com/api/reference
    """
    try:
        # DEXScreener search by token address (works for ETH, BSC, Solana, etc.)
        url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
        
        response = _session.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        pairs = data.get("pairs", [])
        
        if not pairs:
            return None
        
        # Get highest liquidity pair for best price reliability
        pairs_sorted = sorted(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
        best_pair = pairs_sorted[0]
        
        price = float(best_pair.get("priceUsd", 0))
        if price <= 0:
            return None
        
        # Calculate 24h change from h24 data
        price_change_24h = best_pair.get("priceChange", {}).get("h24", 0)
        
        return {
            "symbol": symbol,
            "price": price,
            "provider": "dexscreener",
            "confidence": 0.60,  # Lower confidence for DEX data
            "timestamp": time.time(),
            "available": True,
            "change_24h_pct": float(price_change_24h or 0),
            "market_cap": 0,  # DEXScreener doesn't provide reliable market cap
            "volume_24h_usd": float(best_pair.get("volume", {}).get("h24", 0) or 0),
            "liquidity_usd": float(best_pair.get("liquidity", {}).get("usd", 0) or 0),
            "dex_name": best_pair.get("dexId", "unknown")
        }
        
    except Exception as e:
        LOGGER.debug(f"DEXScreener fetch failed for {symbol} ({contract_address}): {e}")
        return None


def get_last_vip_provider_success() -> dict[str, float]:
    """
    Get last successful provider fetch time for each VIP coin.
    
    Returns:
        {'WEPE': 1731654000.123, 'DORKL': 1731654010.456}
    """
    return _LAST_VIP_PROVIDER_SUCCESS.copy()


def get_vip_provider_health() -> dict[str, Any]:
    """
    Get VIP provider health summary.
    
    Returns:
        {
            'symbols_with_data': 2,
            'symbols_without_data': 3,
            'last_success': {'WEPE': 1731654000, 'DORKL': 1731654010},
            'available_symbols': ['WEPE', 'DORKL']
        }
    """
    available = [sym for sym, cg_id in VIP_COIN_MAP.items() if cg_id is not None]
    
    return {
        "symbols_with_data": len(available),
        "symbols_without_data": len(VIP_COIN_MAP) - len(available),
        "last_success": get_last_vip_provider_success(),
        "available_symbols": available
    }
