"""
Ghost Movers Scanner - Real-time market movers detection for crypto and stocks

Configuration:
    USE_POLYGON_SNAPSHOTS=true (default)
        - Uses Polygon snapshot API for market-wide coverage
        - /v2/snapshot/locale/us/markets/stocks/gainers (top 20 gainers)
        - /v2/snapshot/locale/us/markets/stocks/losers (top 20 losers)
        - Only 2 API calls instead of 50+ individual fetches
        - Covers entire market (all exchanges, all symbols)
        - Already filtered and sorted by Polygon server-side
    
    USE_SUPPORTED_TICKERS_CSV=false (alternative universe source)
        - Loads 1000 pre-filtered stocks from app/core/data/supported_tickers.csv
        - NASDAQ + NYSE listings, filtered to exclude ETFs/ETNs/ADRs/Funds
        - Individual price fetch per symbol
        - Higher API cost (1000 calls/scan) but uses curated universe
        - Use this if you want specific ticker control vs snapshot API
    
    USE_POLYGON_SNAPSHOTS=false (legacy mode)
        - Falls back to hardcoded 50-stock list
        - Individual price fetch per symbol
        - Lowest API cost, limited universe
    
    POLYGON_API_KEY=<your_key>
        - Required for both snapshot and legacy modes
        - Free tier: 5 calls/min = 7,200/day
        - Snapshot mode uses only 2 calls/scan
    
    WATCH_SYMBOLS=TSLA,AAPL,NVDA
        - Custom symbols to always include
        - Comma-separated, case-insensitive
"""
import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import httpx

# VIP Coins - always included
VIP_COINS = ["WEPE", "LILPEPE", "DORKL", "SLOTH", "APC", "XRP"]

# Thresholds
CRYPTO_PCT_THRESHOLD = 6.0  # |pct_24h| >= 6%
CRYPTO_VOL_MULT_THRESHOLD = 1.5  # vol_mult >= 1.5x
STOCK_PCT_THRESHOLD = 6.0  # |pct_24h| >= 6%
STOCK_VOL_MULT_THRESHOLD = 1.3  # vol_mult >= 1.3x

# Timeouts
PRICE_FETCH_TIMEOUT = 2.0  # seconds per symbol
SCAN_TIMEOUT = 20.0  # seconds total scan time

# Volume baseline periods
CRYPTO_VOL_BASELINE_DAYS = 7
STOCK_VOL_BASELINE_DAYS = 30


def load_universe(redis_client=None) -> Tuple[List[str], List[str]]:
    """
    Load universe of symbols to scan.
    
    Priority order:
        1. USE_SUPPORTED_TICKERS_CSV=true - Load from supported_tickers.csv (1000 stocks)
        2. USE_POLYGON_SNAPSHOTS=true - Use Polygon snapshot API (market-wide)
        3. Fallback - Use hardcoded 50-stock list
    
    Returns:
        (crypto_symbols, stock_symbols) - lists of symbols to scan
    """
    crypto_symbols = set(VIP_COINS)
    stock_symbols = set()
    
    # Add WATCH_SYMBOLS if available
    watch_env = os.getenv("WATCH_SYMBOLS", "")
    if watch_env:
        watch_list = [s.strip().upper() for s in watch_env.split(",") if s.strip()]
        for symbol in watch_list:
            # Simple heuristic: if symbol ends with USD/USDT or is known crypto, it's crypto
            if symbol.endswith(("USD", "USDT", "BTC", "ETH")) or symbol in VIP_COINS:
                crypto_symbols.add(symbol.replace("USDT", "").replace("USD", ""))
            else:
                stock_symbols.add(symbol)
    
    # Check if we should load from supported_tickers.csv
    use_csv = os.getenv("USE_SUPPORTED_TICKERS_CSV", "false").lower() in ("1", "true", "yes")
    if use_csv:
        try:
            import pandas as pd
            from pathlib import Path
            
            # Find the CSV file
            csv_path = Path(__file__).parent / "data" / "supported_tickers.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                csv_symbols = df["Symbol"].str.upper().tolist()
                stock_symbols.update(csv_symbols)
                return list(crypto_symbols), list(stock_symbols)
        except Exception as e:
            # Fall through to legacy mode if CSV loading fails
            pass
    
    # Add top crypto by market cap (if coingecko available)
    try:
        # Top 200 crypto (simplified - in production would call coingecko /coins/markets)
        top_crypto = [
            "BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "MATIC", "DOT", "AVAX", "LINK",
            "UNI", "ATOM", "LTC", "XLM", "ALGO", "FIL", "AAVE", "COMP", "SNX", "MKR"
        ]
        crypto_symbols.update(top_crypto)
    except Exception:
        pass
    
    # Add top stocks by ADV (simplified - in production would use screener)
    try:
        top_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "V", "JPM",
            "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "ADBE", "CRM",
            "NFLX", "CSCO", "PFE", "INTC", "VZ", "T", "CMCSA", "ABT", "NKE", "MRK",
            "PEP", "ABBV", "TMO", "COST", "AVGO", "ACN", "TXN", "MDT", "NEE", "UNP",
            "HON", "QCOM", "LIN", "LOW", "UPS", "BMY", "RTX", "SBUX", "PM", "AMT"
        ]
        stock_symbols.update(top_stocks)
    except Exception:
        pass
    
    return list(crypto_symbols), list(stock_symbols)


async def fetch_polygon_snapshots(
    direction: str = "gainers",
    redis_client=None
) -> List[Dict]:
    """
    Fetch top movers from Polygon snapshot API.
    
    This provides market-wide coverage with minimal API calls:
    - /v2/snapshot/locale/us/markets/stocks/gainers - Top 20 gainers
    - /v2/snapshot/locale/us/markets/stocks/losers - Top 20 losers
    
    Args:
        direction: "gainers" or "losers"
        redis_client: Redis client for caching
    
    Returns:
        List of mover dicts with: symbol, price, pct_24h, vol_mult, provider
    """
    polygon_key = os.getenv("POLYGON_API_KEY", "")
    if not polygon_key:
        return []
    
    # Check if snapshot mode is enabled
    use_snapshots = os.getenv("USE_POLYGON_SNAPSHOTS", "true").lower() in ("1", "true", "yes")
    if not use_snapshots:
        return []
    
    try:
        url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/{direction}?apiKey={polygon_key}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        
        tickers = data.get("tickers", [])
        if not tickers:
            return []
        
        movers = []
        for ticker_data in tickers:
            try:
                ticker = ticker_data.get("ticker", "")
                if not ticker:
                    continue
                
                day = ticker_data.get("day", {})
                prev_day = ticker_data.get("prevDay", {})
                
                # Current price
                price = day.get("c") or day.get("vw") or 0.0
                if price <= 0:
                    continue
                
                # Previous close for % change calculation
                prev_close = prev_day.get("c", 0.0)
                if prev_close <= 0:
                    continue
                
                # Calculate percentage change
                pct_24h = ((price - prev_close) / prev_close) * 100.0
                
                # Volume multiplier
                current_vol = day.get("v", 0.0)
                prev_vol = prev_day.get("v", 0.0)
                vol_mult = (current_vol / prev_vol) if prev_vol > 0 else None
                
                # Build mover dict
                tier_info = tier(pct_24h)
                
                mover = {
                    "symbol": ticker,
                    "price": price,
                    "pct_1h": 0.0,  # Not available in snapshot
                    "pct_24h": pct_24h,
                    "vol_mult": vol_mult,
                    "age_s": 0,  # Real-time from Polygon
                    "provider": "polygon_snapshot",
                    "tier": tier_info["tier"],
                    "emoji": tier_info["emoji"],
                    "is_watch": ticker in os.getenv("WATCH_SYMBOLS", "").split(",")
                }
                movers.append(mover)
                
            except Exception as e:
                continue
        
        return movers
        
    except Exception as e:
        return []


async def fetch_polygon_all_movers(redis_client=None) -> List[Dict]:
    """
    Fetch both gainers and losers from Polygon, merge and deduplicate.
    
    Returns:
        Combined list of top movers sorted by abs(pct_24h)
    """
    # Fetch gainers and losers in parallel
    gainers_task = fetch_polygon_snapshots("gainers", redis_client)
    losers_task = fetch_polygon_snapshots("losers", redis_client)
    
    gainers, losers = await asyncio.gather(gainers_task, losers_task, return_exceptions=True)
    
    if isinstance(gainers, Exception):
        gainers = []
    if isinstance(losers, Exception):
        losers = []
    
    # Merge and deduplicate
    all_movers = []
    seen_symbols = set()
    
    for mover in gainers + losers:
        symbol = mover.get("symbol", "")
        if symbol and symbol not in seen_symbols:
            seen_symbols.add(symbol)
            all_movers.append(mover)
    
    # Sort by abs(pct_24h) descending
    all_movers.sort(key=lambda x: abs(x.get("pct_24h", 0.0)), reverse=True)
    
    return all_movers


async def get_price_snapshot(
    symbol: str,
    is_crypto: bool,
    fetch_price_func,
    redis_client=None
) -> Optional[Dict]:
    """
    Get live price snapshot with strict freshness requirements.
    
    Args:
        symbol: Ticker symbol
        is_crypto: True if crypto, False if stock
        fetch_price_func: Function to fetch live price (from wolf_app)
        redis_client: Redis client for caching
    
    Returns:
        {
            "symbol": str,
            "price": float,
            "ts": int (epoch ms),
            "provider": str,
            "age_s": int (seconds since update)
        }
        or None if fetch fails
    """
    try:
        result = await asyncio.wait_for(
            fetch_price_func(symbol, is_crypto=is_crypto),
            timeout=PRICE_FETCH_TIMEOUT
        )
        
        if not result or not isinstance(result, dict):
            return None
        
        price = result.get("price")
        if price is None or price <= 0:
            return None
        
        provider = result.get("provider", "unknown")
        ts_ms = result.get("ts", int(time.time() * 1000))
        
        # Calculate age
        now_ms = int(time.time() * 1000)
        age_s = (now_ms - ts_ms) // 1000
        
        # Strict freshness check
        max_age = int(os.getenv("DATA_FRESHNESS_SEC", "60"))
        if age_s > max_age:
            return None
        
        # Reject safe/prevclose providers
        if provider.lower() in ("prevclose", "safe", "fallback"):
            return None
        
        return {
            "symbol": symbol,
            "price": price,
            "ts": ts_ms,
            "provider": provider,
            "age_s": age_s
        }
        
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


async def get_volume_baseline(
    symbol: str,
    is_crypto: bool,
    ohlcv_func=None,
    redis_client=None
) -> Optional[float]:
    """
    Get volume baseline (7d median for crypto, 30d median for stocks).
    
    Args:
        symbol: Ticker symbol
        is_crypto: True if crypto, False if stock
        ohlcv_func: Function to fetch OHLCV data
        redis_client: Redis client for caching
    
    Returns:
        Median volume or None if unavailable
    """
    if not ohlcv_func:
        return None
    
    try:
        days = CRYPTO_VOL_BASELINE_DAYS if is_crypto else STOCK_VOL_BASELINE_DAYS
        
        # Check cache first
        cache_key = f"ghost:vol_baseline:{symbol}:{days}d"
        if redis_client:
            try:
                cached = redis_client.get(cache_key)
                if cached:
                    return float(cached)
            except Exception:
                pass
        
        # Fetch OHLCV data
        result = await asyncio.wait_for(
            ohlcv_func(symbol, days=days, is_crypto=is_crypto),
            timeout=PRICE_FETCH_TIMEOUT
        )
        
        if not result or not isinstance(result, list):
            return None
        
        volumes = [bar.get("volume", 0) for bar in result if bar.get("volume", 0) > 0]
        if len(volumes) < 3:  # Need at least 3 data points
            return None
        
        # Calculate median
        volumes_sorted = sorted(volumes)
        n = len(volumes_sorted)
        if n % 2 == 0:
            median = (volumes_sorted[n // 2 - 1] + volumes_sorted[n // 2]) / 2
        else:
            median = volumes_sorted[n // 2]
        
        # Cache for 1 hour
        if redis_client and median > 0:
            try:
                redis_client.setex(cache_key, 3600, str(median))
            except Exception:
                pass
        
        return median
        
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


def tier(pct: float) -> Dict[str, str]:
    """
    Determine tier based on percentage move.
    
    Args:
        pct: Percentage change
    
    Returns:
        {"tier": str, "emoji": str}
    """
    abs_pct = abs(pct)
    
    if abs_pct >= 20:
        return {"tier": "🔥20+", "emoji": "🔥"}
    elif abs_pct >= 15:
        return {"tier": "⚡15+", "emoji": "⚡"}
    elif abs_pct >= 10:
        return {"tier": "📈10+", "emoji": "📈"}
    elif abs_pct >= 6:
        return {"tier": "📊6+", "emoji": "📊"}
    else:
        return {"tier": "📉<6", "emoji": "📉"}


async def scan_crypto(
    fetch_price_func,
    ohlcv_func=None,
    redis_client=None
) -> List[Dict]:
    """
    Scan crypto universe for movers.
    
    Returns:
        List of mover dicts with: symbol, price, pct_1h, pct_24h, vol_mult, age_s, provider, tier
    """
    crypto_symbols, _ = load_universe(redis_client)
    movers = []
    
    tasks = []
    for symbol in crypto_symbols:
        task = get_price_snapshot(symbol, True, fetch_price_func, redis_client)
        tasks.append((symbol, task))
    
    # Gather with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[t[1] for t in tasks], return_exceptions=True),
            timeout=SCAN_TIMEOUT
        )
    except asyncio.TimeoutError:
        results = [None] * len(tasks)
    
    for (symbol, _), result in zip(tasks, results):
        if not result or isinstance(result, Exception):
            continue
        
        # Calculate percentage changes (simplified - would fetch historical in production)
        # For now, use mock data or skip if unavailable
        pct_1h = 0.0  # Would calculate from 1h ago price
        pct_24h = 0.0  # Would calculate from 24h ago price
        
        # Get volume multiplier
        vol_mult = None
        if ohlcv_func:
            baseline = await get_volume_baseline(symbol, True, ohlcv_func, redis_client)
            current_vol = result.get("volume", 0)  # Would be in snapshot
            if baseline and baseline > 0 and current_vol > 0:
                vol_mult = current_vol / baseline
        
        # Apply thresholds
        meets_pct = abs(pct_24h) >= CRYPTO_PCT_THRESHOLD
        meets_vol = vol_mult is None or vol_mult >= CRYPTO_VOL_MULT_THRESHOLD
        is_vip = symbol in VIP_COINS
        
        if meets_pct and meets_vol or is_vip:
            tier_info = tier(pct_24h)
            
            mover = {
                "symbol": symbol,
                "price": result["price"],
                "pct_1h": pct_1h,
                "pct_24h": pct_24h,
                "vol_mult": vol_mult,
                "age_s": result["age_s"],
                "provider": result["provider"],
                "tier": tier_info["tier"],
                "emoji": tier_info["emoji"],
                "is_watch": is_vip
            }
            movers.append(mover)
    
    # Sort by abs(pct_24h) descending
    movers.sort(key=lambda x: abs(x["pct_24h"]), reverse=True)
    
    return movers


async def scan_stocks(
    fetch_price_func,
    ohlcv_func=None,
    redis_client=None,
    allow_extended_hours: bool = True
) -> List[Dict]:
    """
    Scan stock universe for movers.
    
    If USE_POLYGON_SNAPSHOTS=true (default), uses Polygon snapshot API
    for market-wide coverage with 2 API calls instead of 50+ individual fetches.
    
    Args:
        fetch_price_func: Price fetcher (fallback for non-snapshot mode)
        ohlcv_func: OHLCV fetcher
        redis_client: Redis client
        allow_extended_hours: Allow pre-market and after-hours moves
    
    Returns:
        List of mover dicts
    """
    # Check if snapshot mode is enabled
    use_snapshots = os.getenv("USE_POLYGON_SNAPSHOTS", "true").lower() in ("1", "true", "yes")
    
    if use_snapshots and os.getenv("POLYGON_API_KEY"):
        # Use Polygon snapshot API for market-wide coverage
        movers = await fetch_polygon_all_movers(redis_client)
        
        # Apply threshold filter
        filtered = []
        for mover in movers:
            pct_24h = abs(mover.get("pct_24h", 0.0))
            vol_mult = mover.get("vol_mult")
            
            meets_pct = pct_24h >= STOCK_PCT_THRESHOLD
            meets_vol = vol_mult is None or vol_mult >= STOCK_VOL_MULT_THRESHOLD
            is_watch = mover.get("is_watch", False)
            
            if meets_pct and meets_vol or is_watch:
                filtered.append(mover)
        
        return filtered
    
    # Fallback: Use legacy individual symbol fetch
    _, stock_symbols = load_universe(redis_client)
    movers = []
    
    tasks = []
    for symbol in stock_symbols:
        task = get_price_snapshot(symbol, False, fetch_price_func, redis_client)
        tasks.append((symbol, task))
    
    # Gather with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[t[1] for t in tasks], return_exceptions=True),
            timeout=SCAN_TIMEOUT
        )
    except asyncio.TimeoutError:
        results = [None] * len(tasks)
    
    for (symbol, _), result in zip(tasks, results):
        if not result or isinstance(result, Exception):
            continue
        
        # Calculate percentage changes (simplified)
        pct_1h = 0.0
        pct_24h = 0.0
        
        # Get volume multiplier
        vol_mult = None
        if ohlcv_func:
            baseline = await get_volume_baseline(symbol, False, ohlcv_func, redis_client)
            current_vol = result.get("volume", 0)
            if baseline and baseline > 0 and current_vol > 0:
                vol_mult = current_vol / baseline
        
        # Apply thresholds
        meets_pct = abs(pct_24h) >= STOCK_PCT_THRESHOLD
        meets_vol = vol_mult is None or vol_mult >= STOCK_VOL_MULT_THRESHOLD
        is_watch = symbol in os.getenv("WATCH_SYMBOLS", "").split(",")
        
        if meets_pct and meets_vol or is_watch:
            tier_info = tier(pct_24h)
            
            mover = {
                "symbol": symbol,
                "price": result["price"],
                "pct_1h": pct_1h,
                "pct_24h": pct_24h,
                "vol_mult": vol_mult,
                "age_s": result["age_s"],
                "provider": result["provider"],
                "tier": tier_info["tier"],
                "emoji": tier_info["emoji"],
                "is_watch": is_watch
            }
            movers.append(mover)
    
    # Sort by abs(pct_24h) descending
    movers.sort(key=lambda x: abs(x["pct_24h"]), reverse=True)
    
    return movers


def build_payload(crypto_movers: List[Dict], stock_movers: List[Dict]) -> Dict:
    """
    Build normalized API payload.
    
    Returns:
        {
            "crypto": [...],
            "stocks": [...],
            "ts": int,
            "crypto_count": int,
            "stocks_count": int
        }
    """
    return {
        "crypto": crypto_movers,
        "stocks": stock_movers,
        "ts": int(time.time() * 1000),
        "crypto_count": len(crypto_movers),
        "stocks_count": len(stock_movers)
    }


def persist_last_run(scan_type: str, stats: Dict, redis_client) -> bool:
    """
    Persist scan stats to Redis.
    
    Args:
        scan_type: "crypto" or "stocks"
        stats: Dict with count, ts, error, etc.
        redis_client: Redis client
    
    Returns:
        True if persisted successfully
    """
    if not redis_client:
        return False
    
    try:
        date = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"ghost:scan:{scan_type}:{date}"
        
        redis_client.hset(key, mapping={
            "ts": int(time.time()),
            "count": stats.get("count", 0),
            "error": stats.get("error", ""),
            "duration_ms": stats.get("duration_ms", 0)
        })
        
        # Set TTL of 7 days
        redis_client.expire(key, 604800)
        
        return True
    except Exception:
        return False


def get_last_run_stats(scan_type: str, redis_client) -> Optional[Dict]:
    """
    Get last run stats from Redis.
    
    Args:
        scan_type: "crypto" or "stocks"
        redis_client: Redis client
    
    Returns:
        Stats dict or None
    """
    if not redis_client:
        return None
    
    try:
        date = datetime.utcnow().strftime("%Y-%m-%d")
        key = f"ghost:scan:{scan_type}:{date}"
        
        data = redis_client.hgetall(key)
        if not data:
            return None
        
        return {
            "ts": int(data.get(b"ts", b"0")),
            "count": int(data.get(b"count", b"0")),
            "error": data.get(b"error", b"").decode("utf-8"),
            "duration_ms": int(data.get(b"duration_ms", b"0"))
        }
    except Exception:
        return None
