#!/usr/bin/env python3
"""
Ghost Protocol - Real-Time Market Movers Scanner
=================================================

Discovers TODAY's biggest moving stocks and crypto in real-time.
Automatically feeds discoveries into the prediction pipeline.

Data Sources (all FREE):
1. Yahoo Finance - Top gainers/losers (no API key needed)
2. CoinGecko - Top crypto movers (free tier)
3. Finviz - Stock screener (free tier)

Runs every 30 minutes during market hours to catch movers EARLY.
"""

import asyncio
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import aiohttp

LOGGER = logging.getLogger("core.realtime_market_movers")

# Configuration
MOVERS_ENABLED = os.getenv("REALTIME_MOVERS_ENABLED", "1") == "1"
CHECK_INTERVAL_MINUTES = int(os.getenv("MOVERS_CHECK_INTERVAL", "30"))
MIN_MOVE_PCT = float(os.getenv("MOVERS_MIN_MOVE_PCT", "3.0"))  # 3%+ moves
MIN_VOLUME = int(os.getenv("MOVERS_MIN_VOLUME", "500000"))  # 500k volume
MAX_DISCOVERIES = int(os.getenv("MOVERS_MAX_DISCOVERIES", "20"))  # Top 20 movers

# Track discovered symbols to avoid duplicates
_DISCOVERED_TODAY: Set[str] = set()
_LAST_DISCOVERY_DATE: str = ""
_MOVERS_THREAD: Optional[threading.Thread] = None
_STOP_EVENT = threading.Event()

# Callback to add discovered symbols to prediction pipeline
_ON_DISCOVERY_CALLBACK = None


def set_discovery_callback(callback):
    """Set callback function for when new movers are discovered."""
    global _ON_DISCOVERY_CALLBACK
    _ON_DISCOVERY_CALLBACK = callback
    LOGGER.info("Discovery callback registered")


class RealtimeMoversScanner:
    """Scans multiple free sources for real-time market movers."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.et_tz = ZoneInfo("America/New_York")
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    def is_market_hours(self) -> bool:
        """Check if US stock market is open."""
        now = datetime.now(self.et_tz)
        # Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        if now.weekday() >= 5:  # Weekend
            return False
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        return market_open <= now <= market_close
    
    async def scan_all_sources(self) -> List[Dict[str, Any]]:
        """Scan all sources for today's movers."""
        all_movers = []
        
        # Scan stocks
        try:
            yahoo_movers = await self._scan_yahoo_movers()
            all_movers.extend(yahoo_movers)
            LOGGER.info(f"Yahoo: Found {len(yahoo_movers)} stock movers")
        except Exception as e:
            LOGGER.error(f"Yahoo scan failed: {e}")
        
        # Scan crypto
        try:
            crypto_movers = await self._scan_crypto_movers()
            all_movers.extend(crypto_movers)
            LOGGER.info(f"Crypto: Found {len(crypto_movers)} crypto movers")
        except Exception as e:
            LOGGER.error(f"Crypto scan failed: {e}")
        
        # Sort by absolute move percentage
        all_movers.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
        
        return all_movers[:MAX_DISCOVERIES]
    
    async def _scan_yahoo_movers(self) -> List[Dict[str, Any]]:
        """
        Scan Yahoo Finance for top gainers and losers.
        Uses the public screener API (no key needed).
        """
        movers = []
        
        # Yahoo Finance screener URLs
        urls = [
            # Day gainers - stocks up the most today
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers&count=50",
            # Day losers - stocks down the most today  
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_losers&count=50",
            # Most active by volume
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=50",
        ]
        
        for url in urls:
            try:
                async with self.session.get(url, timeout=15) as resp:
                    if resp.status != 200:
                        continue
                    
                    data = await resp.json()
                    quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
                    
                    for quote in quotes:
                        symbol = quote.get("symbol", "")
                        change_pct = quote.get("regularMarketChangePercent", 0)
                        volume = quote.get("regularMarketVolume", 0)
                        price = quote.get("regularMarketPrice", 0)
                        name = quote.get("shortName", symbol)
                        
                        # Filter criteria
                        if not symbol or "." in symbol:  # Skip ADRs with dots
                            continue
                        if abs(change_pct) < MIN_MOVE_PCT:
                            continue
                        if volume < MIN_VOLUME:
                            continue
                        if price < 1:  # Skip penny stocks
                            continue
                        
                        movers.append({
                            "symbol": symbol,
                            "name": name,
                            "price": price,
                            "change_pct": change_pct,
                            "volume": volume,
                            "type": "stock",
                            "source": "yahoo",
                            "direction": "UP" if change_pct > 0 else "DOWN",
                        })
                
                await asyncio.sleep(0.5)  # Rate limit
                
            except Exception as e:
                LOGGER.warning(f"Yahoo URL failed: {e}")
        
        # Deduplicate by symbol
        seen = set()
        unique_movers = []
        for m in movers:
            if m["symbol"] not in seen:
                seen.add(m["symbol"])
                unique_movers.append(m)
        
        return unique_movers
    
    async def _scan_crypto_movers(self) -> List[Dict[str, Any]]:
        """
        Scan CoinGecko for top crypto movers.
        Free tier allows 10-30 calls/minute.
        """
        movers = []
        
        try:
            # Get top 250 coins sorted by 24h price change
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "volume_desc",  # Changed from price_change - more stable
                "per_page": 100,
                "page": 1,
                "sparkline": "false",  # Use string "false" not bool
            }
            
            async with self.session.get(url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    LOGGER.warning(f"CoinGecko returned {resp.status}")
                    return []
                
                data = await resp.json()
                
                for coin in data:
                    symbol = coin.get("symbol", "").upper()
                    change_pct = coin.get("price_change_percentage_24h")
                    volume = coin.get("total_volume")
                    price = coin.get("current_price")
                    name = coin.get("name", symbol)
                    
                    # Handle None values
                    if change_pct is None:
                        continue
                    if volume is None:
                        volume = 0
                    if price is None:
                        continue
                    
                    # Filter criteria
                    if not symbol:
                        continue
                    if abs(change_pct) < MIN_MOVE_PCT:
                        continue
                    if volume < 1000000:  # $1M minimum volume for crypto
                        continue
                    
                    movers.append({
                        "symbol": symbol,
                        "name": name,
                        "price": price,
                        "change_pct": change_pct,
                        "volume": volume,
                        "type": "crypto",
                        "source": "coingecko",
                        "direction": "UP" if change_pct > 0 else "DOWN",
                    })
        
        except Exception as e:
            LOGGER.error(f"CoinGecko scan error: {e}")
        
        # Sort by absolute change and return top movers
        movers.sort(key=lambda x: abs(x.get("change_pct", 0)), reverse=True)
        
        return movers[:30]  # Top 30 crypto movers


async def discover_movers() -> List[Dict[str, Any]]:
    """
    Discover today's market movers and return new discoveries.
    Filters out symbols already discovered today.
    """
    global _DISCOVERED_TODAY, _LAST_DISCOVERY_DATE
    
    # Reset daily tracking
    today = datetime.now().strftime("%Y-%m-%d")
    if today != _LAST_DISCOVERY_DATE:
        _DISCOVERED_TODAY.clear()
        _LAST_DISCOVERY_DATE = today
        LOGGER.info(f"New day - reset discovered symbols for {today}")
    
    async with RealtimeMoversScanner() as scanner:
        movers = await scanner.scan_all_sources()
    
    # Filter to NEW discoveries only
    new_movers = []
    for mover in movers:
        symbol = mover["symbol"]
        if symbol not in _DISCOVERED_TODAY:
            _DISCOVERED_TODAY.add(symbol)
            new_movers.append(mover)
            LOGGER.info(
                f"🔥 NEW MOVER: {symbol} ({mover['type']}) "
                f"{mover['change_pct']:+.1f}% @ ${mover['price']:.2f}"
            )
    
    return new_movers


def _movers_loop():
    """Background loop that scans for movers."""
    LOGGER.info("Real-time movers scanner started")
    
    while not _STOP_EVENT.is_set():
        try:
            # Run discovery
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                new_movers = loop.run_until_complete(discover_movers())
                
                if new_movers and _ON_DISCOVERY_CALLBACK:
                    LOGGER.info(f"Discovered {len(new_movers)} new movers, triggering callback")
                    try:
                        _ON_DISCOVERY_CALLBACK(new_movers)
                    except Exception as e:
                        LOGGER.error(f"Discovery callback failed: {e}")
                
            finally:
                loop.close()
            
        except Exception as e:
            LOGGER.error(f"Movers loop error: {e}", exc_info=True)
        
        # Sleep until next check
        for _ in range(CHECK_INTERVAL_MINUTES * 60):
            if _STOP_EVENT.is_set():
                break
            time.sleep(1)
    
    LOGGER.info("Real-time movers scanner stopped")


def start_movers_scanner():
    """Start the background movers scanner."""
    global _MOVERS_THREAD
    
    if not MOVERS_ENABLED:
        LOGGER.info("Real-time movers scanner disabled (REALTIME_MOVERS_ENABLED=0)")
        return
    
    if _MOVERS_THREAD and _MOVERS_THREAD.is_alive():
        LOGGER.warning("Movers scanner already running")
        return
    
    _STOP_EVENT.clear()
    _MOVERS_THREAD = threading.Thread(
        target=_movers_loop,
        name="realtime-movers",
        daemon=True
    )
    _MOVERS_THREAD.start()
    LOGGER.info(f"✅ Real-time movers scanner started (checking every {CHECK_INTERVAL_MINUTES} min)")


def stop_movers_scanner():
    """Stop the background movers scanner."""
    _STOP_EVENT.set()
    if _MOVERS_THREAD and _MOVERS_THREAD.is_alive():
        _MOVERS_THREAD.join(timeout=5)
    LOGGER.info("Real-time movers scanner stopped")


def get_scanner_status() -> Dict[str, Any]:
    """Get current scanner status."""
    return {
        "enabled": MOVERS_ENABLED,
        "running": _MOVERS_THREAD.is_alive() if _MOVERS_THREAD else False,
        "check_interval_minutes": CHECK_INTERVAL_MINUTES,
        "min_move_pct": MIN_MOVE_PCT,
        "discovered_today": list(_DISCOVERED_TODAY),
        "discovered_count": len(_DISCOVERED_TODAY),
        "last_discovery_date": _LAST_DISCOVERY_DATE,
    }


# Manual trigger for testing
async def manual_scan() -> List[Dict[str, Any]]:
    """Manually trigger a scan (for testing/API endpoint)."""
    return await discover_movers()


if __name__ == "__main__":
    # Test the scanner
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        movers = await discover_movers()
        print(f"\n=== Found {len(movers)} NEW movers ===\n")
        for m in movers[:10]:
            print(f"{m['symbol']:8} {m['type']:6} {m['change_pct']:+6.1f}% ${m['price']:>10.2f}  {m['name'][:30]}")
    
    asyncio.run(test())
