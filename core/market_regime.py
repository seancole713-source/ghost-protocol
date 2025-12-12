"""
📊 MARKET REGIME DETECTOR
Identifies current market phase: Bull, Bear, Crash, Recovery, Sideways
Uses: VIX, SPY trend, sector rotation, breadth indicators
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Literal

import aiohttp

LOGGER = logging.getLogger(__name__)

# Market regime types
MarketRegime = Literal["BULL", "BEAR", "CRASH", "RECOVERY", "SIDEWAYS"]

# Cache
_REGIME_CACHE: dict[str, any] = {"regime": None, "timestamp": 0}
_CACHE_TTL = 300  # 5 minutes


# ============================================================================
# VIX ANALYSIS (Fear Gauge)
# ============================================================================

async def get_vix_level() -> float:
    """
    Get current VIX level (0-100+)
    <20 = Low volatility (bull market)
    20-30 = Elevated volatility
    30-50 = High volatility (bear market)
    >50 = Panic (crash)
    """
    try:
        # Use Polygon.io for VIX
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            return 20.0  # Default neutral
        
        url = f"https://api.polygon.io/v2/aggs/ticker/VIX/prev?adjusted=true&apiKey={api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return 20.0
                
                data = await resp.json()
                results = data.get("results", [])
                
                if not results:
                    return 20.0
                
                vix_close = results[0].get("c", 20.0)
                return float(vix_close)
                
    except Exception as e:
        LOGGER.error(f"VIX fetch failed: {e}")
        return 20.0


# ============================================================================
# SPY TREND ANALYSIS
# ============================================================================

async def get_spy_trend() -> dict:
    """
    Analyze SPY trend: above/below SMA50, SMA200
    """
    try:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            return {"price": 500, "sma50": 500, "sma200": 500, "above_sma50": True, "above_sma200": True}
        
        # Get SPY daily bars (last 200 days)
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{(datetime.utcnow() - timedelta(days=250)).strftime('%Y-%m-%d')}/{datetime.utcnow().strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apiKey={api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return {"price": 500, "sma50": 500, "sma200": 500, "above_sma50": True, "above_sma200": True}
                
                data = await resp.json()
                results = data.get("results", [])
                
                if len(results) < 200:
                    return {"price": 500, "sma50": 500, "sma200": 500, "above_sma50": True, "above_sma200": True}
                
                # Calculate SMAs
                closes = [bar["c"] for bar in results]
                
                current_price = closes[-1]
                sma50 = sum(closes[-50:]) / 50
                sma200 = sum(closes[-200:]) / 200
                
                return {
                    "price": current_price,
                    "sma50": sma50,
                    "sma200": sma200,
                    "above_sma50": current_price > sma50,
                    "above_sma200": current_price > sma200
                }
                
    except Exception as e:
        LOGGER.error(f"SPY trend analysis failed: {e}")
        return {"price": 500, "sma50": 500, "sma200": 500, "above_sma50": True, "above_sma200": True}


# ============================================================================
# SECTOR ROTATION ANALYSIS
# ============================================================================

async def get_sector_rotation() -> dict:
    """
    Check which sectors are leading/lagging
    Bullish: Tech, Discretionary leading
    Bearish: Utilities, Healthcare leading
    """
    try:
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            return {"rotation": "NEUTRAL"}
        
        # Sector ETFs
        sectors = {
            "XLK": "Technology",
            "XLY": "Consumer Discretionary",
            "XLF": "Financials",
            "XLE": "Energy",
            "XLU": "Utilities",
            "XLV": "Healthcare"
        }
        
        sector_performance = {}
        
        async with aiohttp.ClientSession() as session:
            for ticker in sectors.keys():
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={api_key}"
                
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        
                        if results:
                            prev_close = results[0].get("c", 0)
                            open_price = results[0].get("o", 0)
                            
                            if prev_close > 0:
                                perf = ((prev_close - open_price) / open_price) * 100
                                sector_performance[ticker] = perf
        
        if not sector_performance:
            return {"rotation": "NEUTRAL"}
        
        # Check if defensive (XLU, XLV) outperforming
        defensive_perf = (sector_performance.get("XLU", 0) + sector_performance.get("XLV", 0)) / 2
        offensive_perf = (sector_performance.get("XLK", 0) + sector_performance.get("XLY", 0)) / 2
        
        if offensive_perf > defensive_perf + 0.5:
            return {"rotation": "BULLISH", "leaders": "Tech/Discretionary"}
        elif defensive_perf > offensive_perf + 0.5:
            return {"rotation": "BEARISH", "leaders": "Utilities/Healthcare"}
        else:
            return {"rotation": "NEUTRAL"}
        
    except Exception as e:
        LOGGER.error(f"Sector rotation analysis failed: {e}")
        return {"rotation": "NEUTRAL"}


# ============================================================================
# MARKET BREADTH (Advance/Decline)
# ============================================================================

async def get_market_breadth() -> dict:
    """
    Check advance/decline ratio
    >1.5 = Strong breadth (bull)
    <0.7 = Weak breadth (bear)
    """
    try:
        # Placeholder: Would integrate with NYSE advance/decline API
        # For now, return neutral
        return {"advance_decline_ratio": 1.0, "breadth": "NEUTRAL"}
        
    except Exception as e:
        LOGGER.error(f"Market breadth failed: {e}")
        return {"advance_decline_ratio": 1.0, "breadth": "NEUTRAL"}


# ============================================================================
# REGIME DETECTION
# ============================================================================

async def detect_market_regime() -> dict:
    """
    Combine all signals to determine market regime
    """
    # Check cache
    if _REGIME_CACHE["regime"] is not None:
        if datetime.utcnow().timestamp() - _REGIME_CACHE["timestamp"] < _CACHE_TTL:
            return _REGIME_CACHE["regime"]
    
    try:
        # Fetch all indicators in parallel
        vix, spy_trend, sector_rotation, breadth = await asyncio.gather(
            get_vix_level(),
            get_spy_trend(),
            get_sector_rotation(),
            get_market_breadth(),
            return_exceptions=True
        )
        
        # Default values if errors
        if isinstance(vix, Exception):
            vix = 20.0
        if isinstance(spy_trend, Exception):
            spy_trend = {"above_sma50": True, "above_sma200": True}
        if isinstance(sector_rotation, Exception):
            sector_rotation = {"rotation": "NEUTRAL"}
        if isinstance(breadth, Exception):
            breadth = {"breadth": "NEUTRAL"}
        
        # Regime decision logic
        regime: MarketRegime = "SIDEWAYS"
        confidence = 50
        
        # CRASH: VIX >50, SPY below both SMAs
        if vix > 50 and not spy_trend["above_sma50"]:
            regime = "CRASH"
            confidence = 95
        
        # BEAR: VIX 30-50, SPY below SMA50, defensive sectors leading
        elif vix > 30 and not spy_trend["above_sma50"] and sector_rotation["rotation"] == "BEARISH":
            regime = "BEAR"
            confidence = 85
        
        # BULL: VIX <20, SPY above both SMAs, offensive sectors leading
        elif vix < 20 and spy_trend["above_sma50"] and spy_trend["above_sma200"] and sector_rotation["rotation"] == "BULLISH":
            regime = "BULL"
            confidence = 90
        
        # RECOVERY: VIX declining from high, SPY above SMA50 but below SMA200
        elif vix < 25 and spy_trend["above_sma50"] and not spy_trend["above_sma200"]:
            regime = "RECOVERY"
            confidence = 75
        
        # SIDEWAYS: VIX 15-25, mixed signals
        else:
            regime = "SIDEWAYS"
            confidence = 60
        
        result = {
            "regime": regime,
            "confidence": confidence,
            "vix": vix,
            "spy_above_sma50": spy_trend["above_sma50"],
            "spy_above_sma200": spy_trend["above_sma200"],
            "sector_rotation": sector_rotation["rotation"],
            "market_breadth": breadth["breadth"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache result
        _REGIME_CACHE["regime"] = result
        _REGIME_CACHE["timestamp"] = datetime.utcnow().timestamp()
        
        LOGGER.info(f"Market Regime: {regime} (confidence {confidence}%, VIX {vix:.1f})")
        
        return result
        
    except Exception as e:
        LOGGER.error(f"Regime detection failed: {e}")
        return {
            "regime": "SIDEWAYS",
            "confidence": 50,
            "vix": 20.0,
            "spy_above_sma50": True,
            "spy_above_sma200": True,
            "sector_rotation": "NEUTRAL",
            "market_breadth": "NEUTRAL",
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# REGIME SCHEDULER
# ============================================================================

async def regime_detector_loop():
    """
    Background loop to continuously update market regime
    """
    LOGGER.info("🚀 Market Regime Detector: STARTED")
    
    while True:
        try:
            regime = await detect_market_regime()
            LOGGER.info(f"📊 Market Regime: {regime['regime']} ({regime['confidence']}%)")
            
            # Update every 5 minutes
            await asyncio.sleep(300)
            
        except Exception as e:
            LOGGER.error(f"Regime detector loop error: {e}")
            await asyncio.sleep(60)
