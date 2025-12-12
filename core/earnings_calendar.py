"""
📅 EARNINGS CALENDAR INTEGRATION
Avoid earnings surprises, track beat/miss patterns, IV crush strategies
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)

# Cache
_EARNINGS_CACHE: dict[str, dict] = {}
_CACHE_TTL = 3600  # 1 hour


# ============================================================================
# EARNINGS CALENDAR
# ============================================================================

async def get_upcoming_earnings(symbol: str) -> dict:
    """
    Get upcoming earnings date for symbol
    """
    try:
        # Check cache
        if symbol in _EARNINGS_CACHE:
            cached = _EARNINGS_CACHE[symbol]
            if datetime.utcnow().timestamp() - cached["timestamp"] < _CACHE_TTL:
                return cached["data"]
        
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return {"earnings_date": None, "days_until": 999}
        
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={symbol}&apikey={api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return {"earnings_date": None, "days_until": 999}
                
                csv_data = await resp.text()
                
                # Parse CSV (first data line after header)
                lines = csv_data.strip().split("\n")
                
                if len(lines) < 2:
                    return {"earnings_date": None, "days_until": 999}
                
                # First data line
                data_line = lines[1].split(",")
                
                if len(data_line) < 2:
                    return {"earnings_date": None, "days_until": 999}
                
                earnings_date_str = data_line[1]  # Format: YYYY-MM-DD
                earnings_date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
                
                # Days until earnings
                days_until = (earnings_date - datetime.utcnow()).days
                
                result = {
                    "earnings_date": earnings_date_str,
                    "days_until": days_until,
                    "has_earnings_soon": days_until <= 2
                }
                
                # Cache result
                _EARNINGS_CACHE[symbol] = {
                    "timestamp": datetime.utcnow().timestamp(),
                    "data": result
                }
                
                return result
                
    except Exception as e:
        LOGGER.error(f"Earnings calendar fetch failed for {symbol}: {e}")
        return {"earnings_date": None, "days_until": 999}


# ============================================================================
# EARNINGS HISTORY (Beat/Miss Patterns)
# ============================================================================

async def get_earnings_history(symbol: str) -> dict:
    """
    Get historical earnings results (beat/miss track record)
    """
    try:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return {"beat_rate": 50, "history": []}
        
        url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return {"beat_rate": 50, "history": []}
                
                data = await resp.json()
                
                quarterly_earnings = data.get("quarterlyEarnings", [])
                
                if not quarterly_earnings:
                    return {"beat_rate": 50, "history": []}
                
                # Analyze last 8 quarters
                recent_earnings = quarterly_earnings[:8]
                
                beats = 0
                misses = 0
                
                for quarter in recent_earnings:
                    reported_eps = float(quarter.get("reportedEPS", 0))
                    estimated_eps = float(quarter.get("estimatedEPS", 0))
                    
                    if reported_eps > estimated_eps:
                        beats += 1
                    elif reported_eps < estimated_eps:
                        misses += 1
                
                total = beats + misses
                beat_rate = (beats / total * 100) if total > 0 else 50
                
                return {
                    "beat_rate": beat_rate,
                    "beats": beats,
                    "misses": misses,
                    "history": recent_earnings[:4]  # Last 4 quarters
                }
                
    except Exception as e:
        LOGGER.error(f"Earnings history fetch failed for {symbol}: {e}")
        return {"beat_rate": 50, "history": []}


# ============================================================================
# IV CRUSH RISK
# ============================================================================

async def calculate_iv_crush_risk(symbol: str, days_until_earnings: int) -> dict:
    """
    Calculate IV crush risk (options premium drop after earnings)
    """
    try:
        if days_until_earnings > 7:
            return {
                "iv_crush_risk": "LOW",
                "risk_score": 20,
                "recommendation": "SAFE"
            }
        
        if days_until_earnings <= 1:
            return {
                "iv_crush_risk": "EXTREME",
                "risk_score": 95,
                "recommendation": "AVOID"
            }
        
        if days_until_earnings <= 3:
            return {
                "iv_crush_risk": "HIGH",
                "risk_score": 75,
                "recommendation": "CAUTION"
            }
        
        return {
            "iv_crush_risk": "MODERATE",
            "risk_score": 50,
            "recommendation": "MONITOR"
        }
        
    except Exception as e:
        LOGGER.error(f"IV crush risk calculation failed: {e}")
        return {"iv_crush_risk": "UNKNOWN", "risk_score": 50}


# ============================================================================
# EARNINGS STRATEGY
# ============================================================================

async def get_earnings_strategy(symbol: str) -> dict:
    """
    Recommend earnings play strategy
    """
    try:
        earnings = await get_upcoming_earnings(symbol)
        history = await get_earnings_history(symbol)
        
        days_until = earnings["days_until"]
        beat_rate = history["beat_rate"]
        
        # Avoid earnings entirely if <24hrs
        if days_until <= 1:
            return {
                "strategy": "AVOID",
                "reason": "Earnings in <24hrs - extreme volatility risk",
                "confidence": 95
            }
        
        # Earnings play if >70% beat rate and 2-5 days away
        if beat_rate >= 70 and 2 <= days_until <= 5:
            return {
                "strategy": "EARNINGS_PLAY",
                "reason": f"Strong beat rate ({beat_rate:.0f}%) - consider LONG",
                "confidence": 80
            }
        
        # Avoid if poor beat rate and earnings soon
        if beat_rate < 40 and days_until <= 5:
            return {
                "strategy": "AVOID",
                "reason": f"Poor beat rate ({beat_rate:.0f}%) - avoid earnings risk",
                "confidence": 75
            }
        
        # Safe to trade if >7 days away
        if days_until > 7:
            return {
                "strategy": "SAFE",
                "reason": "Earnings far enough away - no impact",
                "confidence": 90
            }
        
        # Default: monitor
        return {
            "strategy": "MONITOR",
            "reason": f"Earnings in {days_until} days - proceed with caution",
            "confidence": 60
        }
        
    except Exception as e:
        LOGGER.error(f"Earnings strategy failed for {symbol}: {e}")
        return {"strategy": "UNKNOWN", "confidence": 50}


# ============================================================================
# COMPREHENSIVE EARNINGS ANALYSIS
# ============================================================================

async def analyze_earnings_risk(symbol: str, asset_type: str) -> dict:
    """
    Complete earnings risk analysis
    """
    try:
        # Skip for crypto
        if asset_type == "crypto":
            return {
                "has_earnings_risk": False,
                "earnings_safe": True,
                "earnings_score": 100
            }
        
        # Get earnings data
        upcoming, history, strategy = await asyncio.gather(
            get_upcoming_earnings(symbol),
            get_earnings_history(symbol),
            get_earnings_strategy(symbol),
            return_exceptions=True
        )
        
        # Handle errors
        if isinstance(upcoming, Exception):
            upcoming = {"earnings_date": None, "days_until": 999}
        if isinstance(history, Exception):
            history = {"beat_rate": 50}
        if isinstance(strategy, Exception):
            strategy = {"strategy": "UNKNOWN"}
        
        days_until = upcoming["days_until"]
        
        # Earnings score (0-100, higher = safer)
        if days_until > 7:
            earnings_score = 100
        elif days_until > 3:
            earnings_score = 70
        elif days_until > 1:
            earnings_score = 40
        else:
            earnings_score = 10
        
        return {
            "has_earnings_risk": days_until <= 7,
            "earnings_safe": days_until > 3,
            "earnings_score": earnings_score,
            "earnings_date": upcoming["earnings_date"],
            "days_until": days_until,
            "beat_rate": history["beat_rate"],
            "strategy": strategy["strategy"],
            "recommendation": strategy["reason"]
        }
        
    except Exception as e:
        LOGGER.error(f"Earnings risk analysis failed for {symbol}: {e}")
        return {
            "has_earnings_risk": False,
            "earnings_safe": True,
            "earnings_score": 100
        }
