"""
Ghost Protocol - Earnings Calendar
Know when companies report earnings to avoid coin-flip trades
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Free APIs for earnings data
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")


class EarningsCalendar:
    """Track earnings dates to avoid trading near reports"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600 * 6  # 6 hours
    
    async def get_next_earnings(self, symbol: str) -> Optional[Dict]:
        """Get next earnings date for a symbol"""
        # Check cache first
        cache_key = f"earnings_{symbol}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        try:
            # Try Finnhub first
            if FINNHUB_API_KEY:
                result = await self._finnhub_earnings(symbol)
                if result:
                    self.cache[cache_key] = (result, datetime.now())
                    return result
            # Fallback to Alpha Vantage
            if ALPHA_VANTAGE_KEY:
                result = await self._alphavantage_earnings(symbol)
                if result:
                    self.cache[cache_key] = (result, datetime.now())
                    return result
            
            logger.warning(f"No earnings API key configured for {symbol}")
            return None
        except Exception as e:
            logger.error(f"Earnings lookup failed for {symbol}: {e}")
            return None
    
    async def _finnhub_earnings(self, symbol: str) -> Optional[Dict]:
        """Fetch from Finnhub API"""
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            "symbol": symbol,
            "from": datetime.now().strftime("%Y-%m-%d"),
            "to": (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
            "token": FINNHUB_API_KEY
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        calendar = data.get("earningsCalendar", [])
                        if calendar:
                            next_earnings = calendar[0]
                            return {
                                "symbol": symbol,
                                "date": next_earnings.get("date"),
                                "hour": next_earnings.get("hour", "unknown"),  # BMO, AMC
                                "estimate_eps": next_earnings.get("epsEstimate"),
                                "days_until": self._days_until(next_earnings.get("date"))
                            }
        except asyncio.TimeoutError:
            logger.warning(f"Finnhub earnings timeout for {symbol}")
        except Exception as e:
            logger.error(f"Finnhub earnings error for {symbol}: {e}")
        return None
    
    async def _alphavantage_earnings(self, symbol: str) -> Optional[Dict]:
        """Fetch from Alpha Vantage API"""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "EARNINGS_CALENDAR",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_KEY
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        # Parse CSV response
                        lines = text.strip().split("\n")
                        if len(lines) > 1:
                            parts = lines[1].split(",")
                            if len(parts) >= 3:
                                return {
                                    "symbol": symbol,
                                    "date": parts[2] if len(parts) > 2 else None,
                                    "days_until": self._days_until(parts[2]) if len(parts) > 2 else 999
                                }
        except asyncio.TimeoutError:
            logger.warning(f"Alpha Vantage earnings timeout for {symbol}")
        except Exception as e:
            logger.error(f"Alpha Vantage earnings error for {symbol}: {e}")
        return None
    
    def _days_until(self, date_str: str) -> int:
        """Calculate days until a date"""
        if not date_str:
            return 999
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d")
            return (target - datetime.now()).days
        except Exception:
            return 999
    
    async def is_earnings_risky(self, symbol: str, days_threshold: int = 3) -> Dict:
        """Check if trading near earnings is risky"""
        earnings = await self.get_next_earnings(symbol)
        
        if not earnings:
            return {
                "risky": False,
                "reason": "No earnings data available",
                "confidence_penalty": 0
            }
        
        days_until = earnings.get("days_until", 999)
        
        if days_until <= 1:
            return {
                "risky": True,
                "reason": f"Earnings TOMORROW ({earnings['date']}) - EXTREMELY RISKY",
                "confidence_penalty": 50,
                "earnings": earnings
            }
        elif days_until <= 3:
            return {
                "risky": True,
                "reason": f"Earnings in {days_until} days ({earnings['date']}) - HIGH RISK",
                "confidence_penalty": 30,
                "earnings": earnings
            }
        elif days_until <= 7:
            return {
                "risky": True,
                "reason": f"Earnings in {days_until} days - MODERATE RISK",
                "confidence_penalty": 15,
                "earnings": earnings
            }
        else:
            return {
                "risky": False,
                "reason": f"Earnings in {days_until} days - Safe to trade",
                "confidence_penalty": 0,
                "earnings": earnings
            }


# Need asyncio for TimeoutError
import asyncio

# Singleton
_calendar = None


def get_earnings_calendar() -> EarningsCalendar:
    global _calendar
    if _calendar is None:
        _calendar = EarningsCalendar()
    return _calendar


async def check_earnings_risk(symbol: str) -> Dict:
    """Quick check for earnings risk"""
    return await get_earnings_calendar().is_earnings_risky(symbol)
