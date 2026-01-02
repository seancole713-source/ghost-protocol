"""
Ghost Protocol - Seasonal Pattern Analyzer
Find recurring patterns: Does BTC pump after Christmas? Does AAPL rally before earnings?
"""

import os
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")

# Crypto symbol to CoinGecko ID mapping
CRYPTO_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
    "MATIC": "matic-network", "DOT": "polkadot", "LINK": "chainlink",
    "BCH": "bitcoin-cash", "LTC": "litecoin", "ZEC": "zcash",
    "AVAX": "avalanche-2", "SHIB": "shiba-inu", "BNB": "binancecoin",
    "TRX": "tron", "ATOM": "cosmos", "ETC": "ethereum-classic",
    "XLM": "stellar", "NEAR": "near", "UNI": "uniswap",
    "AAVE": "aave", "MKR": "maker", "FIL": "filecoin",
    "VET": "vechain", "ALGO": "algorand", "ICP": "internet-computer",
    "HBAR": "hedera-hashgraph", "METIS": "metis-token"
}

# Known crypto symbols
CRYPTO_SYMBOLS = set(CRYPTO_MAP.keys())


class SeasonalPatternAnalyzer:
    """Analyze historical seasonal patterns"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600 * 24  # 24 hours for historical data
    
    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is crypto"""
        return symbol.upper() in CRYPTO_SYMBOLS
    
    async def get_historical_prices(self, symbol: str, years: int = 3) -> List[Dict]:
        """Get historical daily prices for analysis"""
        # Check cache
        cache_key = f"historical_{symbol}_{years}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        # Use appropriate source
        if self._is_crypto(symbol):
            result = await self._crypto_historical(symbol, years)
        else:
            result = await self._polygon_historical(symbol, years)
        
        # Cache result
        if result:
            self.cache[cache_key] = (result, datetime.now())
        
        return result
    
    async def _polygon_historical(self, symbol: str, years: int) -> List[Dict]:
        """Get historical data from Polygon"""
        if not POLYGON_API_KEY:
            logger.warning("No POLYGON_API_KEY configured")
            return []
        
        from_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
        params = {"apiKey": POLYGON_API_KEY, "limit": 5000}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        return [
                            {
                                "date": datetime.fromtimestamp(r["t"]/1000).strftime("%Y-%m-%d"),
                                "month": datetime.fromtimestamp(r["t"]/1000).month,
                                "day": datetime.fromtimestamp(r["t"]/1000).day,
                                "week_of_year": datetime.fromtimestamp(r["t"]/1000).isocalendar()[1],
                                "open": r["o"],
                                "high": r["h"],
                                "low": r["l"],
                                "close": r["c"],
                                "volume": r["v"]
                            }
                            for r in results
                        ]
        except asyncio.TimeoutError:
            logger.warning(f"Polygon historical timeout for {symbol}")
        except Exception as e:
            logger.error(f"Polygon historical error for {symbol}: {e}")
        return []
    
    async def _crypto_historical(self, symbol: str, years: int) -> List[Dict]:
        """Get historical data for crypto from CoinGecko"""
        coin_id = CRYPTO_MAP.get(symbol.upper(), symbol.lower())
        days = min(years * 365, 365)  # CoinGecko limits to ~1 year for free tier
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        prices = data.get("prices", [])
                        return [
                            {
                                "date": datetime.fromtimestamp(p[0]/1000).strftime("%Y-%m-%d"),
                                "month": datetime.fromtimestamp(p[0]/1000).month,
                                "day": datetime.fromtimestamp(p[0]/1000).day,
                                "week_of_year": datetime.fromtimestamp(p[0]/1000).isocalendar()[1],
                                "close": p[1]
                            }
                            for p in prices
                        ]
                    elif resp.status == 429:
                        logger.warning(f"CoinGecko rate limit for {symbol}")
        except asyncio.TimeoutError:
            logger.warning(f"CoinGecko historical timeout for {symbol}")
        except Exception as e:
            logger.error(f"CoinGecko historical error for {symbol}: {e}")
        return []
    
    async def analyze_seasonal_patterns(self, symbol: str) -> Dict:
        """Analyze seasonal patterns for a symbol"""
        prices = await self.get_historical_prices(symbol, years=2)
        
        if len(prices) < 60:  # Need at least ~2 months of data
            return {
                "symbol": symbol,
                "has_pattern": False,
                "reason": "Insufficient historical data",
                "patterns": [],
                "confidence_adjustment": 0
            }
        
        patterns = []
        
        # 1. Month-of-year analysis
        monthly_returns = self._analyze_monthly_returns(prices)
        patterns.append(monthly_returns)
        
        # 2. Week-of-year analysis (for seasonality)
        weekly_returns = self._analyze_weekly_returns(prices)
        patterns.append(weekly_returns)
        
        # 3. Holiday patterns
        holiday_patterns = self._analyze_holiday_patterns(prices)
        patterns.append(holiday_patterns)
        
        # 4. Current period recommendation
        current_month = datetime.now().month
        current_week = datetime.now().isocalendar()[1]
        
        month_outlook = monthly_returns.get("by_month", {}).get(current_month, {})
        week_outlook = weekly_returns.get("by_week", {}).get(current_week, {})
        
        return {
            "symbol": symbol,
            "has_pattern": True,
            "patterns": patterns,
            "current_month": current_month,
            "current_week": current_week,
            "month_historical_return": month_outlook.get("avg_return", 0),
            "month_win_rate": month_outlook.get("win_rate", 50),
            "week_historical_return": week_outlook.get("avg_return", 0),
            "recommendation": self._get_seasonal_recommendation(month_outlook, week_outlook),
            "confidence_adjustment": self._calculate_seasonal_adjustment(month_outlook, week_outlook)
        }
    
    def _analyze_monthly_returns(self, prices: List[Dict]) -> Dict:
        """Analyze returns by month"""
        by_month = {i: [] for i in range(1, 13)}
        
        for i in range(1, len(prices)):
            month = prices[i].get("month")
            curr_close = prices[i].get("close")
            prev_close = prices[i-1].get("close")
            
            if month and curr_close and prev_close and prev_close > 0:
                ret = (curr_close - prev_close) / prev_close * 100
                by_month[month].append(ret)
        
        results = {}
        for month, returns in by_month.items():
            if returns:
                results[month] = {
                    "avg_return": round(statistics.mean(returns), 2),
                    "win_rate": round(len([r for r in returns if r > 0]) / len(returns) * 100, 1),
                    "sample_size": len(returns)
                }
        
        # Find best and worst months
        valid_months = [(m, data) for m, data in results.items() if data.get("sample_size", 0) > 5]
        
        best_month = max(valid_months, key=lambda x: x[1]["avg_return"], default=(0, {}))
        worst_month = min(valid_months, key=lambda x: x[1]["avg_return"], default=(0, {}))
        
        return {
            "type": "monthly",
            "by_month": results,
            "best_month": {"month": best_month[0], **best_month[1]} if best_month[1] else None,
            "worst_month": {"month": worst_month[0], **worst_month[1]} if worst_month[1] else None
        }
    
    def _analyze_weekly_returns(self, prices: List[Dict]) -> Dict:
        """Analyze returns by week of year"""
        by_week = {i: [] for i in range(1, 53)}
        
        for i in range(1, len(prices)):
            week = prices[i].get("week_of_year")
            curr_close = prices[i].get("close")
            prev_close = prices[i-1].get("close")
            
            if week and curr_close and prev_close and prev_close > 0:
                ret = (curr_close - prev_close) / prev_close * 100
                by_week[week].append(ret)
        
        results = {}
        for week, returns in by_week.items():
            if len(returns) >= 2:  # Need at least some data
                results[week] = {
                    "avg_return": round(statistics.mean(returns), 2),
                    "win_rate": round(len([r for r in returns if r > 0]) / len(returns) * 100, 1),
                    "sample_size": len(returns)
                }
        
        return {
            "type": "weekly",
            "by_week": results
        }
    
    def _analyze_holiday_patterns(self, prices: List[Dict]) -> Dict:
        """Analyze patterns around major holidays"""
        # Christmas (Dec 25), New Year (Jan 1), Thanksgiving (late Nov)
        holidays = {
            "christmas_week": {"month": 12, "days": [23, 24, 25, 26, 27]},
            "new_year_week": {"month": 1, "days": [1, 2, 3, 4, 5]},
            "post_christmas": {"month": 12, "days": [26, 27, 28, 29, 30, 31]}
        }
        
        results = {}
        for holiday_name, criteria in holidays.items():
            returns = []
            for i in range(1, len(prices)):
                price_month = prices[i].get("month")
                price_day = prices[i].get("day")
                curr_close = prices[i].get("close")
                prev_close = prices[i-1].get("close")
                
                if (price_month == criteria["month"] and 
                    price_day in criteria["days"] and
                    curr_close and prev_close and prev_close > 0):
                    ret = (curr_close - prev_close) / prev_close * 100
                    returns.append(ret)
            
            if returns:
                results[holiday_name] = {
                    "avg_return": round(statistics.mean(returns), 2),
                    "win_rate": round(len([r for r in returns if r > 0]) / len(returns) * 100, 1),
                    "sample_size": len(returns)
                }
        
        return {
            "type": "holiday",
            "patterns": results
        }
    
    def _get_seasonal_recommendation(self, month_data: Dict, week_data: Dict) -> str:
        """Generate recommendation based on seasonal data"""
        month_return = month_data.get("avg_return", 0)
        month_win = month_data.get("win_rate", 50)
        
        if month_return > 2 and month_win > 60:
            return f"BULLISH SEASON - Historically +{month_return}% this month ({month_win}% win rate)"
        elif month_return < -2 and month_win < 40:
            return f"BEARISH SEASON - Historically {month_return}% this month ({month_win}% win rate)"
        else:
            return f"NEUTRAL SEASON - Historically {month_return}% this month"
    
    def _calculate_seasonal_adjustment(self, month_data: Dict, week_data: Dict) -> int:
        """Calculate confidence adjustment based on seasonality"""
        month_return = month_data.get("avg_return", 0)
        month_win = month_data.get("win_rate", 50)
        
        if month_win > 65 and month_return > 3:
            return 10  # Strong bullish season
        elif month_win > 60 and month_return > 1:
            return 5   # Moderate bullish season
        elif month_win < 35 and month_return < -3:
            return -10  # Strong bearish season
        elif month_win < 40 and month_return < -1:
            return -5   # Moderate bearish season
        else:
            return 0


# Singleton
_analyzer = None


def get_seasonal_analyzer() -> SeasonalPatternAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SeasonalPatternAnalyzer()
    return _analyzer


async def analyze_seasonal(symbol: str) -> Dict:
    """Quick seasonal analysis"""
    return await get_seasonal_analyzer().analyze_seasonal_patterns(symbol)
