"""
Ghost Protocol - Historical Performance Analyzer
What did this stock do last year same time? How did it perform in similar market conditions?
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

# Crypto symbol to CoinGecko ID mapping
CRYPTO_MAP = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
    "MATIC": "matic-network", "DOT": "polkadot", "LINK": "chainlink",
    "BCH": "bitcoin-cash", "LTC": "litecoin", "ZEC": "zcash",
    "AVAX": "avalanche-2", "SHIB": "shiba-inu", "BNB": "binancecoin",
    "METIS": "metis-token"
}

CRYPTO_SYMBOLS = set(CRYPTO_MAP.keys())


class HistoricalAnalyzer:
    """Analyze historical performance patterns"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600 * 6  # 6 hours
    
    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is crypto"""
        return symbol.upper() in CRYPTO_SYMBOLS
    
    async def analyze_same_period_last_year(self, symbol: str) -> Dict:
        """What did this asset do same time last year?"""
        if self._is_crypto(symbol):
            return await self._crypto_same_period(symbol)
        else:
            return await self._stock_same_period(symbol)
    
    async def _stock_same_period(self, symbol: str) -> Dict:
        """Get stock data for same period last year"""
        if not POLYGON_API_KEY:
            return {"symbol": symbol, "error": "No POLYGON_API_KEY configured"}
        
        today = datetime.now()
        
        # Get data for same period last year
        last_year_start = (today - timedelta(days=365+7)).strftime("%Y-%m-%d")
        last_year_end = (today - timedelta(days=365-7)).strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{last_year_start}/{last_year_end}"
        params = {"apiKey": POLYGON_API_KEY}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        
                        if len(results) >= 2:
                            first_price = results[0]["c"]
                            last_price = results[-1]["c"]
                            change = (last_price - first_price) / first_price * 100
                            
                            return {
                                "symbol": symbol,
                                "period": f"{last_year_start} to {last_year_end}",
                                "start_price": round(first_price, 2),
                                "end_price": round(last_price, 2),
                                "change_pct": round(change, 2),
                                "direction": "UP" if change > 0 else "DOWN",
                                "insight": f"Same period last year: {'+' if change > 0 else ''}{round(change, 2)}%"
                            }
        except asyncio.TimeoutError:
            logger.warning(f"Polygon timeout for {symbol} same period")
        except Exception as e:
            logger.error(f"Historical analysis error for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "Could not fetch historical data"}
    
    async def _crypto_same_period(self, symbol: str) -> Dict:
        """Get crypto data for same period last year"""
        coin_id = CRYPTO_MAP.get(symbol.upper(), symbol.lower())
        
        # CoinGecko historical endpoint
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 365}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        prices = data.get("prices", [])
                        
                        if len(prices) >= 14:
                            # Get prices from ~1 year ago (first 7 days of data)
                            first_price = prices[0][1]
                            last_price = prices[7][1] if len(prices) > 7 else prices[-1][1]
                            change = (last_price - first_price) / first_price * 100
                            
                            return {
                                "symbol": symbol,
                                "period": "Same week last year",
                                "start_price": round(first_price, 4),
                                "end_price": round(last_price, 4),
                                "change_pct": round(change, 2),
                                "direction": "UP" if change > 0 else "DOWN",
                                "insight": f"Same period last year: {'+' if change > 0 else ''}{round(change, 2)}%"
                            }
        except asyncio.TimeoutError:
            logger.warning(f"CoinGecko timeout for {symbol} same period")
        except Exception as e:
            logger.error(f"Crypto historical analysis error for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "Could not fetch historical data"}
    
    async def analyze_ytd_performance(self, symbol: str) -> Dict:
        """Year-to-date performance"""
        if self._is_crypto(symbol):
            return await self._crypto_ytd(symbol)
        else:
            return await self._stock_ytd(symbol)
    
    async def _stock_ytd(self, symbol: str) -> Dict:
        """Stock YTD performance"""
        if not POLYGON_API_KEY:
            return {"symbol": symbol, "error": "No POLYGON_API_KEY configured"}
        
        year_start = datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{year_start}/{today}"
        params = {"apiKey": POLYGON_API_KEY}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        
                        if len(results) >= 2:
                            first_price = results[0]["c"]
                            last_price = results[-1]["c"]
                            change = (last_price - first_price) / first_price * 100
                            
                            # Calculate volatility
                            daily_returns = []
                            for i in range(1, len(results)):
                                if results[i-1]["c"] > 0:
                                    ret = (results[i]["c"] - results[i-1]["c"]) / results[i-1]["c"]
                                    daily_returns.append(ret)
                            
                            volatility = statistics.stdev(daily_returns) * 100 if len(daily_returns) > 1 else 0
                            
                            return {
                                "symbol": symbol,
                                "ytd_change_pct": round(change, 2),
                                "ytd_direction": "UP" if change > 0 else "DOWN",
                                "daily_volatility": round(volatility, 2),
                                "trading_days": len(results),
                                "trend": self._get_trend(change)
                            }
        except asyncio.TimeoutError:
            logger.warning(f"Polygon timeout for {symbol} YTD")
        except Exception as e:
            logger.error(f"YTD analysis error for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "Could not fetch YTD data"}
    
    async def _crypto_ytd(self, symbol: str) -> Dict:
        """Crypto YTD performance"""
        coin_id = CRYPTO_MAP.get(symbol.upper(), symbol.lower())
        
        # Calculate days since Jan 1
        now = datetime.now()
        jan1 = datetime(now.year, 1, 1)
        days_ytd = (now - jan1).days + 1
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": min(days_ytd, 365)}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        prices = data.get("prices", [])
                        
                        if len(prices) >= 2:
                            first_price = prices[0][1]
                            last_price = prices[-1][1]
                            change = (last_price - first_price) / first_price * 100
                            
                            # Calculate volatility
                            daily_returns = []
                            for i in range(1, len(prices)):
                                if prices[i-1][1] > 0:
                                    ret = (prices[i][1] - prices[i-1][1]) / prices[i-1][1]
                                    daily_returns.append(ret)
                            
                            volatility = statistics.stdev(daily_returns) * 100 if len(daily_returns) > 1 else 0
                            
                            return {
                                "symbol": symbol,
                                "ytd_change_pct": round(change, 2),
                                "ytd_direction": "UP" if change > 0 else "DOWN",
                                "daily_volatility": round(volatility, 2),
                                "trading_days": len(prices),
                                "trend": self._get_trend(change)
                            }
        except asyncio.TimeoutError:
            logger.warning(f"CoinGecko timeout for {symbol} YTD")
        except Exception as e:
            logger.error(f"Crypto YTD analysis error for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "Could not fetch YTD data"}
    
    def _get_trend(self, change: float) -> str:
        """Determine trend based on change percentage"""
        if change > 20:
            return "STRONG_UP"
        elif change > 5:
            return "UP"
        elif change > -5:
            return "FLAT"
        elif change > -20:
            return "DOWN"
        else:
            return "STRONG_DOWN"
    
    async def analyze_52_week_range(self, symbol: str) -> Dict:
        """52-week high/low analysis"""
        if self._is_crypto(symbol):
            return await self._crypto_52_week(symbol)
        else:
            return await self._stock_52_week(symbol)
    
    async def _stock_52_week(self, symbol: str) -> Dict:
        """Stock 52-week range"""
        if not POLYGON_API_KEY:
            return {"symbol": symbol, "error": "No POLYGON_API_KEY configured"}
        
        year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{year_ago}/{today}"
        params = {"apiKey": POLYGON_API_KEY}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        
                        if results:
                            highs = [r["h"] for r in results]
                            lows = [r["l"] for r in results]
                            current = results[-1]["c"]
                            
                            week_52_high = max(highs)
                            week_52_low = min(lows)
                            
                            if week_52_high > week_52_low:
                                range_position = (current - week_52_low) / (week_52_high - week_52_low) * 100
                            else:
                                range_position = 50
                            
                            return {
                                "symbol": symbol,
                                "current_price": round(current, 2),
                                "52_week_high": round(week_52_high, 2),
                                "52_week_low": round(week_52_low, 2),
                                "pct_from_high": round((current - week_52_high) / week_52_high * 100, 2),
                                "pct_from_low": round((current - week_52_low) / week_52_low * 100, 2),
                                "range_position": round(range_position, 1),  # 0 = at low, 100 = at high
                                "insight": self._get_range_insight(range_position)
                            }
        except asyncio.TimeoutError:
            logger.warning(f"Polygon timeout for {symbol} 52-week")
        except Exception as e:
            logger.error(f"52-week range error for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "Could not fetch 52-week data"}
    
    async def _crypto_52_week(self, symbol: str) -> Dict:
        """Crypto 52-week range"""
        coin_id = CRYPTO_MAP.get(symbol.upper(), symbol.lower())
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": 365}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        prices = data.get("prices", [])
                        
                        if prices:
                            price_values = [p[1] for p in prices]
                            current = price_values[-1]
                            
                            week_52_high = max(price_values)
                            week_52_low = min(price_values)
                            
                            if week_52_high > week_52_low:
                                range_position = (current - week_52_low) / (week_52_high - week_52_low) * 100
                            else:
                                range_position = 50
                            
                            return {
                                "symbol": symbol,
                                "current_price": round(current, 4),
                                "52_week_high": round(week_52_high, 4),
                                "52_week_low": round(week_52_low, 4),
                                "pct_from_high": round((current - week_52_high) / week_52_high * 100, 2),
                                "pct_from_low": round((current - week_52_low) / week_52_low * 100, 2),
                                "range_position": round(range_position, 1),
                                "insight": self._get_range_insight(range_position)
                            }
        except asyncio.TimeoutError:
            logger.warning(f"CoinGecko timeout for {symbol} 52-week")
        except Exception as e:
            logger.error(f"Crypto 52-week range error for {symbol}: {e}")
        
        return {"symbol": symbol, "error": "Could not fetch 52-week data"}
    
    def _get_range_insight(self, range_position: float) -> str:
        """Generate insight based on 52-week range position"""
        if range_position > 90:
            return "NEAR 52-WEEK HIGH - Potential resistance, risky to BUY"
        elif range_position > 70:
            return "UPPER RANGE - Strong momentum, but watch for pullback"
        elif range_position > 30:
            return "MID RANGE - Neutral territory"
        elif range_position > 10:
            return "LOWER RANGE - Potential value, but confirm support"
        else:
            return "NEAR 52-WEEK LOW - High risk, potential capitulation"


# Singleton
_analyzer = None


def get_historical_analyzer() -> HistoricalAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = HistoricalAnalyzer()
    return _analyzer


async def analyze_historical(symbol: str) -> Dict:
    """Full historical analysis"""
    analyzer = get_historical_analyzer()
    
    # Run all analyses in parallel
    same_period, ytd, range_52 = await asyncio.gather(
        analyzer.analyze_same_period_last_year(symbol),
        analyzer.analyze_ytd_performance(symbol),
        analyzer.analyze_52_week_range(symbol),
        return_exceptions=True
    )
    
    # Handle exceptions
    if isinstance(same_period, Exception):
        same_period = {"error": str(same_period)}
    if isinstance(ytd, Exception):
        ytd = {"error": str(ytd)}
    if isinstance(range_52, Exception):
        range_52 = {"error": str(range_52)}
    
    return {
        "symbol": symbol,
        "same_period_last_year": same_period,
        "ytd_performance": ytd,
        "52_week_range": range_52
    }
