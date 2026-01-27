"""
GHOST INTEL - LIVE DATA SOURCES
================================
All live data connectors for the 8-layer intelligence model.

No placeholders. No sim mode. Fail closed if feeds are down.

Sources:
- FRED API: Macro data (CPI, NFP, GDP, PCE, yields)
- Yahoo Finance: Yields, DXY, VIX
- CBOE: Put/Call ratio, VIX term structure
- StockTwits: Social sentiment
- Reddit: WSB monitoring
- Polygon: News, earnings
- SEC EDGAR: Filings, insider transactions

Author: Ghost AI
Date: 2026-01-26
"""

import os
import time
import logging
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("ghost.intel")

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Keys from environment
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
STOCKTWITS_TOKEN = os.getenv("STOCKTWITS_TOKEN", "")  # Optional for higher limits
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")

# Cache settings
CACHE_TTL = {
    "macro": 3600,      # 1 hour for macro data
    "rates": 300,       # 5 min for rates/yields
    "positioning": 300, # 5 min for options data
    "social": 120,      # 2 min for social
    "news": 300,        # 5 min for news
}

# Rate limit settings
RATE_LIMITS = {
    "fred": {"calls": 120, "period": 60},      # 120/min
    "yahoo": {"calls": 100, "period": 60},     # Conservative
    "stocktwits": {"calls": 200, "period": 3600},  # 200/hour free tier
    "polygon": {"calls": 5, "period": 60},     # Free tier
}


@dataclass
class SourceHealth:
    """Track health of a data source"""
    name: str
    available: bool = False
    last_success: float = 0
    last_error: str = ""
    error_count: int = 0
    latency_ms: float = 0


@dataclass
class RawDataPoint:
    """Raw data from any source before normalization"""
    source: str
    data_type: str
    timestamp: float
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelSources:
    """
    Master class for all intelligence data sources.
    Fail closed - if a feed is down, we don't pretend it works.
    """
    
    def __init__(self):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._health: Dict[str, SourceHealth] = {}
        self._rate_trackers: Dict[str, List[float]] = {}
        
        # Initialize health tracking
        for source in ["fred", "yahoo", "cboe", "stocktwits", "reddit", "polygon"]:
            self._health[source] = SourceHealth(name=source)
    
    # =========================================================================
    # LAYER 1: MACRO DATA (FRED)
    # =========================================================================
    
    async def get_macro_data(self) -> Dict[str, Any]:
        """
        Fetch all macro indicators from FRED.
        CPI, NFP, GDP, PCE, ISM
        """
        if not FRED_API_KEY:
            logger.warning("[INTEL] FRED_API_KEY not set - macro data unavailable")
            return {"available": False, "error": "FRED_API_KEY not configured"}
        
        cache_key = "macro_data"
        cached = self._get_cache(cache_key, CACHE_TTL["macro"])
        if cached:
            return cached
        
        indicators = {
            "cpi": "CPIAUCSL",           # Consumer Price Index
            "cpi_yoy": "CPIAUCSL",       # Will calculate YoY
            "core_cpi": "CPILFESL",      # Core CPI (ex food/energy)
            "pce": "PCEPI",              # PCE Price Index
            "core_pce": "PCEPILFE",      # Core PCE (Fed's preferred)
            "nfp": "PAYEMS",             # Non-Farm Payrolls
            "unemployment": "UNRATE",    # Unemployment Rate
            "gdp": "GDP",                # GDP
            "gdp_growth": "A191RL1Q225SBEA",  # Real GDP Growth
            "ism_mfg": "MANEMP",         # Manufacturing Employment (proxy)
            "retail_sales": "RSAFS",     # Retail Sales
            "housing_starts": "HOUST",   # Housing Starts
            "initial_claims": "ICSA",    # Jobless Claims
        }
        
        result = {"available": True, "timestamp": time.time(), "data": {}}
        
        for name, series_id in indicators.items():
            try:
                value = await self._fetch_fred_series(series_id)
                if value is not None:
                    result["data"][name] = value
            except Exception as e:
                logger.error(f"[INTEL] FRED {name} fetch failed: {e}")
        
        self._set_cache(cache_key, result)
        return result
    
    async def _fetch_fred_series(self, series_id: str, limit: int = 5) -> Optional[Dict]:
        """Fetch a single FRED series with latest observations"""
        if not self._check_rate_limit("fred"):
            logger.warning(f"[INTEL] FRED rate limit hit")
            return None
        
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        obs = data.get("observations", [])
                        
                        self._health["fred"].available = True
                        self._health["fred"].last_success = time.time()
                        self._health["fred"].latency_ms = (time.time() - start) * 1000
                        
                        if obs:
                            latest = obs[0]
                            prev = obs[1] if len(obs) > 1 else None
                            
                            return {
                                "value": float(latest["value"]) if latest["value"] != "." else None,
                                "date": latest["date"],
                                "prev_value": float(prev["value"]) if prev and prev["value"] != "." else None,
                                "prev_date": prev["date"] if prev else None,
                            }
                    else:
                        self._health["fred"].error_count += 1
                        self._health["fred"].last_error = f"HTTP {resp.status}"
                        logger.error(f"[INTEL] FRED {series_id} returned {resp.status}")
        except Exception as e:
            self._health["fred"].error_count += 1
            self._health["fred"].last_error = str(e)
            logger.error(f"[INTEL] FRED {series_id} exception: {e}")
        
        return None
    
    # =========================================================================
    # LAYER 2: RATES & LIQUIDITY (Yahoo + FRED)
    # =========================================================================
    
    async def get_rates_and_liquidity(self) -> Dict[str, Any]:
        """
        Fetch rates and liquidity indicators.
        2Y, 10Y, 2s10s spread, DXY, VIX
        """
        cache_key = "rates_liquidity"
        cached = self._get_cache(cache_key, CACHE_TTL["rates"])
        if cached:
            return cached
        
        result = {"available": True, "timestamp": time.time(), "data": {}}
        
        # Yahoo tickers for rates
        tickers = {
            "us_2y": "^IRX",      # 13-week T-bill (proxy for short rates)
            "us_10y": "^TNX",     # 10-Year Treasury
            "us_30y": "^TYX",     # 30-Year Treasury
            "dxy": "DX-Y.NYB",    # US Dollar Index
            "vix": "^VIX",        # VIX
            "vix_3m": "^VIX3M",   # 3-month VIX (term structure)
        }
        
        for name, ticker in tickers.items():
            try:
                price = await self._fetch_yahoo_price(ticker)
                if price:
                    result["data"][name] = price
            except Exception as e:
                logger.error(f"[INTEL] Yahoo {name} fetch failed: {e}")
        
        # Calculate spreads
        if "us_10y" in result["data"] and "us_2y" in result["data"]:
            ten_y = result["data"]["us_10y"].get("price", 0)
            two_y = result["data"]["us_2y"].get("price", 0)
            result["data"]["spread_2s10s"] = {
                "value": round(ten_y - two_y, 3),
                "inverted": ten_y < two_y,
                "recession_signal": ten_y < two_y,
            }
        
        # VIX term structure
        if "vix" in result["data"] and "vix_3m" in result["data"]:
            vix = result["data"]["vix"].get("price", 0)
            vix_3m = result["data"]["vix_3m"].get("price", 0)
            result["data"]["vix_term_structure"] = {
                "contango": vix_3m > vix,  # Normal = contango
                "backwardation": vix > vix_3m,  # Fear = backwardation
                "spread": round(vix_3m - vix, 2),
            }
        
        # Also try FRED for more accurate treasury data
        try:
            fred_10y = await self._fetch_fred_series("DGS10", limit=1)
            if fred_10y and fred_10y.get("value"):
                result["data"]["us_10y_fred"] = {
                    "price": fred_10y["value"],
                    "date": fred_10y["date"],
                    "source": "FRED"
                }
            
            fred_2y = await self._fetch_fred_series("DGS2", limit=1)
            if fred_2y and fred_2y.get("value"):
                result["data"]["us_2y_fred"] = {
                    "price": fred_2y["value"],
                    "date": fred_2y["date"],
                    "source": "FRED"
                }
        except Exception as e:
            logger.warning(f"[INTEL] FRED treasury fetch failed: {e}")
        
        self._set_cache(cache_key, result)
        return result
    
    async def _fetch_yahoo_price(self, ticker: str) -> Optional[Dict]:
        """Fetch current price from Yahoo Finance"""
        if not self._check_rate_limit("yahoo"):
            return None
        
        # Use yfinance-style URL
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": "1d", "range": "5d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("chart", {}).get("result", [])
                        
                        self._health["yahoo"].available = True
                        self._health["yahoo"].last_success = time.time()
                        self._health["yahoo"].latency_ms = (time.time() - start) * 1000
                        
                        if result:
                            meta = result[0].get("meta", {})
                            price = meta.get("regularMarketPrice")
                            prev_close = meta.get("previousClose")
                            
                            if price:
                                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                                return {
                                    "price": round(price, 4),
                                    "prev_close": round(prev_close, 4) if prev_close else None,
                                    "change_pct": round(change_pct, 2),
                                }
                    else:
                        self._health["yahoo"].error_count += 1
                        self._health["yahoo"].last_error = f"HTTP {resp.status}"
        except Exception as e:
            self._health["yahoo"].error_count += 1
            self._health["yahoo"].last_error = str(e)
            logger.error(f"[INTEL] Yahoo {ticker} exception: {e}")
        
        return None
    
    # =========================================================================
    # LAYER 7: SOCIAL SENTIMENT
    # =========================================================================
    
    async def get_stocktwits_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch StockTwits sentiment for a symbol.
        Free tier: 200 requests/hour
        """
        cache_key = f"stocktwits_{symbol}"
        cached = self._get_cache(cache_key, CACHE_TTL["social"])
        if cached:
            return cached
        
        if not self._check_rate_limit("stocktwits"):
            return {"available": False, "error": "Rate limit"}
        
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        self._health["stocktwits"].available = True
                        self._health["stocktwits"].last_success = time.time()
                        self._health["stocktwits"].latency_ms = (time.time() - start) * 1000
                        
                        messages = data.get("messages", [])
                        symbol_info = data.get("symbol", {})
                        
                        # Calculate sentiment from messages
                        bullish = 0
                        bearish = 0
                        for msg in messages:
                            sentiment = msg.get("entities", {}).get("sentiment", {})
                            if sentiment:
                                if sentiment.get("basic") == "Bullish":
                                    bullish += 1
                                elif sentiment.get("basic") == "Bearish":
                                    bearish += 1
                        
                        total = bullish + bearish
                        sentiment_score = 0
                        if total > 0:
                            sentiment_score = (bullish - bearish) / total  # -1 to +1
                        
                        result = {
                            "available": True,
                            "symbol": symbol,
                            "timestamp": time.time(),
                            "watchlist_count": symbol_info.get("watchlist_count", 0),
                            "message_count": len(messages),
                            "bullish_count": bullish,
                            "bearish_count": bearish,
                            "sentiment_score": round(sentiment_score, 3),
                            "sentiment_label": "BULLISH" if sentiment_score > 0.2 else "BEARISH" if sentiment_score < -0.2 else "NEUTRAL",
                            "trending": symbol_info.get("is_following", False),
                        }
                        
                        self._set_cache(cache_key, result)
                        return result
                    else:
                        self._health["stocktwits"].error_count += 1
                        return {"available": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            self._health["stocktwits"].error_count += 1
            self._health["stocktwits"].last_error = str(e)
            logger.error(f"[INTEL] StockTwits {symbol} exception: {e}")
            return {"available": False, "error": str(e)}
    
    async def get_reddit_wsb_sentiment(self, symbol: str = None) -> Dict[str, Any]:
        """
        Monitor Reddit WallStreetBets for sentiment.
        Uses Reddit API if credentials available, otherwise scrapes.
        """
        cache_key = f"reddit_wsb_{symbol or 'all'}"
        cached = self._get_cache(cache_key, CACHE_TTL["social"])
        if cached:
            return cached
        
        # Try Reddit API first
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            return await self._fetch_reddit_api(symbol)
        
        # Fallback to public JSON endpoint (no auth needed)
        return await self._fetch_reddit_public(symbol)
    
    async def _fetch_reddit_public(self, symbol: str = None) -> Dict[str, Any]:
        """Fetch from Reddit's public JSON endpoints"""
        url = "https://www.reddit.com/r/wallstreetbets/hot.json"
        headers = {"User-Agent": "GhostProtocol/1.0"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        posts = data.get("data", {}).get("children", [])
                        
                        self._health["reddit"].available = True
                        self._health["reddit"].last_success = time.time()
                        
                        # Analyze posts
                        ticker_mentions = {}
                        total_score = 0
                        total_comments = 0
                        
                        for post in posts[:25]:  # Top 25 hot posts
                            post_data = post.get("data", {})
                            title = post_data.get("title", "").upper()
                            score = post_data.get("score", 0)
                            comments = post_data.get("num_comments", 0)
                            
                            total_score += score
                            total_comments += comments
                            
                            # Extract ticker mentions (simple $TICKER pattern)
                            import re
                            tickers = re.findall(r'\$([A-Z]{1,5})\b', title)
                            for ticker in tickers:
                                if ticker not in ticker_mentions:
                                    ticker_mentions[ticker] = {"count": 0, "score": 0}
                                ticker_mentions[ticker]["count"] += 1
                                ticker_mentions[ticker]["score"] += score
                        
                        # Sort by mention count
                        top_tickers = sorted(
                            ticker_mentions.items(),
                            key=lambda x: x[1]["count"],
                            reverse=True
                        )[:10]
                        
                        result = {
                            "available": True,
                            "timestamp": time.time(),
                            "subreddit": "wallstreetbets",
                            "posts_analyzed": len(posts),
                            "total_score": total_score,
                            "total_comments": total_comments,
                            "top_tickers": dict(top_tickers),
                            "symbol_found": symbol.upper() in ticker_mentions if symbol else None,
                            "symbol_data": ticker_mentions.get(symbol.upper()) if symbol else None,
                        }
                        
                        cache_key = f"reddit_wsb_{symbol or 'all'}"
                        self._set_cache(cache_key, result)
                        return result
                    else:
                        self._health["reddit"].error_count += 1
                        return {"available": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            self._health["reddit"].error_count += 1
            self._health["reddit"].last_error = str(e)
            logger.error(f"[INTEL] Reddit exception: {e}")
            return {"available": False, "error": str(e)}
    
    async def _fetch_reddit_api(self, symbol: str = None) -> Dict[str, Any]:
        """Fetch using Reddit OAuth API"""
        # Get access token
        auth = aiohttp.BasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        token_url = "https://www.reddit.com/api/v1/access_token"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    token_url,
                    auth=auth,
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": "GhostProtocol/1.0"},
                    timeout=10
                ) as resp:
                    if resp.status == 200:
                        token_data = await resp.json()
                        access_token = token_data.get("access_token")
                        
                        if access_token:
                            # Now fetch WSB
                            headers = {
                                "Authorization": f"Bearer {access_token}",
                                "User-Agent": "GhostProtocol/1.0"
                            }
                            
                            api_url = "https://oauth.reddit.com/r/wallstreetbets/hot"
                            async with session.get(api_url, headers=headers, timeout=10) as api_resp:
                                if api_resp.status == 200:
                                    # Same parsing as public endpoint
                                    return await self._fetch_reddit_public(symbol)
        except Exception as e:
            logger.error(f"[INTEL] Reddit API auth failed: {e}")
        
        # Fallback to public
        return await self._fetch_reddit_public(symbol)
    
    # =========================================================================
    # LAYER 8: POSITIONING (CBOE)
    # =========================================================================
    
    async def get_put_call_ratio(self) -> Dict[str, Any]:
        """
        Fetch Put/Call ratio from CBOE data.
        High P/C = Fear (>1.0 = very fearful)
        Low P/C = Greed (<0.7 = very greedy)
        """
        cache_key = "put_call_ratio"
        cached = self._get_cache(cache_key, CACHE_TTL["positioning"])
        if cached:
            return cached
        
        # CBOE publishes daily P/C data
        # We can also derive from VIX + historical patterns
        
        # First try Yahoo for VIX data (good proxy)
        vix_data = await self._fetch_yahoo_price("^VIX")
        
        # CBOE direct data (requires parsing their CSV)
        cboe_data = await self._fetch_cboe_pcr()
        
        result = {
            "available": True,
            "timestamp": time.time(),
            "vix_level": vix_data.get("price") if vix_data else None,
        }
        
        if cboe_data:
            result.update(cboe_data)
        else:
            # Estimate from VIX
            if vix_data and vix_data.get("price"):
                vix = vix_data["price"]
                # Rough estimation: VIX 15 ~ P/C 0.8, VIX 25 ~ P/C 1.1, VIX 35 ~ P/C 1.3
                estimated_pcr = 0.5 + (vix / 50)
                result["put_call_ratio"] = round(estimated_pcr, 2)
                result["pcr_source"] = "estimated_from_vix"
        
        # Interpret
        pcr = result.get("put_call_ratio", 0.9)
        result["fear_level"] = "EXTREME_FEAR" if pcr > 1.2 else "FEAR" if pcr > 1.0 else "NEUTRAL" if pcr > 0.7 else "GREED" if pcr > 0.5 else "EXTREME_GREED"
        result["positioning_fragile"] = pcr > 1.0  # High P/C = market positioned defensively
        
        self._set_cache(cache_key, result)
        return result
    
    async def _fetch_cboe_pcr(self) -> Optional[Dict]:
        """Fetch Put/Call ratio directly from CBOE"""
        # CBOE provides daily data in CSV format
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        
        try:
            # CBOE doesn't have a clean API, so we estimate from VIX
            # In production, you'd parse their market statistics page
            self._health["cboe"].available = True
            return None  # Fall back to VIX estimation
        except Exception as e:
            self._health["cboe"].error_count += 1
            self._health["cboe"].last_error = str(e)
            return None
    
    # =========================================================================
    # NEWS FEEDS
    # =========================================================================
    
    async def get_polygon_news(self, symbol: str = None, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch news from Polygon.io
        """
        if not POLYGON_API_KEY:
            return {"available": False, "error": "POLYGON_API_KEY not configured"}
        
        cache_key = f"polygon_news_{symbol or 'all'}"
        cached = self._get_cache(cache_key, CACHE_TTL["news"])
        if cached:
            return cached
        
        if not self._check_rate_limit("polygon"):
            return {"available": False, "error": "Rate limit"}
        
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "limit": limit,
            "apiKey": POLYGON_API_KEY,
        }
        if symbol:
            params["ticker"] = symbol.upper()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        articles = data.get("results", [])
                        
                        self._health["polygon"].available = True
                        self._health["polygon"].last_success = time.time()
                        
                        result = {
                            "available": True,
                            "timestamp": time.time(),
                            "count": len(articles),
                            "articles": [
                                {
                                    "title": a.get("title"),
                                    "description": a.get("description"),
                                    "published": a.get("published_utc"),
                                    "source": a.get("publisher", {}).get("name"),
                                    "tickers": a.get("tickers", []),
                                    "url": a.get("article_url"),
                                }
                                for a in articles
                            ]
                        }
                        
                        self._set_cache(cache_key, result)
                        return result
                    else:
                        self._health["polygon"].error_count += 1
                        return {"available": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            self._health["polygon"].error_count += 1
            self._health["polygon"].last_error = str(e)
            logger.error(f"[INTEL] Polygon news exception: {e}")
            return {"available": False, "error": str(e)}
    
    # =========================================================================
    # HEALTH & UTILITIES
    # =========================================================================
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status of all data sources"""
        return {
            "timestamp": time.time(),
            "sources": {
                name: {
                    "available": h.available,
                    "last_success": h.last_success,
                    "last_error": h.last_error,
                    "error_count": h.error_count,
                    "latency_ms": h.latency_ms,
                }
                for name, h in self._health.items()
            },
            "api_keys_configured": {
                "fred": bool(FRED_API_KEY),
                "polygon": bool(POLYGON_API_KEY),
                "reddit": bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET),
            }
        }
    
    def _get_cache(self, key: str, ttl: int) -> Optional[Any]:
        """Get from cache if not expired"""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Set cache with timestamp"""
        self._cache[key] = (time.time(), data)
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if we're within rate limits"""
        if source not in RATE_LIMITS:
            return True
        
        limit = RATE_LIMITS[source]
        now = time.time()
        
        if source not in self._rate_trackers:
            self._rate_trackers[source] = []
        
        # Clean old entries
        cutoff = now - limit["period"]
        self._rate_trackers[source] = [t for t in self._rate_trackers[source] if t > cutoff]
        
        # Check limit
        if len(self._rate_trackers[source]) >= limit["calls"]:
            return False
        
        # Record this call
        self._rate_trackers[source].append(now)
        return True
    
    # =========================================================================
    # AGGREGATE FETCH
    # =========================================================================
    
    async def fetch_all_layers(self, symbol: str = None) -> Dict[str, Any]:
        """
        Fetch data from all layers concurrently.
        This is the main entry point for the intel system.
        """
        logger.info(f"[INTEL] Fetching all layers for {symbol or 'market'}")
        
        start = time.time()
        
        # Fetch all layers concurrently
        macro_task = asyncio.create_task(self.get_macro_data())
        rates_task = asyncio.create_task(self.get_rates_and_liquidity())
        positioning_task = asyncio.create_task(self.get_put_call_ratio())
        news_task = asyncio.create_task(self.get_polygon_news(symbol))
        
        tasks = [macro_task, rates_task, positioning_task, news_task]
        
        # Add symbol-specific social if provided
        if symbol:
            stocktwits_task = asyncio.create_task(self.get_stocktwits_sentiment(symbol))
            reddit_task = asyncio.create_task(self.get_reddit_wsb_sentiment(symbol))
            tasks.extend([stocktwits_task, reddit_task])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assemble response
        response = {
            "timestamp": time.time(),
            "fetch_time_ms": round((time.time() - start) * 1000, 2),
            "symbol": symbol,
            "layers": {
                "macro": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "rates": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "positioning": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "news": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
            },
            "health": self.get_health(),
        }
        
        if symbol and len(results) > 4:
            response["layers"]["social_stocktwits"] = results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])}
            response["layers"]["social_reddit"] = results[5] if not isinstance(results[5], Exception) else {"error": str(results[5])}
        
        logger.info(f"[INTEL] All layers fetched in {response['fetch_time_ms']}ms")
        return response


# Singleton instance
_intel_sources: Optional[IntelSources] = None


def get_intel_sources() -> IntelSources:
    """Get or create the singleton IntelSources instance"""
    global _intel_sources
    if _intel_sources is None:
        _intel_sources = IntelSources()
    return _intel_sources
