"""
Ghost Protocol - News Analyzer
Get recent news and sentiment for each symbol before predicting
"""

import os
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# API Keys
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Sentiment keywords
BULLISH_KEYWORDS = [
    "surge", "soar", "rally", "breakthrough", "beat", "exceeds", "upgrade",
    "bullish", "buy", "outperform", "growth", "profit", "record high",
    "partnership", "acquisition", "approval", "launch", "innovation",
    "strong", "positive", "gain", "rise", "jump", "boost", "momentum"
]

BEARISH_KEYWORDS = [
    "crash", "plunge", "fall", "drop", "miss", "downgrade", "bearish",
    "sell", "underperform", "loss", "decline", "layoff", "recall",
    "investigation", "lawsuit", "fraud", "bankruptcy", "warning",
    "weak", "negative", "concern", "risk", "trouble", "struggle"
]


class NewsAnalyzer:
    """Analyze recent news for trading signals"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 1800  # 30 minutes
    
    async def get_recent_news(self, symbol: str, days: int = 3) -> List[Dict]:
        """Get recent news articles for a symbol"""
        # Check cache
        cache_key = f"news_{symbol}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        news = []
        
        # Try multiple sources in parallel
        tasks = []
        if POLYGON_API_KEY:
            tasks.append(self._polygon_news(symbol, days))
        if FINNHUB_API_KEY:
            tasks.append(self._finnhub_news(symbol, days))
        if ALPHA_VANTAGE_KEY:
            tasks.append(self._alphavantage_news(symbol))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    news.extend(result)
        
        # Deduplicate by headline
        seen = set()
        unique_news = []
        for article in news:
            headline = article.get("headline", "")[:50]
            if headline and headline not in seen:
                seen.add(headline)
                unique_news.append(article)
        
        # Cache result
        self.cache[cache_key] = (unique_news[:20], datetime.now())
        
        return unique_news[:20]  # Return top 20
    
    async def _polygon_news(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from Polygon.io"""
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker": symbol,
            "limit": 10,
            "apiKey": POLYGON_API_KEY
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [
                            {
                                "headline": r.get("title"),
                                "summary": (r.get("description", "") or "")[:500],
                                "source": r.get("publisher", {}).get("name") if isinstance(r.get("publisher"), dict) else r.get("publisher"),
                                "url": r.get("article_url"),
                                "published": r.get("published_utc"),
                                "sentiment": self._analyze_headline(r.get("title", ""))
                            }
                            for r in data.get("results", [])
                        ]
        except asyncio.TimeoutError:
            logger.warning(f"Polygon news timeout for {symbol}")
        except Exception as e:
            logger.error(f"Polygon news error for {symbol}: {e}")
        return []
    
    async def _finnhub_news(self, symbol: str, days: int) -> List[Dict]:
        """Fetch news from Finnhub"""
        url = "https://finnhub.io/api/v1/company-news"
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_API_KEY
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [
                            {
                                "headline": r.get("headline"),
                                "summary": (r.get("summary", "") or "")[:500],
                                "source": r.get("source"),
                                "url": r.get("url"),
                                "published": datetime.fromtimestamp(r.get("datetime", 0)).isoformat() if r.get("datetime") else None,
                                "sentiment": self._analyze_headline(r.get("headline", ""))
                            }
                            for r in (data[:10] if isinstance(data, list) else [])
                        ]
        except asyncio.TimeoutError:
            logger.warning(f"Finnhub news timeout for {symbol}")
        except Exception as e:
            logger.error(f"Finnhub news error for {symbol}: {e}")
        return []
    
    async def _alphavantage_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Alpha Vantage"""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": ALPHA_VANTAGE_KEY,
            "limit": 10
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [
                            {
                                "headline": r.get("title"),
                                "summary": (r.get("summary", "") or "")[:500],
                                "source": r.get("source"),
                                "url": r.get("url"),
                                "published": r.get("time_published"),
                                "sentiment": r.get("overall_sentiment_label", "neutral").lower(),
                                "sentiment_score": r.get("overall_sentiment_score", 0)
                            }
                            for r in data.get("feed", [])
                        ]
        except asyncio.TimeoutError:
            logger.warning(f"Alpha Vantage news timeout for {symbol}")
        except Exception as e:
            logger.error(f"Alpha Vantage news error for {symbol}: {e}")
        return []
    
    def _analyze_headline(self, headline: str) -> str:
        """Analyze headline sentiment using keywords"""
        if not headline:
            return "neutral"
        
        headline_lower = headline.lower()
        
        bullish_count = sum(1 for word in BULLISH_KEYWORDS if word in headline_lower)
        bearish_count = sum(1 for word in BEARISH_KEYWORDS if word in headline_lower)
        
        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        else:
            return "neutral"
    
    async def get_news_summary(self, symbol: str) -> Dict:
        """Get aggregated news summary with sentiment"""
        news = await self.get_recent_news(symbol)
        
        if not news:
            return {
                "symbol": symbol,
                "article_count": 0,
                "sentiment": "unknown",
                "sentiment_score": 0,
                "confidence_adjustment": 0,
                "key_headlines": [],
                "recommendation": "No recent news - rely on technicals"
            }
        
        # Count sentiments
        sentiments = [n.get("sentiment", "neutral") for n in news]
        bullish = sentiments.count("bullish")
        bearish = sentiments.count("bearish")
        neutral = sentiments.count("neutral")
        
        # Calculate overall sentiment
        total = len(news)
        if bullish > bearish and bullish > neutral:
            overall = "bullish"
            score = (bullish - bearish) / total
        elif bearish > bullish and bearish > neutral:
            overall = "bearish"
            score = (bearish - bullish) / total
        else:
            overall = "neutral"
            score = 0
        
        # Confidence adjustment based on news
        if abs(score) > 0.5:
            confidence_adj = 10 if score > 0 else -10
        elif abs(score) > 0.3:
            confidence_adj = 5 if score > 0 else -5
        else:
            confidence_adj = 0
        
        return {
            "symbol": symbol,
            "article_count": len(news),
            "sentiment": overall,
            "sentiment_score": round(score, 2),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "confidence_adjustment": confidence_adj,
            "key_headlines": [n.get("headline") for n in news[:5] if n.get("headline")],
            "recommendation": self._get_recommendation(overall, score)
        }
    
    def _get_recommendation(self, sentiment: str, score: float) -> str:
        """Generate recommendation based on news"""
        if sentiment == "bullish" and score > 0.5:
            return "Strong bullish news - supports BUY"
        elif sentiment == "bullish":
            return "Slightly bullish news - minor BUY support"
        elif sentiment == "bearish" and score < -0.5:
            return "Strong bearish news - supports SELL"
        elif sentiment == "bearish":
            return "Slightly bearish news - minor SELL support"
        else:
            return "Mixed/neutral news - rely on technicals"


# Singleton
_analyzer = None


def get_news_analyzer() -> NewsAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = NewsAnalyzer()
    return _analyzer


async def analyze_news(symbol: str) -> Dict:
    """Quick news analysis"""
    return await get_news_analyzer().get_news_summary(symbol)
