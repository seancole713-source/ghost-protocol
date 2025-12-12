"""
🧠 SENTIMENT FUSION ENGINE
Aggregates sentiment from: News + Reddit + Twitter + Options Flow + Insider Trading
Provides 0-1 sentiment score for any symbol
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)

# API Keys
ALPHA_VANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Caching
_SENTIMENT_CACHE: dict[str, dict] = {}
_CACHE_TTL = 300  # 5 minutes


# ============================================================================
# NEWS SENTIMENT (Alpha Vantage)
# ============================================================================

async def get_news_sentiment(symbol: str) -> float:
    """
    Get news sentiment from Alpha Vantage (-1 to +1)
    """
    try:
        if not ALPHA_VANTAGE_KEY:
            return 0.0
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return 0.0
                
                data = await resp.json()
                
                if "feed" not in data:
                    return 0.0
                
                articles = data["feed"][:10]  # Latest 10 articles
                
                if not articles:
                    return 0.0
                
                # Calculate average sentiment
                sentiments = []
                for article in articles:
                    ticker_sentiments = article.get("ticker_sentiment", [])
                    for ts in ticker_sentiments:
                        if ts.get("ticker") == symbol:
                            score = float(ts.get("ticker_sentiment_score", 0))
                            sentiments.append(score)
                
                if not sentiments:
                    return 0.0
                
                avg_sentiment = sum(sentiments) / len(sentiments)
                return min(1.0, max(-1.0, avg_sentiment))
                
    except Exception as e:
        LOGGER.error(f"News sentiment failed for {symbol}: {e}")
        return 0.0


# ============================================================================
# REDDIT SENTIMENT (WallStreetBets)
# ============================================================================

async def get_reddit_sentiment(symbol: str) -> float:
    """
    Scrape Reddit WallStreetBets for ticker mentions & sentiment (-1 to +1)
    """
    try:
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            return 0.0
        
        # Get Reddit OAuth token
        auth = aiohttp.BasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
        
        async with aiohttp.ClientSession() as session:
            # Get access token
            token_url = "https://www.reddit.com/api/v1/access_token"
            token_data = {"grant_type": "client_credentials"}
            
            async with session.post(token_url, auth=auth, data=token_data) as resp:
                if resp.status != 200:
                    return 0.0
                
                token_json = await resp.json()
                access_token = token_json.get("access_token")
            
            # Search WallStreetBets for symbol
            headers = {
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "GhostProtocol/1.0"
            }
            
            search_url = f"https://oauth.reddit.com/r/wallstreetbets/search?q={symbol}&restrict_sr=1&sort=new&limit=50"
            
            async with session.get(search_url, headers=headers) as resp:
                if resp.status != 200:
                    return 0.0
                
                data = await resp.json()
                
                posts = data.get("data", {}).get("children", [])
                
                if not posts:
                    return 0.0
                
                # Count positive/negative keywords
                positive_keywords = ["buy", "moon", "rocket", "bullish", "calls", "long", "pump"]
                negative_keywords = ["sell", "crash", "bearish", "puts", "short", "dump"]
                
                positive_count = 0
                negative_count = 0
                
                for post in posts:
                    title = post.get("data", {}).get("title", "").lower()
                    
                    for keyword in positive_keywords:
                        if keyword in title:
                            positive_count += 1
                    
                    for keyword in negative_keywords:
                        if keyword in title:
                            negative_count += 1
                
                total = positive_count + negative_count
                if total == 0:
                    return 0.0
                
                sentiment = (positive_count - negative_count) / total
                return min(1.0, max(-1.0, sentiment))
                
    except Exception as e:
        LOGGER.error(f"Reddit sentiment failed for {symbol}: {e}")
        return 0.0


# ============================================================================
# TWITTER/X SENTIMENT
# ============================================================================

async def get_twitter_sentiment(symbol: str) -> float:
    """
    Get Twitter mentions & sentiment for ticker (-1 to +1)
    """
    try:
        if not TWITTER_BEARER_TOKEN:
            return 0.0
        
        headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
        
        # Search recent tweets
        query = f"${symbol} OR #{symbol}"
        url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=50"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return 0.0
                
                data = await resp.json()
                
                tweets = data.get("data", [])
                
                if not tweets:
                    return 0.0
                
                # Simple sentiment analysis (keyword-based)
                positive_keywords = ["buy", "moon", "bullish", "up", "long", "calls"]
                negative_keywords = ["sell", "crash", "bearish", "down", "short", "puts"]
                
                positive_count = 0
                negative_count = 0
                
                for tweet in tweets:
                    text = tweet.get("text", "").lower()
                    
                    for keyword in positive_keywords:
                        if keyword in text:
                            positive_count += 1
                    
                    for keyword in negative_keywords:
                        if keyword in text:
                            negative_count += 1
                
                total = positive_count + negative_count
                if total == 0:
                    return 0.0
                
                sentiment = (positive_count - negative_count) / total
                return min(1.0, max(-1.0, sentiment))
                
    except Exception as e:
        LOGGER.error(f"Twitter sentiment failed for {symbol}: {e}")
        return 0.0


# ============================================================================
# OPTIONS FLOW SENTIMENT (Placeholder)
# ============================================================================

async def get_options_sentiment(symbol: str) -> float:
    """
    Detect unusual options activity (bullish/bearish)
    TODO: Integrate with options flow API (e.g., Unusual Whales, FlowAlgo)
    """
    try:
        # Placeholder: Would integrate with options flow API
        # For now, return neutral
        return 0.0
        
    except Exception as e:
        LOGGER.error(f"Options sentiment failed for {symbol}: {e}")
        return 0.0


# ============================================================================
# INSIDER TRADING SENTIMENT (SEC Form 4)
# ============================================================================

async def get_insider_sentiment(symbol: str) -> float:
    """
    Check recent insider buying/selling activity (-1 to +1)
    Scrapes SEC Form 4 filings
    """
    try:
        # SEC EDGAR API
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}&type=4&dateb=&owner=only&count=10"
        
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": "GhostProtocol ghost@example.com"}
            
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return 0.0
                
                html = await resp.text()
                
                # Simple parsing: count "Purchase" vs "Sale" in recent filings
                purchase_count = html.lower().count("purchase")
                sale_count = html.lower().count("sale")
                
                total = purchase_count + sale_count
                if total == 0:
                    return 0.0
                
                sentiment = (purchase_count - sale_count) / total
                return min(1.0, max(-1.0, sentiment))
                
    except Exception as e:
        LOGGER.error(f"Insider sentiment failed for {symbol}: {e}")
        return 0.0


# ============================================================================
# AGGREGATED SENTIMENT
# ============================================================================

async def get_aggregated_sentiment(symbol: str) -> dict[str, float]:
    """
    Aggregate all sentiment sources, return normalized scores
    """
    # Check cache
    if symbol in _SENTIMENT_CACHE:
        cached = _SENTIMENT_CACHE[symbol]
        if time.time() - cached["timestamp"] < _CACHE_TTL:
            return cached["data"]
    
    try:
        # Fetch all sentiments in parallel
        results = await asyncio.gather(
            get_news_sentiment(symbol),
            get_reddit_sentiment(symbol),
            get_twitter_sentiment(symbol),
            get_options_sentiment(symbol),
            get_insider_sentiment(symbol),
            return_exceptions=True
        )
        
        news = results[0] if not isinstance(results[0], Exception) else 0.0
        reddit = results[1] if not isinstance(results[1], Exception) else 0.0
        twitter = results[2] if not isinstance(results[2], Exception) else 0.0
        options = results[3] if not isinstance(results[3], Exception) else 0.0
        insider = results[4] if not isinstance(results[4], Exception) else 0.0
        
        # Weighted aggregation
        aggregated = {
            "news_sentiment": news,
            "social_sentiment": (reddit * 0.6 + twitter * 0.4),  # Reddit weighted higher
            "options_sentiment": options,
            "insider_sentiment": insider,
            "overall_sentiment": (
                news * 0.35 +
                reddit * 0.25 +
                twitter * 0.15 +
                options * 0.15 +
                insider * 0.10
            )
        }
        
        # Cache result
        _SENTIMENT_CACHE[symbol] = {
            "timestamp": time.time(),
            "data": aggregated
        }
        
        return aggregated
        
    except Exception as e:
        LOGGER.error(f"Aggregated sentiment failed for {symbol}: {e}")
        return {
            "news_sentiment": 0.0,
            "social_sentiment": 0.0,
            "options_sentiment": 0.0,
            "insider_sentiment": 0.0,
            "overall_sentiment": 0.0
        }
