"""
GHOST Social Sentiment Monitor
==============================
24/7 live monitoring of Twitter/X and Reddit for stock sentiment.

Features:
- Twitter/X mentions and sentiment analysis
- Reddit WallStreetBets tracking
- Real-time sentiment scoring
- Trending stock mentions
- Viral signal detection

Author: Ghost AI
Date: 2025-11-17
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Cache settings
SENTIMENT_CACHE: dict[str, dict[str, Any]] = {}
CACHE_TTL_SECONDS = 600  # 10 minutes


def fetch_twitter_sentiment(symbol: str, limit: int = 20) -> dict[str, Any]:
    """
    Fetch Twitter/X sentiment for a stock symbol.
    
    Uses Twitter API v2 to search recent tweets mentioning the symbol.
    Analyzes sentiment from tweet content.
    
    Args:
        symbol: Stock ticker (e.g., 'TSLA')
        limit: Number of tweets to analyze
        
    Returns:
        Dict with sentiment_score (-1.0 to +1.0), mention_count, trending flag
    """
    cache_key = f"twitter_{symbol}"
    cached = SENTIMENT_CACHE.get(cache_key)
    
    if cached and (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]
    
    try:
        # FIXED: Implement Twitter API v2 integration
        import requests
        
        twitter_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not twitter_token:
            logger.warning("TWITTER_BEARER_TOKEN not set - Twitter sentiment unavailable")
            return {
                "ok": False,
                "error": "Twitter API not configured",
                "sentiment_score": 0.0,
                "mention_count": 0
            }
        
        # Real Twitter API v2 implementation
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {twitter_token}"}
        params = {
            "query": f"${symbol} OR #{symbol}",
            "max_results": min(limit, 100),
            "tweet.fields": "public_metrics,created_at"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code != 200:
            logger.error(f"Twitter API error: {response.status_code}")
            return {"ok": False, "error": f"HTTP {response.status_code}", "sentiment_score": 0.0, "mention_count": 0}
        
        data = response.json()
        tweets = data.get("data", [])
        
        # Simple sentiment analysis (positive/negative word counting)
        positive_words = ["bullish", "buy", "moon", "rocket", "up", "gain"]
        negative_words = ["bearish", "sell", "down", "crash", "loss", "drop"]
        
        sentiment_scores = []
        for tweet in tweets:
            text = tweet.get("text", "").lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
            sentiment_scores.append(score)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        result = {
            "ok": True,
            "symbol": symbol,
            "sentiment_score": avg_sentiment,
            "mention_count": len(tweets),
            "trending": len(tweets) > 50,
            "top_tweets": tweets[:5],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "twitter"
        }
        
        SENTIMENT_CACHE[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch Twitter sentiment for {symbol}: {e}")
        return {
            "ok": False,
            "error": str(e),
            "sentiment_score": 0.0,
            "mention_count": 0
        }


def fetch_reddit_sentiment(symbol: str, subreddit: str = "wallstreetbets") -> dict[str, Any]:
    """
    Fetch Reddit sentiment for a stock symbol from specified subreddit.
    
    Args:
        symbol: Stock ticker (e.g., 'GME')
        subreddit: Subreddit name (default: wallstreetbets)
        
    Returns:
        Dict with sentiment_score, mention_count, top_posts
    """
    cache_key = f"reddit_{symbol}_{subreddit}"
    cached = SENTIMENT_CACHE.get(cache_key)
    
    if cached and (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]
    
    try:
        # FIXED: Implement Reddit API (PRAW) integration
        import praw
        
        reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "GhostProtocol/1.0")
        
        if not reddit_client_id or not reddit_secret:
            logger.warning("Reddit API credentials not set")
            return {
                "ok": False,
                "error": "Reddit API not configured",
                "sentiment_score": 0.0,
                "mention_count": 0
            }
        
        # Initialize PRAW Reddit client
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_secret,
            user_agent=reddit_user_agent
        )
        
        # Search for posts mentioning the symbol
        subreddit_obj = reddit.subreddit(subreddit)
        posts = list(subreddit_obj.search(symbol, limit=50, time_filter="day"))
        
        # Sentiment analysis
        positive_words = ["calls", "moon", "bullish", "buy", "rocket", "gains"]
        negative_words = ["puts", "bearish", "sell", "crash", "loss", "rip"]
        
        sentiment_scores = []
        top_posts = []
        
        for post in posts:
            title = post.title.lower()
            body = post.selftext.lower() if post.selftext else ""
            full_text = title + " " + body
            
            pos = sum(1 for word in positive_words if word in full_text)
            neg = sum(1 for word in negative_words if word in full_text)
            score = (pos - neg) / max(pos + neg, 1)
            sentiment_scores.append(score)
            
            top_posts.append({
                "title": post.title,
                "score": post.score,
                "url": f"https://reddit.com{post.permalink}",
                "created": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat()
            })
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        result = {
            "ok": True,
            "symbol": symbol,
            "sentiment_score": avg_sentiment,
            "mention_count": len(posts),
            "top_posts": sorted(top_posts, key=lambda x: x["score"], reverse=True)[:5],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "reddit"
        }
        
        SENTIMENT_CACHE[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
        
    except ImportError:
        logger.error("PRAW library not installed: pip install praw")
        return {"ok": False, "error": "PRAW not installed", "sentiment_score": 0.0, "mention_count": 0}
    except Exception as e:
        logger.error(f"Failed to fetch Reddit sentiment for {symbol}: {e}")
        return {"ok": False, "error": str(e), "sentiment_score": 0.0, "mention_count": 0}
    """
    Fetch Reddit sentiment from WallStreetBets and other trading subreddits.
    
    Uses Reddit API (PRAW) to monitor posts and comments.
    
    Args:
        symbol: Stock ticker
        subreddit: Subreddit to monitor (default: wallstreetbets)
        
    Returns:
        Dict with sentiment_score, mention_count, hot_posts
    """
    cache_key = f"reddit_{symbol}_{subreddit}"
    cached = SENTIMENT_CACHE.get(cache_key)
    
    if cached and (time.time() - cached["timestamp"]) < CACHE_TTL_SECONDS:
        return cached["data"]
    
    try:
        # Reddit API (PRAW) integration
        reddit_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")
        reddit_agent = os.getenv("REDDIT_USER_AGENT", "Ghost:v1.0 (by /u/ghost_trader)")
        
        if not reddit_id or not reddit_secret:
            logger.warning("Reddit API credentials not set - using graceful fallback")
            return {
                "ok": False,
                "error": "Reddit API not configured",
                "sentiment_score": 0.0,
                "mention_count": 0
            }
        
        # Attempt PRAW import (optional dependency)
        try:
            import praw
        except ImportError:
            logger.warning("praw library not installed - Reddit sentiment unavailable")
            return {
                "ok": False,
                "error": "praw library not installed (pip install praw)",
                "sentiment_score": 0.0,
                "mention_count": 0
            }
        
        # Initialize Reddit client
        reddit = praw.Reddit(
            client_id=reddit_id,
            client_secret=reddit_secret,
            user_agent=reddit_agent
        )
        
        # Search for symbol mentions in subreddit
        mentions = []
        bullish_keywords = [
            "moon", "rocket", "diamond", "hands", "buy", "calls",
            "bullish", "pump", "squeeze", "yolo", "tendies", "apes", "hodl"
        ]
        bearish_keywords = [
            "dump", "crash", "puts", "bearish", "short", "sell", "rug",
            "scam", "dead", "bankruptcy", "baghold"
        ]
        
        # Search recent posts (limit to prevent rate limiting)
        for submission in reddit.subreddit(subreddit).search(
            symbol, time_filter="day", limit=50
        ):
            text = (submission.title + " " + submission.selftext).lower()
            
            bullish_score = sum(1 for kw in bullish_keywords if kw in text)
            bearish_score = sum(1 for kw in bearish_keywords if kw in text)
            
            # Weight by engagement
            engagement_weight = min(submission.score / 100, 5.0)
            
            mentions.append({
                "title": submission.title,
                "score": submission.score,
                "comments": submission.num_comments,
                "bullish": bullish_score * engagement_weight,
                "bearish": bearish_score * engagement_weight,
                "url": submission.url
            })
        
        if not mentions:
            result = {
                "ok": True,
                "symbol": symbol,
                "subreddit": subreddit,
                "sentiment_score": 0.0,
                "mention_count": 0,
                "top_posts": [],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            # Calculate aggregate sentiment
            total_bullish = sum(m["bullish"] for m in mentions)
            total_bearish = sum(m["bearish"] for m in mentions)
            
            # Normalize to -1.0 to +1.0 scale
            if total_bullish + total_bearish > 0:
                sentiment = (total_bullish - total_bearish) / (total_bullish + total_bearish)
            else:
                sentiment = 0.0
            
            # Get top 3 posts by engagement
            top_posts = sorted(mentions, key=lambda x: x["score"], reverse=True)[:3]
            
            result = {
                "ok": True,
                "symbol": symbol,
                "subreddit": subreddit,
                "sentiment_score": round(sentiment, 3),
                "mention_count": len(mentions),
                "top_posts": [{
                    "title": p["title"],
                    "score": p["score"],
                    "url": p["url"]
                } for p in top_posts],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        SENTIMENT_CACHE[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch Reddit sentiment for {symbol}: {e}")
        return {
            "ok": False,
            "error": str(e),
            "sentiment_score": 0.0,
            "mention_count": 0
        }


def get_combined_social_sentiment(symbol: str) -> dict[str, Any]:
    """
    Combine Twitter and Reddit sentiment into single score.
    
    Args:
        symbol: Stock ticker
        
    Returns:
        Dict with aggregated sentiment from all sources
    """
    twitter = fetch_twitter_sentiment(symbol)
    reddit = fetch_reddit_sentiment(symbol)
    
    # Calculate weighted average (50% Twitter, 50% Reddit)
    twitter_score = twitter.get("sentiment_score", 0.0) if twitter.get("ok") else 0.0
    reddit_score = reddit.get("sentiment_score", 0.0) if reddit.get("ok") else 0.0
    
    twitter_weight = 0.5 if twitter.get("ok") else 0.0
    reddit_weight = 0.5 if reddit.get("ok") else 0.0
    
    total_weight = twitter_weight + reddit_weight
    if total_weight > 0:
        combined_score = (twitter_score * twitter_weight + reddit_score * reddit_weight) / total_weight
    else:
        combined_score = 0.0
    
    # Detect viral signals (high mentions + strong sentiment)
    twitter_mentions = twitter.get("mention_count", 0)
    reddit_mentions = reddit.get("mention_count", 0)
    total_mentions = twitter_mentions + reddit_mentions
    
    is_viral = total_mentions > 100 and abs(combined_score) > 0.5
    
    return {
        "ok": True,
        "symbol": symbol,
        "combined_sentiment": combined_score,
        "twitter": twitter,
        "reddit": reddit,
        "total_mentions": total_mentions,
        "is_viral": is_viral,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def get_trending_stocks(min_mentions: int = 50) -> list[dict[str, Any]]:
    """
    Get list of stocks trending on social media.
    
    Identifies stocks with unusual mention volume or sentiment.
    
    Args:
        min_mentions: Minimum mentions to be considered trending
        
    Returns:
        List of trending stocks with sentiment scores
    """
    # Trending detection: Compare recent volume vs historical average
    try:
        import praw
    except ImportError:
        logger.warning("praw not installed - trending detection unavailable")
        return []
    
    reddit_id = os.getenv("REDDIT_CLIENT_ID")
    reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")
    
    if not reddit_id or not reddit_secret:
        return []  # Graceful fallback
    
    try:
        reddit = praw.Reddit(
            client_id=reddit_id,
            client_secret=reddit_secret,
            user_agent=os.getenv("REDDIT_USER_AGENT", "Ghost:v1.0")
        )
        
        # Track mention counts for common tickers
        common_symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "GME", "AMC"]
        trending = []
        
        for symbol in common_symbols:
            # Count recent mentions (1 hour)
            recent_count = sum(
                1 for _ in reddit.subreddit("wallstreetbets").search(
                    symbol, time_filter="hour", limit=100
                )
            )
            
            # Only include if above threshold
            if recent_count >= min_mentions:
                sentiment = fetch_reddit_sentiment(symbol)
                trending.append({
                    "symbol": symbol,
                    "mentions": recent_count,
                    "sentiment": sentiment.get("sentiment_score", 0.0)
                })
        
        # Sort by mention count
        return sorted(trending, key=lambda x: x["mentions"], reverse=True)
    
    except Exception as e:
        logger.error(f"Trending detection failed: {e}")
        return []


# Sentiment adjustment for Ghost predictions
def adjust_confidence_with_social(
    symbol: str, 
    base_confidence: float
) -> tuple[float, str]:
    """
    Adjust Ghost's prediction confidence based on social sentiment.
    
    Args:
        symbol: Stock ticker
        base_confidence: Original confidence (0.0 to 1.0)
        
    Returns:
        Tuple of (adjusted_confidence, reason)
    """
    social = get_combined_social_sentiment(symbol)
    
    if not social.get("ok"):
        return base_confidence, "No social data"
    
    sentiment = social.get("combined_sentiment", 0.0)
    mentions = social.get("total_mentions", 0)
    
    # Strong positive sentiment + high mentions = boost confidence
    if sentiment > 0.5 and mentions > 100:
        adjusted = min(1.0, base_confidence + 0.05)
        return adjusted, f"Social boost: {mentions} mentions, {sentiment:+.2f} sentiment"
    
    # Strong negative sentiment + high mentions = reduce confidence
    elif sentiment < -0.5 and mentions > 100:
        adjusted = max(0.0, base_confidence - 0.05)
        return adjusted, f"Social risk: {mentions} mentions, {sentiment:+.2f} sentiment"
    
    # Viral signal = significant boost
    elif social.get("is_viral"):
        adjusted = min(1.0, base_confidence + 0.10)
        return adjusted, f"Viral signal detected! {mentions} mentions"
    
    else:
        return base_confidence, "Neutral social sentiment"


# NEW: Market sentiment overview function for cockpit integration
def get_market_sentiment_overview():
    """Get overview of market sentiment from multiple sources."""
    try:
        headlines = []
        symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
        for symbol in symbols:
            twitter_data = fetch_twitter_sentiment(symbol, limit=10)
            if twitter_data.get("ok"):
                headlines.append({
                    "symbol": symbol,
                    "source": "twitter",
                    "sentiment": twitter_data.get("sentiment_score", 0.0),
                    "mentions": twitter_data.get("mention_count", 0)
                })
        return {"ok": True, "headlines": headlines, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Failed to get market sentiment overview: {e}")
        return {"ok": False, "headlines": [], "error": str(e)}
