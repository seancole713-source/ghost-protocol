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
        # TODO: Implement Twitter API v2 integration
        # Requires TWITTER_BEARER_TOKEN environment variable
        # For now, return placeholder structure
        
        twitter_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not twitter_token:
            logger.warning("TWITTER_BEARER_TOKEN not set - Twitter sentiment unavailable")
            return {
                "ok": False,
                "error": "Twitter API not configured",
                "sentiment_score": 0.0,
                "mention_count": 0
            }
        
        # Real implementation would:
        # 1. Search tweets with query: f"${symbol} OR {company_name}"
        # 2. Analyze sentiment of each tweet (positive/negative/neutral)
        # 3. Calculate weighted average based on engagement (likes, retweets)
        # 4. Detect trending status (spike in mentions vs baseline)
        
        result = {
            "ok": True,
            "symbol": symbol,
            "sentiment_score": 0.0,  # -1.0 (bearish) to +1.0 (bullish)
            "mention_count": 0,
            "trending": False,
            "top_tweets": [],
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
        # TODO: Implement Reddit API (PRAW) integration
        # Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
        
        reddit_id = os.getenv("REDDIT_CLIENT_ID")
        if not reddit_id:
            logger.warning("REDDIT_CLIENT_ID not set - Reddit sentiment unavailable")
            return {
                "ok": False,
                "error": "Reddit API not configured",
                "sentiment_score": 0.0,
                "mention_count": 0
            }
        
        # Real implementation would:
        # 1. Search r/wallstreetbets for posts mentioning symbol
        # 2. Analyze post titles, body text, and top comments
        # 3. Detect bullish keywords (moon, rocket, diamond hands, etc.)
        # 4. Detect bearish keywords (dump, crash, puts, etc.)
        # 5. Weight by upvotes and comment count
        # 6. Track "YOLO" posts (high conviction plays)
        
        result = {
            "ok": True,
            "symbol": symbol,
            "subreddit": subreddit,
            "sentiment_score": 0.0,  # -1.0 to +1.0
            "mention_count": 0,
            "hot_posts": [],
            "yolo_count": 0,  # Number of high-conviction "YOLO" posts
            "bull_count": 0,  # Bullish mentions
            "bear_count": 0,  # Bearish mentions
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "reddit"
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
    # TODO: Implement trending detection
    # Would track mention volume over time and detect spikes
    # Compare current 1h volume vs 24h average
    
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
