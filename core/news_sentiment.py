"""
News and sentiment analysis integration.
Fetches news articles and calculates sentiment scores to enhance predictions.
"""

import os
import time
from typing import Any
import requests

# Cache for news data
_NEWS_CACHE: dict[str, dict[str, Any]] = {}
_NEWS_CACHE_TTL = 3600  # 1 hour


def fetch_news_sentiment(symbol: str, limit: int = 10) -> dict[str, Any]:
    """
    Fetch news articles and calculate sentiment score for a symbol.
    
    Uses multiple news sources:
    - Alpha Vantage News (if API key available)
    - Fallback to simple sentiment estimation
    
    Args:
        symbol: Stock/crypto ticker
        limit: Number of articles to fetch
    
    Returns:
        {
            "ok": True/False,
            "symbol": symbol,
            "articles": [...],
            "sentiment_score": -1.0 to +1.0,
            "sentiment_label": "VERY_NEGATIVE" | "NEGATIVE" | "NEUTRAL" | "POSITIVE" | "VERY_POSITIVE",
            "article_count": N,
            "timestamp": unix_ts
        }
    """
    # Check cache first
    cache_key = f"{symbol}_news"
    if cache_key in _NEWS_CACHE:
        cached = _NEWS_CACHE[cache_key]
        age = time.time() - cached.get("timestamp", 0)
        if age < _NEWS_CACHE_TTL:
            cached["cached"] = True
            return cached
    
    try:
        # Try Alpha Vantage News API
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if alpha_vantage_key:
            articles = _fetch_alpha_vantage_news(symbol, alpha_vantage_key, limit)
        else:
            # Fallback to basic news simulation (in production, use NewsAPI, Finnhub, etc.)
            articles = []
        
        # Calculate sentiment from articles
        sentiment_score = _calculate_sentiment_score(articles)
        sentiment_label = _get_sentiment_label(sentiment_score)
        
        result = {
            "ok": True,
            "symbol": symbol,
            "articles": articles[:limit],
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "article_count": len(articles),
            "timestamp": time.time()
        }
        
        # Cache result
        _NEWS_CACHE[cache_key] = result
        
        return result
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol,
            "sentiment_score": 0.0,
            "sentiment_label": "NEUTRAL"
        }


def _fetch_alpha_vantage_news(symbol: str, api_key: str, limit: int) -> list[dict[str, Any]]:
    """Fetch news from Alpha Vantage API."""
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": api_key,
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "feed" not in data:
            return []
        
        articles = []
        for item in data["feed"][:limit]:
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "published": item.get("time_published", ""),
                "sentiment_score": float(item.get("overall_sentiment_score", 0.0)),
                "sentiment_label": item.get("overall_sentiment_label", "Neutral")
            })
        
        return articles
        
    except Exception:
        return []


def _calculate_sentiment_score(articles: list[dict[str, Any]]) -> float:
    """
    Calculate aggregate sentiment score from articles.
    
    Returns:
        Score from -1.0 (very negative) to +1.0 (very positive)
    """
    if not articles:
        return 0.0
    
    # Average sentiment scores from articles
    scores = [a.get("sentiment_score", 0.0) for a in articles]
    
    if not scores:
        return 0.0
    
    avg_score = sum(scores) / len(scores)
    
    # Normalize to -1 to +1 range (Alpha Vantage returns -1 to +1)
    return max(-1.0, min(1.0, avg_score))


def _get_sentiment_label(score: float) -> str:
    """Convert sentiment score to human label."""
    if score >= 0.5:
        return "VERY_POSITIVE"
    elif score >= 0.15:
        return "POSITIVE"
    elif score <= -0.5:
        return "VERY_NEGATIVE"
    elif score <= -0.15:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def adjust_confidence_with_sentiment(
    base_confidence: float,
    sentiment_score: float,
    sentiment_weight: float = 0.1
) -> float:
    """
    Adjust prediction confidence based on news sentiment.
    
    Args:
        base_confidence: Original model confidence (0.0 - 1.0)
        sentiment_score: News sentiment (-1.0 to +1.0)
        sentiment_weight: How much sentiment affects confidence (default 10%)
    
    Returns:
        Adjusted confidence
    """
    # Positive sentiment increases confidence, negative decreases it
    adjustment = sentiment_score * sentiment_weight
    
    adjusted = base_confidence + adjustment
    
    # Clamp to valid range
    return max(0.0, min(1.0, adjusted))


def get_sentiment_indicators(symbol: str) -> dict[str, Any]:
    """
    Get sentiment indicators for display in predictions.
    
    Returns emoji and warning messages based on sentiment.
    """
    sentiment_data = fetch_news_sentiment(symbol)
    
    if not sentiment_data.get("ok"):
        return {
            "emoji": "📰",
            "message": "No recent news",
            "score": 0.0
        }
    
    score = sentiment_data.get("sentiment_score", 0.0)
    label = sentiment_data.get("sentiment_label", "NEUTRAL")
    
    # Map to emojis and messages
    emoji_map = {
        "VERY_POSITIVE": "🚀",
        "POSITIVE": "📈",
        "NEUTRAL": "📰",
        "NEGATIVE": "📉",
        "VERY_NEGATIVE": "⚠️"
    }
    
    message_map = {
        "VERY_POSITIVE": "Very positive news momentum",
        "POSITIVE": "Positive news sentiment",
        "NEUTRAL": "Neutral news sentiment",
        "NEGATIVE": "Negative news sentiment",
        "VERY_NEGATIVE": "⚠️ Strong negative news"
    }
    
    return {
        "emoji": emoji_map.get(label, "📰"),
        "message": message_map.get(label, "No sentiment data"),
        "score": score,
        "label": label
    }
