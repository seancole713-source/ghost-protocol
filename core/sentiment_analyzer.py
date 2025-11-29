#!/usr/bin/env python3
"""
Ghost Protocol - Real-Time Market Sentiment Analyzer
===================================================
Aggregates sentiment from Twitter/X, Reddit, and news APIs

Catches momentum trades before they peak
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

# API Configuration
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY", "")


@dataclass
class SentimentScore:
    """Aggregated sentiment score"""
    symbol: str
    score: float  # -1.0 (bearish) to +1.0 (bullish)
    confidence: float  # 0.0-1.0
    twitter_score: float
    reddit_score: float
    news_score: float
    mentions: int
    timestamp: float


class SentimentAnalyzer:
    """Multi-source sentiment aggregator"""
    
    def __init__(self):
        self.sentiment_cache = {}  # symbol -> SentimentScore
        self.mention_history = {}  # symbol -> deque of mention counts
        
    def get_sentiment(self, symbol: str) -> SentimentScore:
        """
        Get aggregated sentiment for symbol
        
        Args:
            symbol: Stock/crypto ticker
        
        Returns:
            SentimentScore with -1.0 to +1.0 score
        """
        # Check cache (5 min TTL)
        cached = self.sentiment_cache.get(symbol)
        if cached and time.time() - cached.timestamp < 300:
            return cached
        
        # Fetch from all sources
        twitter_score, twitter_mentions = self._get_twitter_sentiment(symbol)
        reddit_score, reddit_mentions = self._get_reddit_sentiment(symbol)
        news_score, news_mentions = self._get_news_sentiment(symbol)
        
        # Weighted aggregate
        total_mentions = twitter_mentions + reddit_mentions + news_mentions
        if total_mentions > 0:
            weighted_score = (
                twitter_score * twitter_mentions * 0.4 +
                reddit_score * reddit_mentions * 0.3 +
                news_score * news_mentions * 0.3
            ) / (total_mentions * (0.4 + 0.3 + 0.3))
        else:
            weighted_score = 0.0
        
        # Confidence based on mention volume
        confidence = min(total_mentions / 100, 1.0)  # 100+ mentions = full confidence
        
        result = SentimentScore(
            symbol=symbol,
            score=weighted_score,
            confidence=confidence,
            twitter_score=twitter_score,
            reddit_score=reddit_score,
            news_score=news_score,
            mentions=total_mentions,
            timestamp=time.time()
        )
        
        self.sentiment_cache[symbol] = result
        
        # Track mention history for momentum detection
        if symbol not in self.mention_history:
            self.mention_history[symbol] = deque(maxlen=24)  # 24 hours
        self.mention_history[symbol].append(total_mentions)
        
        return result
    
    def _get_twitter_sentiment(self, symbol: str) -> tuple[float, int]:
        """
        Get Twitter/X sentiment and mention count
        
        Returns:
            (sentiment_score, mention_count)
        """
        if not TWITTER_BEARER_TOKEN:
            return 0.0, 0
        
        try:
            # Twitter API v2 recent search
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
            params = {
                "query": f"${symbol} OR #{symbol}",
                "max_results": 100,
                "tweet.fields": "public_metrics,created_at"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=5)
            if response.status_code != 200:
                logger.warning(f"Twitter API error: {response.status_code}")
                return 0.0, 0
            
            data = response.json()
            tweets = data.get("data", [])
            
            # Simple sentiment: positive - negative keywords
            positive_words = ["moon", "bullish", "buy", "pump", "rocket", "gains"]
            negative_words = ["dump", "crash", "bearish", "sell", "down", "losses"]
            
            positive_count = 0
            negative_count = 0
            
            for tweet in tweets:
                text = tweet.get("text", "").lower()
                if any(word in text for word in positive_words):
                    positive_count += 1
                if any(word in text for word in negative_words):
                    negative_count += 1
            
            total = len(tweets)
            if total == 0:
                return 0.0, 0
            
            # Normalize to -1.0 to +1.0
            score = (positive_count - negative_count) / total
            
            return score, total
            
        except Exception as e:
            logger.error(f"Twitter sentiment fetch failed: {e}")
            return 0.0, 0
    
    def _get_reddit_sentiment(self, symbol: str) -> tuple[float, int]:
        """
        Get Reddit sentiment (WSB + crypto subs)
        
        Returns:
            (sentiment_score, mention_count)
        """
        if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
            return 0.0, 0
        
        try:
            # Reddit API authentication
            auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
            data = {
                "grant_type": "client_credentials",
                "username": "ghost-protocol",
                "password": ""
            }
            headers = {"User-Agent": "GhostProtocol/1.0"}
            
            token_response = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers=headers,
                timeout=5
            )
            
            if token_response.status_code != 200:
                return 0.0, 0
            
            token = token_response.json()["access_token"]
            headers["Authorization"] = f"bearer {token}"
            
            # Search wallstreetbets and crypto subs
            subreddits = ["wallstreetbets", "CryptoCurrency", "stocks"]
            all_posts = []
            
            for subreddit in subreddits:
                search_url = f"https://oauth.reddit.com/r/{subreddit}/search"
                params = {
                    "q": symbol,
                    "sort": "new",
                    "t": "day",
                    "limit": 25
                }
                
                response = requests.get(search_url, headers=headers, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    all_posts.extend(data.get("data", {}).get("children", []))
            
            # Analyze sentiment from upvotes and comments
            bullish_score = 0
            total_score = 0
            
            for post in all_posts:
                post_data = post.get("data", {})
                upvotes = post_data.get("ups", 0)
                comments = post_data.get("num_comments", 0)
                
                # Positive sentiment = high upvotes + engagement
                if upvotes > 100:
                    bullish_score += 1
                total_score += 1
            
            if total_score == 0:
                return 0.0, 0
            
            score = (bullish_score / total_score * 2) - 1  # Normalize to -1 to +1
            
            return score, len(all_posts)
            
        except Exception as e:
            logger.error(f"Reddit sentiment fetch failed: {e}")
            return 0.0, 0
    
    def _get_news_sentiment(self, symbol: str) -> tuple[float, int]:
        """
        Get news sentiment from AlphaVantage/Benzinga
        
        Returns:
            (sentiment_score, article_count)
        """
        if not NEWS_API_KEY and not BENZINGA_API_KEY:
            return 0.0, 0
        
        try:
            # Try AlphaVantage News Sentiment API
            if NEWS_API_KEY:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": symbol,
                    "apikey": NEWS_API_KEY
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    feed = data.get("feed", [])
                    
                    if len(feed) > 0:
                        # Extract overall sentiment score
                        scores = []
                        for article in feed:
                            sentiment = article.get("overall_sentiment_score", 0)
                            scores.append(float(sentiment))
                        
                        avg_score = sum(scores) / len(scores)
                        return avg_score, len(feed)
            
            # Fallback to simple news API
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": symbol,
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": NEWS_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code != 200:
                return 0.0, 0
            
            data = response.json()
            articles = data.get("articles", [])
            
            # Simple keyword sentiment analysis
            positive_words = ["surge", "gains", "bullish", "rally", "growth", "profit"]
            negative_words = ["drop", "fall", "bearish", "loss", "decline", "crash"]
            
            positive_count = 0
            negative_count = 0
            
            for article in articles:
                title = article.get("title", "").lower()
                description = article.get("description", "").lower()
                text = f"{title} {description}"
                
                if any(word in text for word in positive_words):
                    positive_count += 1
                if any(word in text for word in negative_words):
                    negative_count += 1
            
            total = len(articles)
            if total == 0:
                return 0.0, 0
            
            score = (positive_count - negative_count) / total
            
            return score, total
            
        except Exception as e:
            logger.error(f"News sentiment fetch failed: {e}")
            return 0.0, 0
    
    def is_trending(self, symbol: str) -> bool:
        """Check if symbol is trending (viral mentions)"""
        if symbol not in self.mention_history:
            return False
        
        history = list(self.mention_history[symbol])
        if len(history) < 3:
            return False
        
        # Check if mentions doubled in last hour
        recent_avg = sum(history[-3:]) / 3
        older_avg = sum(history[:-3]) / len(history[:-3]) if len(history) > 3 else 1
        
        return recent_avg > older_avg * 2


# Global instance
_sentiment_analyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create global sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
        logger.info("✅ Sentiment analyzer initialized")
    return _sentiment_analyzer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Testing Sentiment Analyzer")
    print("=" * 60)
    
    analyzer = get_sentiment_analyzer()
    
    # Test symbols
    symbols = ["AAPL", "BTC", "TSLA"]
    
    for symbol in symbols:
        sentiment = analyzer.get_sentiment(symbol)
        
        print(f"\n{symbol}:")
        print(f"  Overall Score: {sentiment.score:+.2f}")
        print(f"  Confidence: {sentiment.confidence:.1%}")
        print(f"  Twitter: {sentiment.twitter_score:+.2f}")
        print(f"  Reddit: {sentiment.reddit_score:+.2f}")
        print(f"  News: {sentiment.news_score:+.2f}")
        print(f"  Total Mentions: {sentiment.mentions}")
        print(f"  Trending: {'🔥 YES' if analyzer.is_trending(symbol) else 'No'}")
    
    print("\n✅ Sentiment analysis complete")
