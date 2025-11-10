"""
News API Routes
Provides access to aggregated news feeds with sentiment analysis.
"""

import logging
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Query

# Initialize router
news_router = APIRouter()
LOGGER = logging.getLogger(__name__)


async def get_news_feed(symbol: str | None = None, limit: int = 50) -> dict:
    """
    Get aggregated news feed from multiple sources.

    Args:
        symbol: Filter by ticker symbol (optional)
        limit: Maximum number of articles (1-200)

    Returns:
        {
            "news": [{"title", "summary", "url", "published", "source", "sentiment"}, ...],
            "count": int,
            "timestamp": float,
            "symbol": str | None
        }
    """
    try:
        # Try to use WorldFeedFusion if available
        try:
            from core.world_feed_fusion import WorldFeedFusion

            WorldFeedFusion()

            # Get recent articles
            articles = []
            cutoff = int(time.time()) - (24 * 3600)  # Last 24 hours

            # Query from database
            import sqlite3

            try:
                conn = sqlite3.connect("data/world_feed.db")
                cursor = conn.cursor()

                if symbol:
                    # Filter by symbol
                    cursor.execute(
                        """
                        SELECT title, summary, url, published, source_id, sentiment_score, category
                        FROM articles
                        WHERE published > ? AND symbols LIKE ?
                        ORDER BY published DESC
                        LIMIT ?
                    """,
                        (cutoff, f"%{symbol}%", limit),
                    )
                else:
                    # All articles
                    cursor.execute(
                        """
                        SELECT title, summary, url, published, source_id, sentiment_score, category
                        FROM articles
                        WHERE published > ?
                        ORDER BY published DESC
                        LIMIT ?
                    """,
                        (cutoff, limit),
                    )

                rows = cursor.fetchall()
                conn.close()

                articles = [
                    {
                        "title": row[0],
                        "summary": row[1][:200] if row[1] else "",
                        "url": row[2],
                        "published": datetime.fromtimestamp(row[3]).isoformat() if row[3] else "",
                        "source": row[4],
                        "sentiment": round(row[5], 2) if row[5] else 0.0,
                        "category": row[6] if row[6] else "general",
                    }
                    for row in rows
                ]
            except Exception as e:
                LOGGER.warning(f"Database query failed: {e}")
                articles = []

        except (ImportError, Exception) as e:
            LOGGER.warning(f"WorldFeedFusion not available: {e}, using fallback")
            articles = []

        # Fallback to RSS feeds if no articles from database
        if not articles:
            try:
                import feedparser

                feeds = [
                    ("https://feeds.reuters.com/reuters/businessNews", "Reuters"),
                    ("https://feeds.marketwatch.com/marketwatch/topstories/", "MarketWatch"),
                ]

                for feed_url, source_name in feeds:
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries[: limit // 2]:
                            # Basic symbol filtering if requested
                            if symbol:
                                content = (
                                    f"{entry.get('title', '')} {entry.get('summary', '')}".upper()
                                )
                                if symbol.upper() not in content:
                                    continue

                            articles.append(
                                {
                                    "title": entry.get("title", ""),
                                    "summary": entry.get("summary", "")[:200]
                                    if entry.get("summary")
                                    else "",
                                    "url": entry.get("link", ""),
                                    "published": entry.get("published", ""),
                                    "source": source_name,
                                    "sentiment": 0.0,  # Neutral for RSS fallback
                                    "category": "general",
                                }
                            )
                    except Exception as e:
                        LOGGER.warning(f"RSS feed {source_name} failed: {e}")
                        continue
            except ImportError:
                LOGGER.error("feedparser not available")

        # Final fallback if still no articles
        if not articles:
            articles = [
                {
                    "title": "Market Update",
                    "summary": "Real-time news feed initializing. Check back shortly.",
                    "url": "#",
                    "published": datetime.now().isoformat(),
                    "source": "Ghost Protocol",
                    "sentiment": 0.0,
                    "category": "system",
                }
            ]

        return {
            "news": articles[:limit],
            "count": len(articles),
            "timestamp": time.time(),
            "symbol": symbol,
            "status": "live" if len(articles) > 1 else "fallback",
        }

    except Exception as e:
        LOGGER.error(f"Error in get_news_feed: {e}")
        return {"news": [], "count": 0, "timestamp": time.time(), "symbol": symbol, "error": str(e)}


async def get_recent_news(symbol: str | None = None, minutes: int = 120) -> dict:
    """
    Get recent news articles within specified time window.

    Args:
        symbol: Filter by ticker symbol (optional)
        minutes: Time window in minutes (1-1440, default 120 = 2 hours)

    Returns:
        {
            "news": [...],
            "count": int,
            "timestamp": float,
            "symbol": str | None,
            "timeframe_minutes": int
        }
    """
    try:
        # Get news feed and filter by time
        result = await get_news_feed(symbol=symbol, limit=200)

        if not result.get("news"):
            return {
                "news": [],
                "count": 0,
                "timestamp": time.time(),
                "symbol": symbol,
                "timeframe_minutes": minutes,
            }

        # Filter by time window
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_articles = []

        for article in result["news"]:
            try:
                # Parse published timestamp
                published_str = article.get("published", "")
                if not published_str or published_str == "":
                    continue

                # Try parsing ISO format
                try:
                    published_dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                except Exception:
                    # Try parsing other common formats
                    from dateutil import parser

                    published_dt = parser.parse(published_str)

                # Check if within time window
                if published_dt >= cutoff:
                    recent_articles.append(article)
            except Exception as e:
                LOGGER.debug(f"Could not parse date for article: {e}")
                continue

        return {
            "news": recent_articles,
            "count": len(recent_articles),
            "timestamp": time.time(),
            "symbol": symbol,
            "timeframe_minutes": minutes,
            "status": "live",
        }

    except Exception as e:
        LOGGER.error(f"Error in get_recent_news: {e}")
        return {
            "news": [],
            "count": 0,
            "timestamp": time.time(),
            "symbol": symbol,
            "timeframe_minutes": minutes,
            "error": str(e),
        }


# Route definitions
@news_router.get("")
async def news(
    symbol: str | None = None,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of articles to return"),
):
    """
    Get aggregated news feed from multiple sources.

    - **symbol**: Filter by ticker symbol (optional)
    - **limit**: Maximum articles (1-200, default 50)

    Returns news with sentiment scores when available.
    """
    return await get_news_feed(symbol=symbol, limit=limit)


@news_router.get("/recent")
async def news_recent(
    symbol: str | None = None,
    minutes: int = Query(120, ge=1, le=1440, description="Time window in minutes"),
):
    """
    Get recent news articles within specified time window.

    - **symbol**: Filter by ticker symbol (optional)
    - **minutes**: Time window (1-1440 minutes, default 120 = 2 hours)

    Returns only articles published within the specified timeframe.
    """
    return await get_recent_news(symbol=symbol, minutes=minutes)


@news_router.get("/sentiment/{symbol}")
async def news_sentiment(symbol: str):
    """
    Get aggregated sentiment analysis for a specific symbol.

    - **symbol**: Ticker symbol (e.g., WOLF, AAPL)

    Returns sentiment scores across different timeframes.
    """
    try:
        from core.world_feed_fusion import WorldFeedFusion

        WorldFeedFusion()

        # Get sentiment aggregates
        import sqlite3

        conn = sqlite3.connect("data/world_feed.db")
        cursor = conn.cursor()

        # Calculate sentiment for different timeframes
        timeframes = {"1h": 3600, "6h": 6 * 3600, "1d": 24 * 3600, "7d": 7 * 24 * 3600}

        results = {}
        for tf_name, tf_seconds in timeframes.items():
            cutoff = int(time.time()) - tf_seconds

            cursor.execute(
                """
                SELECT AVG(sentiment_score), COUNT(*),
                       SUM(CASE WHEN sentiment_score > 0.1 THEN 1 ELSE 0 END) as bullish,
                       SUM(CASE WHEN sentiment_score < -0.1 THEN 1 ELSE 0 END) as bearish
                FROM articles
                WHERE published > ? AND symbols LIKE ?
            """,
                (cutoff, f"%{symbol}%"),
            )

            row = cursor.fetchone()
            if row and row[1] > 0:
                results[tf_name] = {
                    "avg_sentiment": round(row[0], 3) if row[0] else 0.0,
                    "article_count": row[1],
                    "bullish": row[2] or 0,
                    "bearish": row[3] or 0,
                    "neutral": row[1] - (row[2] or 0) - (row[3] or 0),
                }
            else:
                results[tf_name] = {
                    "avg_sentiment": 0.0,
                    "article_count": 0,
                    "bullish": 0,
                    "bearish": 0,
                    "neutral": 0,
                }

        conn.close()

        return {
            "symbol": symbol.upper(),
            "timestamp": time.time(),
            "sentiment": results,
            "status": "live",
        }

    except Exception as e:
        LOGGER.error(f"Error getting sentiment for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "timestamp": time.time(),
            "sentiment": {},
            "error": str(e),
        }
