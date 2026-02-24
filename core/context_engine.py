"""
GHOST World Context Engine
===========================
Aggregates 25 news sources and extracts market context.

Features:
- RSS feed parsing (Reuters, MarketWatch, TechCrunch, etc.)
- Named entity recognition (tickers, companies, people)
- Sentiment scoring (VADER for speed)
- Relevance matching to watchlist
- Event tagging (bankruptcy, earnings, merger, etc.)

Author: Ghost AI
Date: 2025-10-05
"""

import json
import logging
import sqlite3
import time
from typing import Any
from urllib import error, request

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lazy load spacy (heavy import)
_nlp = None
_nlp_unavailable = False


def _get_nlp():
    """Lazy load spaCy NLP model."""
    global _nlp, _nlp_unavailable
    if _nlp_unavailable:
        return None
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logging.warning(f"spaCy not available: {e}. NER will be limited.")
            _nlp_unavailable = True
            return None
    return _nlp


class WorldContextEngine:
    """
    Aggregates global news and extracts market context.

    Features:
    - 25 RSS feeds (Reuters, MarketWatch, TechCrunch, etc.)
    - NER extraction (tickers, companies, people)
    - Sentiment scoring (VADER: -1.0 to +1.0)
    - Relevance matching to watchlist
    - Entity linking (CEO → Company → Ticker)
    """

    def __init__(
        self,
        feeds: list[str],
        db_path: str = "data/context_news.db",
        watchlist: list[str] | None = None,
    ):
        """
        Initialize World Context Engine.

        Args:
            feeds: List of RSS feed URLs
            db_path: SQLite database path
            watchlist: List of ticker symbols to track
        """
        self.feeds = feeds
        self.db_path = db_path
        self.watchlist = watchlist or [
            "WOLF",
            "NVDA",
            "PLTR",
            "TSLA",
            "AMD",
            "AAPL",
            "MSFT",
            "GOOGL",
            "META",
            "AMZN",
        ]

        # Initialize database connection
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_db()

        # Initialize sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()

        # Lazy load spacy
        self.nlp = None

        logging.info(
            f"WorldContextEngine initialized: {len(feeds)} feeds, {len(self.watchlist)} watchlist symbols"
        )

    def _init_db(self):
        """Create news storage table."""
        with self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS world_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts BIGINT NOT NULL,
                    source TEXT,
                    headline TEXT,
                    url TEXT UNIQUE,
                    summary TEXT,
                    sentiment REAL,
                    entities TEXT,
                    relevance REAL,
                    tags TEXT,
                    created_at BIGINT DEFAULT (strftime('%s', 'now'))
                )
            """)
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_ts ON world_news(ts)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_source ON world_news(source)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_relevance ON world_news(relevance)")
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_url ON world_news(url)")

        logging.info("Context news database initialized")

    def fetch_and_parse(self, max_per_feed: int = 20):
        """
        Fetch all RSS feeds and parse articles.

        Args:
            max_per_feed: Maximum articles to process per feed
        """
        total_new = 0
        total_skipped = 0
        feed_success = 0
        feed_failed = 0

        for feed_url in self.feeds:
            try:
                # Use feedparser directly (handles HTTP requests internally with better compatibility)
                parsed = feedparser.parse(feed_url)
                
                # Check for parsing errors
                if hasattr(parsed, 'bozo') and parsed.bozo:
                    if hasattr(parsed, 'bozo_exception'):
                        logging.warning(f"Feed parse warning {feed_url}: {parsed.bozo_exception}")
                    # Continue anyway - feedparser often sets bozo=True for minor issues
                
                if not parsed.entries:
                    logging.warning(f"No entries from feed: {feed_url} (status: {getattr(parsed, 'status', 'unknown')})")
                    feed_failed += 1
                    continue

                feed_success += 1
                feed_articles = 0
                for entry in parsed.entries[:max_per_feed]:
                    if self._process_article(entry, feed_url):
                        total_new += 1
                        feed_articles += 1
                    else:
                        total_skipped += 1
                
                if feed_articles > 0:
                    logging.debug(f"✓ {feed_url}: {feed_articles} new articles")

            except Exception as e:
                logging.error(f"Feed error {feed_url}: {e}")
                feed_failed += 1

        logging.info(
            f"Feed fetch complete: {total_new} new articles, {total_skipped} duplicates | "
            f"Feeds: {feed_success} success, {feed_failed} failed"
        )
        return {"new_articles": total_new, "skipped": total_skipped, "feeds_success": feed_success, "feeds_failed": feed_failed}

    def _process_article(self, entry, source: str) -> bool:
        """
        Extract entities, sentiment, relevance from article.

        Args:
            entry: Feedparser entry object
            source: RSS feed URL

        Returns:
            True if article was stored, False if duplicate
        """
        headline = entry.get("title", "").strip()
        summary = entry.get("summary", entry.get("description", ""))[:500].strip()
        url = entry.get("link", "").strip()

        if not headline or not url:
            return False

        # Skip duplicates (check by URL)
        cur = self.db.execute("SELECT COUNT(*) FROM world_news WHERE url=?", (url,))
        if cur.fetchone()[0] > 0:
            return False

        # Named entity extraction (lazy load spacy)
        entities = self._extract_entities(headline + " " + summary)

        # Sentiment scoring (-1.0 to +1.0)
        sentiment = self._score_sentiment(headline + " " + summary)

        # Relevance to watchlist (0.0 to 1.0)
        relevance = self._compute_relevance(entities, headline + " " + summary)

        # Event tagging
        tags = self._extract_tags(headline + " " + summary)

        # Store article
        try:
            self.db.execute(
                """
                INSERT INTO world_news (ts, source, headline, url, summary, sentiment, entities, relevance, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    int(time.time()),
                    source,
                    headline,
                    url,
                    summary,
                    sentiment,
                    json.dumps(entities),
                    relevance,
                    json.dumps(tags),
                ),
            )
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            # Duplicate URL (race condition)
            return False
        except Exception as e:
            logging.error(f"Failed to store article: {e}")
            return False

    def _extract_entities(self, text: str) -> list[str]:
        """
        Extract named entities using spaCy NER.

        Args:
            text: Article text

        Returns:
            List of entity strings
        """
        if self.nlp is None:
            self.nlp = _get_nlp()

        if not self.nlp:
            # Fallback: Simple keyword extraction from watchlist
            entities = []
            text_upper = text.upper()
            for symbol in self.watchlist:
                if symbol in text_upper:
                    entities.append(symbol)
            return entities

        try:
            doc = self.nlp(text[:1000])  # Limit to first 1000 chars
            entities = [
                ent.text for ent in doc.ents if ent.label_ in ("ORG", "PERSON", "GPE", "PRODUCT")
            ]
            return entities[:20]  # Limit to 20 entities
        except Exception as e:
            logging.error(f"NER extraction failed: {e}")
            return []

    def _score_sentiment(self, text: str) -> float:
        """
        Compute sentiment score using VADER.

        Args:
            text: Article text

        Returns:
            Sentiment score: -1.0 (negative) to +1.0 (positive)
        """
        try:
            scores = self.vader.polarity_scores(text)
            return scores["compound"]
        except Exception as e:
            logging.error(f"Sentiment scoring failed: {e}")
            return 0.0

    def _compute_relevance(self, entities: list[str], text: str) -> float:
        """
        Compute relevance to watchlist.

        Args:
            entities: Extracted named entities
            text: Article text

        Returns:
            Relevance score: 0.0 (not relevant) to 1.0 (highly relevant)
        """
        text_upper = text.upper()
        entities_upper = [e.upper() for e in entities]

        matches = 0
        for symbol in self.watchlist:
            # Direct symbol match in text
            if symbol in text_upper:
                matches += 1
            # Symbol in entities
            elif symbol in entities_upper:
                matches += 0.5

        # Normalize to 0-1 scale (3 matches = 100% relevance)
        relevance = min(1.0, matches / 3.0)
        return round(relevance, 3)

    def _extract_tags(self, text: str) -> list[str]:
        """
        Extract event keywords from article.

        Args:
            text: Article text

        Returns:
            List of event tags
        """
        keywords = {
            "bankruptcy": ["bankruptcy", "chapter 11", "restructuring", "insolvency"],
            "earnings": ["earnings", "beat", "miss", "guidance", "eps", "revenue"],
            "merger": ["merger", "acquisition", "m&a", "takeover", "buyout"],
            "product": ["launch", "product", "release", "unveil", "announce"],
            "regulatory": ["fda", "sec", "investigation", "lawsuit", "probe"],
            "upgrade": ["upgrade", "raised", "price target", "outperform"],
            "downgrade": ["downgrade", "lowered", "cut", "underperform"],
            "layoff": ["layoff", "job cut", "workforce reduction", "restructuring"],
            "ipo": ["ipo", "initial public offering", "going public", "debut"],
            "crypto": ["crypto", "bitcoin", "blockchain", "tokenization", "nft"],
        }

        tags = []
        text_lower = text.lower()

        for tag, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                tags.append(tag)

        return tags

    def get_recent_context(self, hours: int = 24, min_relevance: float = 0.3) -> dict[str, Any]:
        """
        Get summary of recent context.

        Args:
            hours: Lookback window in hours
            min_relevance: Minimum relevance score filter

        Returns:
            Dictionary with context summary
        """
        cutoff = int(time.time()) - (hours * 3600)

        # Basic stats
        cur = self.db.execute(
            """
            SELECT
                AVG(sentiment) as avg_sentiment,
                COUNT(*) as article_count,
                COUNT(DISTINCT source) as source_count,
                MIN(sentiment) as min_sentiment,
                MAX(sentiment) as max_sentiment
            FROM world_news
            WHERE ts > ? AND relevance >= ?
        """,
            (cutoff, min_relevance),
        )

        row = cur.fetchone()

        # Get top tags
        cur = self.db.execute(
            """
            SELECT tags
            FROM world_news
            WHERE ts > ? AND relevance >= ? AND tags != '[]'
            ORDER BY relevance DESC, sentiment DESC
            LIMIT 20
        """,
            (cutoff, min_relevance),
        )

        all_tags = []
        for row_tags in cur.fetchall():
            try:
                tags = json.loads(row_tags[0])
                all_tags.extend(tags)
            except Exception:
                pass

        # Count tag occurrences
        from collections import Counter

        tag_counts = Counter(all_tags)
        top_tags = [tag for tag, count in tag_counts.most_common(5)]

        # Get most relevant headlines
        cur = self.db.execute(
            """
            SELECT headline, sentiment, relevance, url
            FROM world_news
            WHERE ts > ? AND relevance >= ?
            ORDER BY relevance DESC, ABS(sentiment) DESC
            LIMIT 5
        """,
            (cutoff, min_relevance),
        )

        top_headlines = []
        for h_row in cur.fetchall():
            top_headlines.append(
                {
                    "headline": h_row[0],
                    "sentiment": round(h_row[1], 3),
                    "relevance": round(h_row[2], 3),
                    "url": h_row[3],
                }
            )

        return {
            "avg_sentiment": round(row[0] or 0.0, 3),
            "article_count": row[1] or 0,
            "source_count": row[2] or 0,
            "sentiment_range": [round(row[3] or 0.0, 3), round(row[4] or 0.0, 3)],
            "trending_events": top_tags,
            "top_headlines": top_headlines,
            "lookback_hours": hours,
            "updated_at": int(time.time()),
        }

    def get_symbol_context(self, symbol: str, hours: int = 24) -> dict[str, Any]:
        """
        Get context specific to a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'NVDA')
            hours: Lookback window

        Returns:
            Symbol-specific context
        """
        cutoff = int(time.time()) - (hours * 3600)

        # Find articles mentioning this symbol
        cur = self.db.execute(
            """
            SELECT headline, sentiment, tags, url, ts
            FROM world_news
            WHERE ts > ?
            AND (headline LIKE ? OR summary LIKE ? OR entities LIKE ?)
            ORDER BY ts DESC
            LIMIT 10
        """,
            (cutoff, f"%{symbol}%", f"%{symbol}%", f"%{symbol}%"),
        )

        articles = []
        sentiments = []
        all_tags = []

        for row in cur.fetchall():
            sentiment = row[1] or 0.0
            sentiments.append(sentiment)

            try:
                tags = json.loads(row[2])
                all_tags.extend(tags)
            except Exception:
                tags = []

            articles.append(
                {
                    "headline": row[0],
                    "sentiment": round(sentiment, 3),
                    "tags": tags,
                    "url": row[3],
                    "age_hours": round((int(time.time()) - row[4]) / 3600, 1),
                }
            )

        avg_sentiment = round(sum(sentiments) / len(sentiments), 3) if sentiments else 0.0

        from collections import Counter

        tag_counts = Counter(all_tags)
        top_events = [tag for tag, count in tag_counts.most_common(3)]

        return {
            "symbol": symbol,
            "article_count": len(articles),
            "avg_sentiment": avg_sentiment,
            "sentiment_trend": "positive"
            if avg_sentiment > 0.2
            else "negative"
            if avg_sentiment < -0.2
            else "neutral",
            "top_events": top_events,
            "recent_articles": articles,
            "lookback_hours": hours,
        }

    def prune_old_articles(self, keep_days: int = 7):
        """
        Remove articles older than keep_days.

        Args:
            keep_days: Days to retain

        Returns:
            Number of articles deleted
        """
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = self.db.execute("DELETE FROM world_news WHERE ts < ?", (cutoff,))
        self.db.commit()
        deleted = cur.rowcount
        logging.info(f"Pruned {deleted} old articles (kept last {keep_days} days)")
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get context engine statistics."""
        cur = self.db.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM world_news")
        row = cur.fetchone()
        total = row[0]
        min_ts = row[1]
        max_ts = row[2]

        span_days = (max_ts - min_ts) / 86400 if min_ts and max_ts else 0

        cur = self.db.execute(
            """
            SELECT COUNT(*) FROM world_news
            WHERE ts > ?
        """,
            (int(time.time()) - 86400,),
        )
        last_24h = cur.fetchone()[0]

        return {
            "total_articles": total,
            "span_days": round(span_days, 1),
            "articles_last_24h": last_24h,
            "feeds_count": len(self.feeds),
            "watchlist_count": len(self.watchlist),
            "db_path": self.db_path,
        }

    def refresh(self, max_per_feed: int = 20) -> int:
        """
        Refresh RSS feeds and return number of new articles.
        
        Args:
            max_per_feed: Maximum articles to process per feed
            
        Returns:
            Number of new articles added
        """
        result = self.fetch_and_parse(max_per_feed=max_per_feed)
        return result.get("new_articles", 0)

    def close(self):
        """Close database connection."""
        self.db.close()
        logging.info("WorldContextEngine closed")


# Convenience function
def create_context_engine(
    feeds_str: str | None, watchlist_str: str | None = None
) -> WorldContextEngine:
    """
    Create WorldContextEngine from comma-separated strings.

    Args:
        feeds_str: Comma-separated RSS feed URLs (optional)
        watchlist_str: Comma-separated ticker symbols (optional)

    Returns:
        WorldContextEngine instance
    """
    feeds = [f.strip() for f in feeds_str.split(",") if f.strip()] if feeds_str else []
    watchlist = (
        [s.strip() for s in watchlist_str.split(",") if s.strip()] if watchlist_str else None
    )

    return WorldContextEngine(feeds, watchlist=watchlist)


# ============================================================================
# BACKGROUND UPDATER - Required for Orchestrator Integration
# ============================================================================

# Global context engine instance for API access
_CONTEXT_ENGINE: WorldContextEngine | None = None


def get_context_engine() -> WorldContextEngine | None:
    """
    Get the active context engine instance.
    
    Returns None if context engine is not initialized or disabled.
    Used by API endpoints to query context statistics.
    """
    return _CONTEXT_ENGINE


async def start_background_updater(
    refresh_interval_minutes: int = 60,
    db_path: str = "data/context_news.db",
    watchlist: list[str] | None = None,
) -> None:
    """
    Background task that refreshes RSS feeds every N minutes.
    
    This is the missing piece that enables Stage 1 Context Engine
    to run continuously in production.
    
    Args:
        refresh_interval_minutes: How often to refresh feeds (default: 60)
        db_path: Path to context news database
        watchlist: List of tickers to track
    
    Usage in orchestrator:
        from core.context_engine import start_background_updater
        _TASKS["context_engine"] = asyncio.create_task(
            start_background_updater(refresh_interval_minutes=60)
        )
    """
    import asyncio
    import os
    
    # Default RSS feeds (25 sources) - Updated with working endpoints
    default_feeds = [
        # Financial News (Major)
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.marketwatch.com/rss/topstories",
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "https://www.cnbc.com/id/10001147/device/rss/rss.html",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        
        # Tech/Business
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://www.wired.com/feed/rss",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.engadget.com/rss.xml",
        
        # Crypto-Specific
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
        "https://cryptopotato.com/feed/",
        
        # Market Data
        "https://www.investing.com/rss/news.rss",
        "https://www.investing.com/rss/news_285.rss",  # Crypto news
        
        # Alternative/Aggregators
        "https://news.ycombinator.com/rss",
        "https://www.reddit.com/r/wallstreetbets/.rss",
        "https://www.reddit.com/r/CryptoCurrency/.rss",
        "https://www.reddit.com/r/stocks/.rss",
        
        # SEC Filings (important for earnings/events)
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=8-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
        
        # Seeking Alpha (if accessible)
        "https://seekingalpha.com/market_currents.xml",
        
        # Yahoo Finance
        "https://finance.yahoo.com/news/rssindex",
        
        # Business Insider
        "https://markets.businessinsider.com/rss/news",
        
        # Bloomberg (backup)
        "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    ]
    
    # Get watchlist from environment or use defaults
    if watchlist is None:
        watchlist_str = os.getenv("CONTEXT_WATCHLIST", "")
        watchlist = [s.strip() for s in watchlist_str.split(",") if s.strip()] if watchlist_str else None
    
    # Initialize context engine
    global _CONTEXT_ENGINE
    engine = WorldContextEngine(
        feeds=default_feeds,
        db_path=db_path,
        watchlist=watchlist,
    )
    _CONTEXT_ENGINE = engine  # Store for API access
    
    logging.info(f"🧠 Context Engine Background Updater: STARTED")
    logging.info(f"   Refresh interval: {refresh_interval_minutes} minutes")
    logging.info(f"   RSS feeds: {len(default_feeds)}")
    logging.info(f"   Watchlist: {len(engine.watchlist)} symbols")
    
    refresh_count = 0
    
    try:
        while True:
            try:
                refresh_count += 1
                logging.info(f"🔄 Context Engine: Starting refresh #{refresh_count}")
                
                # Refresh all RSS feeds
                articles_added = engine.refresh()
                
                # Prune old articles (keep last 7 days)
                deleted = engine.prune_old_articles(keep_days=7)
                
                # Get stats
                stats = engine.get_stats()
                
                logging.info(
                    f"✅ Context Engine: Refresh #{refresh_count} complete | "
                    f"Added: {articles_added}, Deleted: {deleted}, "
                    f"Total: {stats['total_articles']}, Last 24h: {stats['articles_last_24h']}"
                )
                
                # Wait for next refresh interval
                await asyncio.sleep(refresh_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logging.info("🛑 Context Engine: Background updater cancelled")
                break
            except Exception as e:
                logging.error(f"❌ Context Engine: Refresh failed: {e}", exc_info=True)
                # Continue on error, retry after 5 minutes
                await asyncio.sleep(300)
                
    finally:
        engine.close()
        logging.info("🧠 Context Engine: Background updater stopped")
