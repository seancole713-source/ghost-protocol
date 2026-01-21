"""
GHOST Stage 1 Integration Module
=================================
Integrates WorldContextEngine and Market Mood Tracker into wolf_app.py

Usage in wolf_app.py:
    from core.stage1_integration import initialize_stage1, get_enhanced_context

    # On startup:
    stage1_updater = initialize_stage1()

    # In _build_ai_context():
    ctx['world_context'] = get_enhanced_context()['world_context']
    ctx['market_mood'] = get_enhanced_context()['market_mood']
"""

import asyncio
import logging
import os
from typing import Any

# Lazy imports
_context_engine = None
_last_update = 0
_update_interval = 300  # 5 minutes

LOGGER = logging.getLogger(__name__)


def initialize_stage1(
    feeds_str: str | None = None, watchlist_str: str | None = None
) -> asyncio.Task | None:
    """
    Initialize Stage 1 components.

    Args:
        feeds_str: Comma-separated RSS feed URLs (uses NEWS_MANUAL_FEEDS if None)
        watchlist_str: Comma-separated ticker symbols (uses REUTERS_SYMBOLS if None)

    Returns:
        Background task handle or None if initialization failed
    """
    global _context_engine

    try:
        from core.context_engine import WorldContextEngine

        # Default RSS feeds if none configured
        DEFAULT_FEEDS = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.reuters.com/reuters/technologyNews",
            "https://www.marketwatch.com/rss/topstories",
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
        ]

        # Get feeds from environment if not provided
        if feeds_str is None:
            reuters_feeds = os.getenv("REUTERS_FEEDS", "")
            manual_feeds = os.getenv("NEWS_MANUAL_FEEDS", "")
            feeds_str = (
                reuters_feeds + "," + manual_feeds
                if reuters_feeds and manual_feeds
                else reuters_feeds or manual_feeds
            )

        # Get watchlist from environment if not provided
        if watchlist_str is None:
            watchlist_str = os.getenv(
                "REUTERS_SYMBOLS", "WOLF,NVDA,PLTR,TSLA,AMD,AAPL,MSFT,GOOGL,META,AMZN"
            )

        # Parse feeds and watchlist
        feeds = [f.strip() for f in feeds_str.split(",") if f.strip()]
        watchlist = [s.strip() for s in watchlist_str.split(",") if s.strip()]

        # Use default feeds if none configured
        if not feeds:
            LOGGER.info("No RSS feeds configured - using defaults")
            feeds = DEFAULT_FEEDS

        # Initialize context engine
        _context_engine = WorldContextEngine(feeds, watchlist=watchlist)
        LOGGER.info(f"Stage 1 initialized: {len(feeds)} feeds, {len(watchlist)} watchlist symbols")

        # Start background updater
        task = asyncio.create_task(_background_updater())
        return task

    except Exception as e:
        LOGGER.error(f"Stage 1 initialization failed: {e}")
        return None


async def _background_updater():
    """Background task to update context and market mood."""
    global _last_update

    # CRITICAL: Delay first run to let health checks pass
    await asyncio.sleep(10)
    LOGGER.info("[STAGE1] Starting background updater (after 10s delay)...")

    while True:
        try:
            # CRITICAL: Run blocking I/O in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            
            # Update context engine (blocking RSS fetches + parsing)
            if _context_engine:
                await loop.run_in_executor(
                    None,
                    _context_engine.fetch_and_parse,
                    20  # max_per_feed
                )

            # Update market mood (blocking yfinance calls)
            from core.market_mood import update_market_mood
            
            await loop.run_in_executor(None, update_market_mood)

            _last_update = int(__import__("time").time())
            LOGGER.info("[STAGE1] Context updated successfully")

        except Exception as e:
            LOGGER.error(f"[STAGE1] Update error: {e}")

        # Sleep for 5 minutes
        await asyncio.sleep(_update_interval)


def get_enhanced_context(hours: int = 24, min_relevance: float = 0.3) -> dict[str, Any]:
    """
    Get enhanced context (world news + market mood).

    Args:
        hours: Lookback window for news context
        min_relevance: Minimum relevance score filter

    Returns:
        Dictionary with 'world_context' and 'market_mood' keys
    """
    from core.context_engine import get_context_engine
    
    result = {"world_context": {}, "market_mood": {}}
    engine = get_context_engine()

    # Get world context
    if engine:
        try:
            result["world_context"] = engine.get_recent_context(
                hours=hours, min_relevance=min_relevance
            )
        except Exception as e:
            LOGGER.error(f"Failed to get world context: {e}")
            result["world_context"] = {"error": str(e)}
    else:
        result["world_context"] = {"error": "Context engine not initialized"}

    # Get market mood
    try:
        from core.market_mood import get_market_mood, is_market_mood_stale, update_market_mood

        # Update if stale (> 12 hours old)
        if is_market_mood_stale(max_age_hours=12):
            LOGGER.info("Market mood stale, updating...")
            update_market_mood()

        mood = get_market_mood()
        result["market_mood"] = mood if mood else {"error": "Market mood not available"}

    except Exception as e:
        LOGGER.error(f"Failed to get market mood: {e}")
        result["market_mood"] = {"error": str(e)}

    return result


def get_symbol_context(symbol: str, hours: int = 24) -> dict[str, Any]:
    """
    Get context specific to a symbol.

    Args:
        symbol: Ticker symbol (e.g., 'NVDA')
        hours: Lookback window

    Returns:
        Symbol-specific context
    """
    from core.context_engine import get_context_engine
    
    engine = get_context_engine()
    if not engine:
        return {"error": "Context engine not initialized"}

    try:
        return engine.get_symbol_context(symbol, hours)
    except Exception as e:
        LOGGER.error(f"Failed to get symbol context for {symbol}: {e}")
        return {"error": str(e)}


def get_context_stats() -> dict[str, Any]:
    """Get Stage 1 statistics."""
    # Import from context_engine.py to get the actual running engine
    from core.context_engine import get_context_engine
    
    engine = get_context_engine()
    
    stats = {
        "initialized": engine is not None,
        "last_update": _last_update,
        "update_interval": _update_interval,
    }

    if engine:
        try:
            engine_stats = engine.get_stats()
            stats.update(engine_stats)
        except Exception as e:
            stats["error"] = str(e)

    return stats


def prune_old_data(keep_days: int = 7):
    """Prune old news articles."""
    from core.context_engine import get_context_engine
    
    engine = get_context_engine()
    if engine:
        try:
            deleted = engine.prune_old_articles(keep_days)
            LOGGER.info(f"Pruned {deleted} old articles (kept last {keep_days} days)")
            return deleted
        except Exception as e:
            LOGGER.error(f"Failed to prune old data: {e}")
            return 0
    return 0
