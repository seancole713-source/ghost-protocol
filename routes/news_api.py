"""Routes: news_api — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse

try:
    import httpx
except ImportError:
    httpx = None

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

# ── Also inject wolf_helpers globals (private helper functions + shared state) ─
import wolf_helpers as _wh
globals().update({k: v for k, v in vars(_wh).items() if not k.startswith("__")})
del _wh

# ── Inject all app-config globals into this route module ─────────────────────
# Mirrors wolf_app.py's pattern: provides all module-level constants that route
# handlers reference directly, without needing per-name imports.
import engines.app_config as _ac
globals().update({k: v for k, v in vars(_ac).items() if not k.startswith("__")})
del _ac

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- 7 endpoints ---

@router.get("/api/sources")
async def api_news(symbol: str = None, limit: int = 50):
    """
    Get aggregated news feed from multiple sources.

    Args:
        symbol: Filter by ticker symbol (optional)
        limit: Maximum number of articles (1-200, default 50)

    Returns news with sentiment scores when available.
    """
    try:
        import feedparser

        news_items = []
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
                        content = f"{entry.get('title', '')} {entry.get('summary', '')}".upper()
                        if symbol.upper() not in content:
                            continue

                    news_items.append(
                        {
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", "")[:200]
                            if entry.get("summary")
                            else "",
                            "url": entry.get("link", ""),
                            "published": entry.get("published", ""),
                            "source": source_name,
                            "sentiment": 0.0,
                        }
                    )
            except Exception as e:
                print(f"RSS feed {source_name} failed: {e}")
                continue

        # Fallback if no articles — return empty list, not fake news
        if not news_items:
            news_items = []

        return {
            "news": news_items[:limit],
            "count": len(news_items),
            "timestamp": time.time(),
            "symbol": symbol,
            "status": "live" if len(news_items) > 1 else "fallback",
        }
    except Exception as e:
        print(f"Error in /api/news: {e}")
        return {"news": [], "count": 0, "timestamp": time.time(), "symbol": symbol, "error": str(e)}


@router.get("/api/news/recent")
async def api_news_recent(symbol: str = None, minutes: int = 120):
    """
    Get recent news articles within specified time window.

    Args:
        symbol: Filter by ticker symbol (optional)
        minutes: Time window in minutes (1-1440, default 120 = 2 hours)

    Returns only articles published within the specified timeframe.
    """
    try:
        # Get all news and filter by time
        result = await api_news(symbol=symbol, limit=200)

        if not result.get("news"):
            return {
                "news": [],
                "count": 0,
                "timestamp": time.time(),
                "symbol": symbol,
                "timeframe_minutes": minutes,
            }

        # Filter by time window
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_articles = []

        for article in result["news"]:
            try:
                published_str = article.get("published", "")
                if not published_str:
                    continue

                # Parse published timestamp
                try:
                    from dateutil import parser

                    published_dt = parser.parse(published_str)

                    # Check if within time window
                    if published_dt >= cutoff:
                        recent_articles.append(article)
                except Exception:
                    continue
            except Exception:
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
        print(f"Error in /api/news/recent: {e}")
        return {
            "news": [],
            "count": 0,
            "timestamp": time.time(),
            "symbol": symbol,
            "timeframe_minutes": minutes,
            "error": str(e),
        }


@router.get("/source/status")
async def source_status():
    import json
    import os
    import time

    path = os.path.join(os.getcwd(), "source_status_registry.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        st = os.stat(path)
        data["server_time"] = int(time.time())
        data["file_mtime"] = int(st.st_mtime)
        data["ok"] = True
        return data
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/news/feed")
async def api_v3_news_feed(limit: int = 10):
    """
    Get news feed for cockpit news panel.

    Priority: real external news first, Ghost predictions as labeled fallback.
    """
    import asyncio

    # ── 1. Try real RSS news (fast, free, no API key) ─────────────────────
    _RSS_FEEDS = [
        ("https://feeds.marketwatch.com/marketwatch/topstories/", "MarketWatch"),
        ("https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "NYTimes"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "Yahoo Finance"),
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC"),
        ("https://feeds.bbci.co.uk/news/business/rss.xml", "BBC"),
    ]

    def _fetch_rss_sync() -> list[dict]:
        """Parse RSS feeds synchronously (runs in thread pool)."""
        import feedparser
        from email.utils import parsedate_to_datetime
        items = []
        for url, source in _RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in (feed.entries or [])[:5]:
                    # Parse timestamp
                    ts = None
                    for date_field in ("published_parsed", "updated_parsed"):
                        tp = entry.get(date_field)
                        if tp:
                            import calendar
                            ts = calendar.timegm(tp)
                            break
                    if ts is None:
                        for date_str_field in ("published", "updated"):
                            ds = entry.get(date_str_field)
                            if ds:
                                try:
                                    ts = parsedate_to_datetime(ds).timestamp()
                                except Exception:
                                    ts = time.time()
                                break
                    if ts is None:
                        ts = time.time()

                    title = entry.get("title", "").strip()
                    if not title:
                        continue

                    # Sentiment analysis from headline keywords
                    lower = title.lower()
                    sentiment = "neutral"
                    bullish_kw = (
                        "surge", "rally", "gain", "jump", "soar", "rise", "record high",
                        "bull", "boom", "up ", "upgrade", "beat", "outperform", "strong",
                        "growth", "profit", "breakthrough", "recovery", "rebound", "climb",
                        "spike", "advance", "positive", "optimis", "upbeat", "high",
                        "tops", "exceed", "record", "milestone", "momentum", "buy",
                        "all-time high", "ath", "breakout", "inflation ease", "rate cut",
                    )
                    bearish_kw = (
                        "crash", "fall", "drop", "plunge", "sink", "loss", "fear",
                        "bear", "recession", "down ", "downgrade", "miss", "underperform",
                        "weak", "decline", "deficit", "layoff", "cut", "warning", "tumble",
                        "slide", "slump", "selloff", "sell-off", "negative", "pessimis",
                        "concern", "risk", "threat", "crisis", "war", "conflict", "sanctions",
                        "tariff", "inflation rise", "rate hike", "default", "bankruptcy",
                        "collapse", "closure", "shutdown", "attack", "strike",
                        "volatil", "uncertain", "panic", "contagion", "overvalue",
                    )
                    if any(w in lower for w in bullish_kw):
                        sentiment = "bullish"
                    elif any(w in lower for w in bearish_kw):
                        sentiment = "bearish"

                    items.append({
                        "headline": title,
                        "title": title,
                        "timestamp": int(ts),
                        "source": source,
                        "sentiment": sentiment,
                        "url": entry.get("link", ""),
                    })
            except Exception:
                continue
        # Deduplicate by headline, sort newest first
        seen = set()
        deduped = []
        for it in items:
            key = it["headline"][:60].lower()
            if key not in seen:
                seen.add(key)
                deduped.append(it)
        deduped.sort(key=lambda x: x["timestamp"], reverse=True)
        return deduped

    try:
        loop = asyncio.get_event_loop()
        rss_items = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_rss_sync),
            timeout=15.0,  # Increased from 8s - RSS feeds were timing out
        )
    except Exception:
        rss_items = []

    if rss_items:
        return {
            "ok": True,
            "items": rss_items[:limit],
            "feed": rss_items[:limit],
            "count": len(rss_items[:limit]),
            "provider": "rss",
        }

    # ── 2. Fallback: Ghost predictions labeled as "Ghost AI" ──────────────
    LOGGER.warning("[NEWS] RSS feeds all failed — falling back to Ghost predictions")
    hunter_data = await api_v3_hunter_feed(limit=limit)
    if hunter_data.get("ok"):
        feed_items = hunter_data.get("feed", [])
        news_items = []
        for item in feed_items:
            news_items.append({
                "headline": item.get("title"),
                "title": item.get("title"),
                "sentiment": item.get("sentiment", "neutral"),
                "timestamp": item.get("timestamp"),
                "source": "Ghost AI (no external news)",
                "symbol": item.get("symbol"),
            })
        return {
            "ok": True,
            "items": news_items,
            "feed": news_items,
            "count": len(news_items),
            "provider": "ghost_fallback",
        }

    return {"ok": False, "items": [], "feed": [], "error": "All news sources failed"}


@router.get("/api/news/trending")
async def api_news_trending():
    """Return trending news items array. Never returns empty dict."""
    items = STATE.get("news_trending")
    if isinstance(items, list):
        return {"items": items, "ts": int(time.time() * 1000)}
    # Fallback to NEWS_CACHE if available
    try:
        news_items = NEWS_CACHE.get("items", [])
        return {"items": news_items[:10] if isinstance(news_items, list) else [], "ts": int(time.time() * 1000)}
    except Exception:
        return {"items": [], "ts": int(time.time() * 1000)}


@router.get("/api/sources/status")
async def api_sources_status():
    """Alias for /source/status (diagnostics registry)."""
    try:
        return await source_status()  # type: ignore[misc]
    except TypeError:
        # Fallback if source_status is not async in some builds
        return source_status()  # type: ignore[func-returns-value]


@router.get("/api/feeds/sources")
async def api_get_feed_sources():
    """
    World Feed Fusion - Get all RSS feed sources

    Returns:
        List of configured feed sources with status
    """
    from core.world_feed_fusion import get_feed_fusion

    try:
        fusion = get_feed_fusion()
        sources = fusion.get_sources()

        return {
            "sources": sources,
            "count": len(sources),
            "active_count": sum(1 for s in sources if s["is_active"]),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get feed sources failed: {e}", exc_info=True)
        return {"error": f"Get feed sources failed: {str(e)}"}, 500


