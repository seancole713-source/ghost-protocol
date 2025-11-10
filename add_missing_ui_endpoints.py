#!/usr/bin/env python3
"""
Add missing UI endpoints to wolf_app.py

This script adds the missing endpoints that the Ghost Cockpit UI expects:
- /api/agent/decisions
- /api/agent/stats
- /api/news
- /api/news/recent
- /api/snapshot
- /api/research/snapshot/{symbol}
- /api/stage5/execution/analytics
"""

# Read wolf_app.py
with open("wolf_app.py") as f:
    content = f.read()

# Find the location after /api/agent/ask (around line 14468)
# We'll add the new agent endpoints right after that

new_agent_endpoints = '''

@APP.get("/api/agent/decisions")
async def api_agent_decisions(limit: int = 20):
    """
    Get recent agent decisions/trades for the cockpit.
    Returns list of decisions with timestamp, action, symbol, confidence.
    """
    try:
        # Get decisions from agent_state or database
        decisions = []

        # Check if we have a decision log in agent_state
        if hasattr(AGENT_STATE, "decision_log") and AGENT_STATE.decision_log:
            decisions = AGENT_STATE.decision_log[-limit:]
        else:
            # Try to get from goal_tracker or database
            try:
                import sqlite3
                conn = sqlite3.connect("wolf.db")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, action, symbol, confidence, reasoning
                    FROM agent_decisions
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                conn.close()

                decisions = [
                    {
                        "timestamp": row[0],
                        "action": row[1],
                        "symbol": row[2],
                        "confidence": row[3],
                        "reasoning": row[4]
                    }
                    for row in rows
                ]
            except Exception:
                # Table might not exist yet, return empty
                decisions = []

        return {"decisions": decisions, "count": len(decisions)}
    except Exception as e:
        logger.error(f"Error getting agent decisions: {e}")
        return {"decisions": [], "count": 0, "error": str(e)}


@APP.get("/api/agent/stats")
async def api_agent_stats():
    """
    Get agent statistics for the cockpit dashboard.
    Returns decision count, win rate, avg confidence, etc.
    """
    try:
        stats = {
            "total_decisions": 0,
            "win_rate": 0.0,
            "avg_confidence": 0.0,
            "active_goals": 0,
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "timestamp": time.time()
        }

        # Try to get real stats
        try:
            import sqlite3
            conn = sqlite3.connect("wolf.db")
            cursor = conn.cursor()

            # Count total decisions
            cursor.execute("SELECT COUNT(*) FROM agent_decisions")
            stats["total_decisions"] = cursor.fetchone()[0]

            # Calculate win rate (if we have outcome data)
            cursor.execute("""
                SELECT COUNT(*) FROM agent_decisions
                WHERE outcome = 'win'
            """)
            wins = cursor.fetchone()[0]
            if stats["total_decisions"] > 0:
                stats["win_rate"] = wins / stats["total_decisions"]

            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM agent_decisions")
            avg_conf = cursor.fetchone()[0]
            stats["avg_confidence"] = avg_conf if avg_conf else 0.0

            conn.close()
        except Exception:
            pass

        # Get portfolio value
        try:
            portfolio = portfolio_manager.get_portfolio()
            stats["portfolio_value"] = portfolio.get("nav", 0.0)
        except Exception:
            pass

        return stats
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        return {
            "total_decisions": 0,
            "win_rate": 0.0,
            "avg_confidence": 0.0,
            "active_goals": 0,
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "timestamp": time.time(),
            "error": str(e)
        }

'''

# Find a good insertion point (after /api/agent/ask endpoint)
# Look for the next @APP decorator after line 14468
insertion_point = content.find('@APP.post("/api/agent/ask")')
if insertion_point == -1:
    print("❌ Could not find /api/agent/ask endpoint")
    exit(1)

# Find the end of this endpoint (next @APP decorator or end of file)
next_decorator = content.find("\n@APP.", insertion_point + 100)
if next_decorator == -1:
    next_decorator = len(content)

# Insert new endpoints
content = content[:next_decorator] + new_agent_endpoints + content[next_decorator:]

# Now add news endpoints
new_news_endpoints = '''

@APP.get("/api/news")
@APP.get("/api/news/recent")
async def api_news_recent(limit: int = 20):
    """
    Get recent news articles for the cockpit news feed.
    Aggregates from RSS feeds and news sentiment analysis.
    """
    try:
        news_items = []

        # Try to get from news_sentiment module if available
        try:
            if hasattr(AGENT_STATE, "news_feed") and AGENT_STATE.news_feed:
                news_items = AGENT_STATE.news_feed[-limit:]
        except Exception:
            pass

        # If no news in agent state, try RSS feeds
        if not news_items:
            try:
                import feedparser
                feeds = [
                    "https://feeds.reuters.com/reuters/businessNews",
                    "https://feeds.marketwatch.com/marketwatch/topstories/",
                ]

                for feed_url in feeds[:2]:  # Limit to 2 feeds for speed
                    try:
                        feed = feedparser.parse(feed_url)
                        for entry in feed.entries[:limit // 2]:
                            news_items.append({
                                "title": entry.get("title", ""),
                                "link": entry.get("link", ""),
                                "published": entry.get("published", ""),
                                "summary": entry.get("summary", "")[:200],
                                "source": feed_url.split('/')[2]
                            })
                    except Exception:
                        continue
            except Exception:
                pass

        # If still no news, return placeholder
        if not news_items:
            news_items = [
                {
                    "title": "Market Update",
                    "summary": "Real-time news feed initializing...",
                    "published": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Ghost Protocol",
                    "link": "#"
                }
            ]

        return {"news": news_items[-limit:], "count": len(news_items)}
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        return {"news": [], "count": 0, "error": str(e)}

'''

# Add after the agent endpoints we just added
content += new_news_endpoints

# Now add snapshot and research endpoints
new_snapshot_endpoints = '''

@APP.get("/api/snapshot")
async def api_snapshot():
    """
    Get real-time snapshot of entire system state for cockpit.
    Returns portfolio, goals, market regime, forecasts, news, etc.
    """
    try:
        snapshot = {
            "timestamp": time.time(),
            "portfolio": {},
            "market_regime": {},
            "forecasts": {},
            "goals": [],
            "decisions": [],
            "news": []
        }

        # Get portfolio
        try:
            snapshot["portfolio"] = portfolio_manager.get_portfolio()
        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")

        # Get market regime
        try:
            regime = await api_stage3_regime_current()
            snapshot["market_regime"] = regime
        except Exception as e:
            logger.error(f"Error getting regime: {e}")

        # Get recent forecasts
        try:
            forecasts = await api_stage2_forecasts()
            snapshot["forecasts"] = forecasts.get("forecasts", [])[:5]
        except Exception as e:
            logger.error(f"Error getting forecasts: {e}")

        # Get recent decisions
        try:
            decisions_data = await api_agent_decisions(limit=10)
            snapshot["decisions"] = decisions_data.get("decisions", [])
        except Exception as e:
            logger.error(f"Error getting decisions: {e}")

        # Get recent news
        try:
            news_data = await api_news_recent(limit=5)
            snapshot["news"] = news_data.get("news", [])
        except Exception as e:
            logger.error(f"Error getting news: {e}")

        return snapshot
    except Exception as e:
        logger.error(f"Error generating snapshot: {e}")
        return {
            "timestamp": time.time(),
            "error": str(e)
        }


@APP.get("/api/research/snapshot/{symbol}")
async def api_research_snapshot(symbol: str):
    """
    Get research snapshot for a specific symbol.
    Returns price, sentiment, forecasts, risk metrics, etc.
    """
    symbol = symbol.upper().strip()
    try:
        snapshot = {
            "symbol": symbol,
            "timestamp": time.time(),
            "price": {},
            "sentiment": {},
            "forecast": {},
            "risk": {},
            "news": []
        }

        # Get current price
        try:
            from core.price_fetcher import get_latest_price
            price_data = get_latest_price(symbol)
            snapshot["price"] = price_data
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")

        # Get sentiment
        try:
            sentiment_data = news_sentiment.get_symbol_sentiment(symbol)
            snapshot["sentiment"] = sentiment_data
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")

        # Get forecast
        try:
            forecast_data = predictor.get_latest_forecast(symbol)
            snapshot["forecast"] = forecast_data
        except Exception as e:
            logger.error(f"Error getting forecast for {symbol}: {e}")

        # Get news for this symbol
        try:
            # Use the existing ticker_news endpoint
            news_response = await api_watcher_ticker_news(symbol, limit=10)
            snapshot["news"] = news_response.get("news", [])
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")

        return snapshot
    except Exception as e:
        logger.error(f"Error generating research snapshot for {symbol}: {e}")
        return {
            "symbol": symbol,
            "timestamp": time.time(),
            "error": str(e)
        }


@APP.get("/api/stage5/execution/analytics")
async def api_stage5_execution_analytics():
    """
    Get execution analytics for stage 5 (execution/broker integration).
    Returns fill quality, slippage, execution speed, etc.
    """
    try:
        analytics = {
            "total_executions": 0,
            "avg_slippage": 0.0,
            "avg_fill_time": 0.0,
            "fill_rate": 1.0,
            "rejected_orders": 0,
            "timestamp": time.time()
        }

        # Try to get real execution data
        try:
            import sqlite3
            conn = sqlite3.connect("wolf.db")
            cursor = conn.cursor()

            # Count total executions
            cursor.execute("SELECT COUNT(*) FROM executions")
            analytics["total_executions"] = cursor.fetchone()[0]

            # Average slippage
            cursor.execute("SELECT AVG(slippage) FROM executions WHERE slippage IS NOT NULL")
            avg_slip = cursor.fetchone()[0]
            analytics["avg_slippage"] = avg_slip if avg_slip else 0.0

            # Average fill time
            cursor.execute("SELECT AVG(fill_time) FROM executions WHERE fill_time IS NOT NULL")
            avg_fill = cursor.fetchone()[0]
            analytics["avg_fill_time"] = avg_fill if avg_fill else 0.0

            # Rejected orders
            cursor.execute("SELECT COUNT(*) FROM executions WHERE status = 'rejected'")
            analytics["rejected_orders"] = cursor.fetchone()[0]

            # Fill rate
            if analytics["total_executions"] > 0:
                filled = analytics["total_executions"] - analytics["rejected_orders"]
                analytics["fill_rate"] = filled / analytics["total_executions"]

            conn.close()
        except Exception as e:
            logger.error(f"Error querying execution data: {e}")

        return analytics
    except Exception as e:
        logger.error(f"Error getting execution analytics: {e}")
        return {
            "total_executions": 0,
            "avg_slippage": 0.0,
            "avg_fill_time": 0.0,
            "fill_rate": 1.0,
            "rejected_orders": 0,
            "timestamp": time.time(),
            "error": str(e)
        }

'''

# Add at the end of the file, before the uvicorn.run() call
run_index = content.rfind("uvicorn.run(")
if run_index == -1:
    # Just append to end
    content += new_snapshot_endpoints
else:
    # Insert before uvicorn.run
    content = content[:run_index] + new_snapshot_endpoints + content[run_index:]

# Write back
with open("wolf_app.py", "w") as f:
    f.write(content)

print("✅ Added missing UI endpoints to wolf_app.py:")
print("   - /api/agent/decisions")
print("   - /api/agent/stats")
print("   - /api/news")
print("   - /api/news/recent")
print("   - /api/snapshot")
print("   - /api/research/snapshot/{symbol}")
print("   - /api/stage5/execution/analytics")
print()
print("⚠️  You need to restart the server for changes to take effect:")
print("   Kill current server (PID 31013): kill 31013")
print("   Then restart: python3 wolf_app.py")
