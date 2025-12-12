"""
🚀 SPIKE DETECTOR - Catch Random Price Spikes & Catalysts
Detects: News-driven spikes, unusual volume, pre-market activity, social sentiment surges
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SPIKE_DETECTOR_ENABLED = os.getenv("SPIKE_DETECTOR_ENABLED", "1") == "1"
PREMARKET_SPIKE_THRESHOLD = float(os.getenv("PREMARKET_SPIKE_THRESHOLD", "5.0"))  # 5%
VOLUME_ANOMALY_THRESHOLD = float(os.getenv("VOLUME_ANOMALY_THRESHOLD", "10.0"))  # 10x
UNUSUAL_VOLUME_MULTIPLIER = float(os.getenv("UNUSUAL_VOLUME_MULTIPLIER", "5.0"))  # 5x avg
NEWS_SCAN_INTERVAL = int(os.getenv("NEWS_SCAN_INTERVAL", "60"))  # 60 seconds
SOCIAL_SCAN_INTERVAL = int(os.getenv("SOCIAL_SCAN_INTERVAL", "120"))  # 2 minutes

# Alpha Vantage News API
ALPHA_VANTAGE_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# State tracking
_SPIKE_CACHE: dict[str, dict] = {}
_VOLUME_BASELINE: dict[str, float] = {}
_LAST_ALERTS: dict[str, float] = {}
_ALERT_COOLDOWN = 300  # 5 minutes
_DISCOVERED_SYMBOLS: set[str] = set()  # Dynamic watchlist
_DISCOVERY_LAST_RUN = 0
_DISCOVERY_INTERVAL = 300  # 5 minutes


# ============================================================================
# 1. PRE-MARKET UNUSUAL ACTIVITY SCANNER
# ============================================================================

async def scan_premarket_spikes(symbols: list[str]) -> list[dict[str, Any]]:
    """
    Scan for stocks with unusual pre-market activity (>5% moves)
    
    Args:
        symbols: List of stock tickers to monitor
        
    Returns:
        List of opportunities with pre-market spikes
    """
    opportunities = []
    
    # Check if market hours (before 9:30 AM ET)
    now = datetime.now()
    market_open_et = now.replace(hour=9, minute=30, second=0, microsecond=0)
    
    if now >= market_open_et:
        LOGGER.debug("Market open - skipping pre-market scan")
        return opportunities
    
    LOGGER.info(f"🌅 Scanning {len(symbols)} symbols for pre-market activity...")
    
    try:
        from core.providers.turbo_provider import get_turbo_provider
        turbo = get_turbo_provider()
        
        for symbol in symbols:
            try:
                # Get current price
                price_data = await turbo.get_price_async(symbol)
                if not price_data or "price" not in price_data:
                    continue
                
                current_price = price_data["price"]
                prev_close = price_data.get("prev_close")
                
                if not prev_close:
                    continue
                
                # Calculate pre-market change
                change_pct = ((current_price - prev_close) / prev_close) * 100
                
                if abs(change_pct) >= PREMARKET_SPIKE_THRESHOLD:
                    opportunity = {
                        "symbol": symbol,
                        "type": "premarket_spike",
                        "current_price": current_price,
                        "prev_close": prev_close,
                        "change_pct": round(change_pct, 2),
                        "timestamp": int(time.time()),
                        "reason": f"Pre-market move: {change_pct:+.2f}%"
                    }
                    
                    opportunities.append(opportunity)
                    LOGGER.info(f"🚨 PRE-MARKET SPIKE: {symbol} {change_pct:+.2f}%")
                    
            except Exception as e:
                LOGGER.debug(f"Error scanning {symbol}: {e}")
                continue
        
        return opportunities
        
    except Exception as e:
        LOGGER.error(f"Pre-market scan failed: {e}", exc_info=True)
        return opportunities


# ============================================================================
# 2. UNUSUAL VOLUME DETECTOR
# ============================================================================

async def detect_unusual_volume(symbols: list[str]) -> list[dict[str, Any]]:
    """
    Detect stocks with 10x+ average volume
    
    Args:
        symbols: List of tickers to monitor
        
    Returns:
        List of opportunities with unusual volume
    """
    opportunities = []
    
    LOGGER.info(f"📊 Scanning {len(symbols)} symbols for unusual volume...")
    
    try:
        from core.providers.turbo_provider import get_turbo_provider
        turbo = get_turbo_provider()
        
        for symbol in symbols:
            try:
                # Get current volume
                price_data = await turbo.get_price_async(symbol)
                if not price_data or "volume" not in price_data:
                    continue
                
                current_volume = price_data["volume"]
                avg_volume = price_data.get("avg_volume")
                
                if not avg_volume or avg_volume == 0:
                    continue
                
                volume_ratio = current_volume / avg_volume
                
                # Detect anomaly
                if volume_ratio >= VOLUME_ANOMALY_THRESHOLD:
                    opportunity = {
                        "symbol": symbol,
                        "type": "unusual_volume",
                        "current_volume": current_volume,
                        "avg_volume": avg_volume,
                        "volume_ratio": round(volume_ratio, 2),
                        "timestamp": int(time.time()),
                        "reason": f"Volume {volume_ratio:.1f}x average"
                    }
                    
                    opportunities.append(opportunity)
                    LOGGER.info(f"🔊 UNUSUAL VOLUME: {symbol} {volume_ratio:.1f}x average")
                    
                    # Update baseline
                    _VOLUME_BASELINE[symbol] = current_volume
                    
            except Exception as e:
                LOGGER.debug(f"Error scanning {symbol} volume: {e}")
                continue
        
        return opportunities
        
    except Exception as e:
        LOGGER.error(f"Volume scan failed: {e}", exc_info=True)
        return opportunities


# ============================================================================
# 3. REAL-TIME NEWS/CATALYST DETECTOR
# ============================================================================

async def scan_breaking_news(symbols: list[str]) -> list[dict[str, Any]]:
    """
    Scan for breaking news/catalysts using Alpha Vantage News API
    
    Args:
        symbols: List of tickers to monitor
        
    Returns:
        List of opportunities with breaking news
    """
    opportunities = []
    
    if not ALPHA_VANTAGE_KEY:
        LOGGER.debug("Alpha Vantage API key not set - skipping news scan")
        return opportunities
    
    LOGGER.info(f"📰 Scanning news for {len(symbols)} symbols...")
    
    try:
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Check cooldown
                    last_alert = _LAST_ALERTS.get(f"news_{symbol}", 0)
                    if time.time() - last_alert < _ALERT_COOLDOWN:
                        continue
                    
                    # Fetch recent news
                    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_KEY}&limit=5"
                    
                    async with session.get(url, timeout=5) as response:
                        if response.status != 200:
                            continue
                        
                        data = await response.json()
                        
                        if "feed" not in data:
                            continue
                        
                        articles = data["feed"]
                        
                        # Check for recent high-impact news (last hour)
                        one_hour_ago = datetime.now() - timedelta(hours=1)
                        recent_articles = []
                        
                        for article in articles[:5]:
                            try:
                                pub_time = datetime.strptime(
                                    article["time_published"],
                                    "%Y%m%dT%H%M%S"
                                )
                                
                                if pub_time >= one_hour_ago:
                                    recent_articles.append(article)
                            except:
                                continue
                        
                        # If we have recent news, flag it
                        if recent_articles:
                            # Get sentiment scores
                            sentiments = []
                            for article in recent_articles:
                                ticker_sentiments = article.get("ticker_sentiment", [])
                                for ts in ticker_sentiments:
                                    if ts.get("ticker") == symbol:
                                        score = float(ts.get("ticker_sentiment_score", 0))
                                        sentiments.append(score)
                            
                            if sentiments:
                                avg_sentiment = sum(sentiments) / len(sentiments)
                                
                                # Strong sentiment = catalyst
                                if abs(avg_sentiment) >= 0.3:
                                    opportunity = {
                                        "symbol": symbol,
                                        "type": "news_catalyst",
                                        "article_count": len(recent_articles),
                                        "sentiment_score": round(avg_sentiment, 2),
                                        "top_headline": recent_articles[0].get("title", "")[:100],
                                        "timestamp": int(time.time()),
                                        "reason": f"Breaking news: {len(recent_articles)} articles, sentiment {avg_sentiment:+.2f}"
                                    }
                                    
                                    opportunities.append(opportunity)
                                    LOGGER.info(f"📰 NEWS CATALYST: {symbol} - {len(recent_articles)} articles")
                                    
                                    _LAST_ALERTS[f"news_{symbol}"] = time.time()
                    
                    # Rate limit
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    LOGGER.debug(f"Error fetching news for {symbol}: {e}")
                    continue
        
        return opportunities
        
    except Exception as e:
        LOGGER.error(f"News scan failed: {e}", exc_info=True)
        return opportunities


# ============================================================================
# 4. SOCIAL SENTIMENT SURGE DETECTOR
# ============================================================================

async def detect_social_buzz(symbols: list[str]) -> list[dict[str, Any]]:
    """
    Detect sudden social media buzz (Reddit WallStreetBets, Twitter)
    
    NOTE: This is a placeholder - requires StockTwits/Reddit API integration
    
    Args:
        symbols: List of tickers to monitor
        
    Returns:
        List of opportunities with social buzz
    """
    opportunities = []
    
    # TODO: Integrate with:
    # - Reddit API (r/wallstreetbets)
    # - StockTwits API
    # - Twitter/X API
    
    LOGGER.debug("Social buzz detection not yet implemented")
    
    return opportunities


# ============================================================================
# MASTER SPIKE SCANNER
# ============================================================================

async def scan_all_spikes(symbols: list[str]) -> dict[str, Any]:
    """
    Run all spike detection scans in parallel
    
    Args:
        symbols: List of tickers to monitor
        
    Returns:
        {
            "timestamp": unix_ts,
            "premarket_spikes": [...],
            "unusual_volume": [...],
            "news_catalysts": [...],
            "social_buzz": [...],
            "total_opportunities": N
        }
    """
    if not SPIKE_DETECTOR_ENABLED:
        return {
            "timestamp": int(time.time()),
            "premarket_spikes": [],
            "unusual_volume": [],
            "news_catalysts": [],
            "social_buzz": [],
            "total_opportunities": 0
        }
    
    LOGGER.info(f"🔍 Running comprehensive spike detection on {len(symbols)} symbols...")
    
    # Run all scans in parallel
    results = await asyncio.gather(
        scan_premarket_spikes(symbols),
        detect_unusual_volume(symbols),
        scan_breaking_news(symbols),
        detect_social_buzz(symbols),
        return_exceptions=True
    )
    
    premarket = results[0] if not isinstance(results[0], Exception) else []
    volume = results[1] if not isinstance(results[1], Exception) else []
    news = results[2] if not isinstance(results[2], Exception) else []
    social = results[3] if not isinstance(results[3], Exception) else []
    
    total = len(premarket) + len(volume) + len(news) + len(social)
    
    LOGGER.info(
        f"✅ Spike scan complete: {total} opportunities "
        f"(premarket={len(premarket)}, volume={len(volume)}, news={len(news)}, social={len(social)})"
    )
    
    return {
        "timestamp": int(time.time()),
        "premarket_spikes": premarket,
        "unusual_volume": volume,
        "news_catalysts": news,
        "social_buzz": social,
        "total_opportunities": total
    }


# ============================================================================
# CONTINUOUS BACKGROUND SCANNER
# ============================================================================

async def spike_scanner_loop(base_symbols: list[str]):
    """
    Continuous background loop for spike detection.
    Combines core watchlist + dynamically discovered symbols.
    
    Args:
        base_symbols: Core watchlist from beast_scheduler
    """
    LOGGER.info(f"🚀 Spike detector started - monitoring {len(base_symbols)} core symbols + dynamic market discovery")
    
    while True:
        try:
            # Get complete symbol list (core + discovered)
            all_symbols = await get_all_tracked_symbols(base_symbols)
            
            # Scan for spikes
            results = await scan_all_spikes(all_symbols)
            
            # Send alerts for opportunities
            all_opportunities = (
                results["premarket_spikes"] +
                results["unusual_volume"] +
                results["news_catalysts"] +
                results["social_buzz"]
            )
            
            if all_opportunities:
                await send_spike_alerts(all_opportunities)
            
            # Variable sleep based on market hours
            now = datetime.now()
            if now.hour >= 4 and now.hour < 9:
                # Pre-market: scan every 30 seconds
                await asyncio.sleep(30)
            elif now.hour >= 9 and now.hour < 16:
                # Market hours: scan every 60 seconds
                await asyncio.sleep(60)
            else:
                # After hours: scan every 5 minutes
                await asyncio.sleep(300)
                
        except Exception as e:
            LOGGER.error(f"Spike scanner error: {e}", exc_info=True)
            await asyncio.sleep(60)


async def send_spike_alerts(opportunities: list[dict[str, Any]]):
    """
    Send Telegram alerts for spike opportunities
    
    Args:
        opportunities: List of spike opportunities
    """
    try:
        from core.telegram_alerts import send_alert
        
        for opp in opportunities:
            symbol = opp["symbol"]
            
            # Check alert cooldown
            last_alert = _LAST_ALERTS.get(symbol, 0)
            if time.time() - last_alert < _ALERT_COOLDOWN:
                continue
            
            # Format alert message
            if opp["type"] == "premarket_spike":
                message = (
                    f"🌅 PRE-MARKET SPIKE\n"
                    f"Symbol: {symbol}\n"
                    f"Change: {opp['change_pct']:+.2f}%\n"
                    f"Price: ${opp['current_price']:.2f}\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
            elif opp["type"] == "unusual_volume":
                message = (
                    f"🔊 UNUSUAL VOLUME\n"
                    f"Symbol: {symbol}\n"
                    f"Volume: {opp['volume_ratio']:.1f}x average\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
            elif opp["type"] == "news_catalyst":
                message = (
                    f"📰 NEWS CATALYST\n"
                    f"Symbol: {symbol}\n"
                    f"Articles: {opp['article_count']}\n"
                    f"Sentiment: {opp['sentiment_score']:+.2f}\n"
                    f"Headline: {opp['top_headline']}\n"
                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                )
            else:
                continue
            
            await send_alert(message, priority="high")
            _LAST_ALERTS[symbol] = time.time()
            
            LOGGER.info(f"📤 Sent spike alert for {symbol}")
            
    except Exception as e:
        LOGGER.error(f"Failed to send spike alerts: {e}", exc_info=True)


# ============================================================================
# 5. DYNAMIC MARKET-WIDE DISCOVERY (CATCH EVERYTHING!)
# ============================================================================

async def discover_market_movers() -> list[str]:
    """
    Scan ENTIRE market for any stock/crypto spiking >10% or 10x volume.
    Uses Polygon.io "most active" and "top gainers" endpoints.
    
    Returns:
        List of newly discovered symbols with significant activity
    """
    global _DISCOVERY_LAST_RUN, _DISCOVERED_SYMBOLS
    
    discovered = []
    
    # Rate limit: run every 5 minutes
    if time.time() - _DISCOVERY_LAST_RUN < _DISCOVERY_INTERVAL:
        return discovered
    
    LOGGER.info("🌍 SCANNING ENTIRE MARKET for new opportunities...")
    
    try:
        polygon_key = os.getenv("POLYGON_API_KEY")
        if not polygon_key:
            LOGGER.warning("⚠️ POLYGON_API_KEY not set - market-wide discovery disabled")
            return discovered
        
        async with aiohttp.ClientSession() as session:
            # 1. Get top gainers (stocks)
            gainers_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey={polygon_key}"
            async with session.get(gainers_url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "tickers" in data:
                        for ticker in data["tickers"][:50]:  # Top 50
                            symbol = ticker.get("ticker")
                            change_pct = ticker.get("todaysChangePerc", 0)
                            
                            if change_pct >= 10.0:  # >10% gain
                                discovered.append(symbol)
                                _DISCOVERED_SYMBOLS.add(symbol)
                                LOGGER.info(f"🔥 DISCOVERED: {symbol} +{change_pct:.1f}%")
            
            # 2. Get most active by volume (stocks)
            active_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers?apiKey={polygon_key}"
            async with session.get(active_url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "tickers" in data:
                        # Sort by volume
                        sorted_tickers = sorted(
                            data["tickers"],
                            key=lambda x: x.get("day", {}).get("v", 0),
                            reverse=True
                        )[:100]  # Top 100 by volume
                        
                        for ticker in sorted_tickers:
                            symbol = ticker.get("ticker")
                            volume = ticker.get("day", {}).get("v", 0)
                            prev_volume = ticker.get("prevDay", {}).get("v", 1)
                            
                            if prev_volume > 0:
                                volume_ratio = volume / prev_volume
                                if volume_ratio >= 10.0:  # 10x volume
                                    if symbol not in discovered:
                                        discovered.append(symbol)
                                        _DISCOVERED_SYMBOLS.add(symbol)
                                        LOGGER.info(f"🔊 DISCOVERED: {symbol} (volume {volume_ratio:.1f}x)")
            
            # 3. Get crypto gainers (if supported)
            crypto_url = f"https://api.polygon.io/v2/snapshot/locale/global/markets/crypto/gainers?apiKey={polygon_key}"
            async with session.get(crypto_url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "tickers" in data:
                        for ticker in data["tickers"][:30]:  # Top 30 crypto
                            symbol = ticker.get("ticker", "").replace("X:", "")
                            change_pct = ticker.get("todaysChangePerc", 0)
                            
                            if change_pct >= 15.0:  # >15% crypto gain
                                discovered.append(symbol)
                                _DISCOVERED_SYMBOLS.add(symbol)
                                LOGGER.info(f"🚀 DISCOVERED CRYPTO: {symbol} +{change_pct:.1f}%")
        
        _DISCOVERY_LAST_RUN = time.time()
        
        if discovered:
            LOGGER.info(f"✅ Market scan complete: {len(discovered)} new opportunities found")
        else:
            LOGGER.info("ℹ️ Market scan complete: No new high-momentum symbols")
        
        return discovered
        
    except Exception as e:
        LOGGER.error(f"Market discovery failed: {e}", exc_info=True)
        return discovered


async def get_all_tracked_symbols(base_symbols: list[str]) -> list[str]:
    """
    Combine base watchlist + dynamically discovered symbols.
    
    Args:
        base_symbols: Core watchlist from beast_scheduler
        
    Returns:
        Complete list of symbols to scan
    """
    # Run discovery every 5 minutes
    newly_discovered = await discover_market_movers()
    
    # Merge base + discovered
    all_symbols = list(set(base_symbols) | _DISCOVERED_SYMBOLS)
    
    if newly_discovered:
        LOGGER.info(f"📊 Tracking {len(all_symbols)} symbols ({len(base_symbols)} core + {len(_DISCOVERED_SYMBOLS)} discovered)")
    
    return all_symbols
    except Exception as e:
        LOGGER.error(f"Failed to send spike alerts: {e}")
