"""
World Context Module
Aggregates global market context including SPY, VIX, market mood, and news.
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache for VIX to prevent spamming Polygon API (which returns 403 for index data)
_VIX_CACHE = {
    "level": 15.0,  # Default calm market
    "change": 0.0,
    "status": "normal",
    "last_fetch": 0,
    "polygon_403": False,  # Track if Polygon returned 403 (plan doesn't include VIX)
    "source": "default",  # Track where VIX came from
}
_VIX_CACHE_TTL = 300  # 5 minutes


def get_vix_yahoo_chart() -> Optional[float]:
    """
    Get VIX from Yahoo Finance chart API (most reliable, no rate limits).
    Uses the v8 chart endpoint which is more permissive than quoteSummary.
    """
    try:
        import httpx
        resp = httpx.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
            params={"interval": "1d", "range": "1d"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("chart", {}).get("result", [{}])[0]
            meta = result.get("meta", {})
            price = meta.get("regularMarketPrice") or meta.get("previousClose")
            if price and float(price) > 5:  # VIX should be > 5
                logger.info(f"[VIX] Yahoo Chart API: {price:.2f}")
                return float(price)
    except Exception as e:
        logger.warning(f"[VIX] Yahoo Chart API failed: {e}")
    return None


def get_vix_yahoo() -> Optional[float]:
    """
    Get VIX from Yahoo Finance (FREE, no API key needed).
    This is the primary VIX source since Polygon returns 403.
    """
    # Try chart API first (more reliable, less rate limited)
    vix = get_vix_yahoo_chart()
    if vix:
        return vix
    
    # Fallback to yfinance library
    try:
        import yfinance as yf
        vix_ticker = yf.Ticker("^VIX")
        # Try fast_info first (faster), fallback to info
        try:
            price = vix_ticker.fast_info.get('lastPrice') or vix_ticker.fast_info.get('regularMarketPrice')
            if price and price > 0:
                logger.info(f"[VIX] yfinance: {price:.2f}")
                return float(price)
        except Exception:
            pass
        # Fallback to info (slower but more reliable)
        info = vix_ticker.info
        price = info.get('regularMarketPrice') or info.get('previousClose')
        if price and price > 0:
            logger.info(f"[VIX] yfinance (info): {price:.2f}")
            return float(price)
    except ImportError:
        logger.warning("[VIX] yfinance not installed")
    except Exception as e:
        logger.warning(f"[VIX] yfinance failed: {e}")
    return None


def get_vix_cboe() -> Optional[float]:
    """
    Backup: Get VIX from CBOE website (scraping).
    Only used if Yahoo fails.
    """
    try:
        import httpx
        # CBOE has a simple JSON endpoint
        resp = httpx.get(
            "https://cdn.cboe.com/api/global/delayed_quotes/quotes/VIX.json",
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        if resp.status_code == 200:
            data = resp.json()
            price = data.get("data", {}).get("current_price")
            if price and float(price) > 0:
                logger.info(f"[VIX] CBOE: {price}")
                return float(price)
    except Exception as e:
        logger.debug(f"[VIX] CBOE fallback failed: {e}")
    return None


def get_real_vix() -> tuple[float, str]:
    """
    Get REAL VIX value from free sources.
    Returns (vix_value, source_name).
    
    Priority:
    1. Yahoo Finance (free, reliable)
    2. CBOE website (backup)
    3. Default 18.0 (neutral, not fake 15.0)
    """
    global _VIX_CACHE
    
    now = time.time()
    cache_age = now - _VIX_CACHE["last_fetch"]
    
    # Return cached if fresh (5 min TTL)
    if cache_age < _VIX_CACHE_TTL and _VIX_CACHE["source"] != "default":
        return _VIX_CACHE["level"], _VIX_CACHE["source"]
    
    # Try Yahoo first
    vix = get_vix_yahoo()
    if vix:
        _VIX_CACHE["level"] = vix
        _VIX_CACHE["source"] = "yahoo"
        _VIX_CACHE["last_fetch"] = now
        return vix, "yahoo"
    
    # Try CBOE backup
    vix = get_vix_cboe()
    if vix:
        _VIX_CACHE["level"] = vix
        _VIX_CACHE["source"] = "cboe"
        _VIX_CACHE["last_fetch"] = now
        return vix, "cboe"
    
    # Default - use 18.0 (neutral) instead of 15.0 (calm)
    # This is more conservative for trading decisions
    logger.warning("[VIX] All sources failed, using default 18.0")
    return 18.0, "default"


def get_world_context() -> dict[str, Any]:
    """
    Get current world market context.
    
    Returns:
        {
            "spy": {"price": float, "change_pct": float, "provider": str},
            "vix": {"level": float, "change": float, "status": str},
            "market_mood": {"sentiment": str, "score": float, "factors": list},
            "news_summary": {"total": int, "bullish": int, "bearish": int, "top_stories": list},
            "timestamp": float
        }
    """
    from core.price_quorum import get_price_quorum
    
    result = {
        "spy": {"price": None, "change_pct": None, "provider": "unavailable"},
        "vix": {"level": None, "change": None, "status": "unknown"},
        "market_mood": {"sentiment": "neutral", "score": 50.0, "factors": []},
        "news_summary": {"total": 0, "bullish": 0, "bearish": 0, "top_stories": []},
        "timestamp": time.time()
    }
    
    # Get SPY price
    try:
        from core.price_quorum import get_price_quorum
        from core.providers.stock_providers import get_stock_providers
        
        quorum = get_price_quorum()
        providers = get_stock_providers("SPY")
        spy_decision = quorum.get_price("SPY", providers, is_market_open=True)
        
        if spy_decision and spy_decision.price:
            price = spy_decision.price
            result["spy"]["price"] = round(price, 2)
            result["spy"]["provider"] = spy_decision.provider_label
            
            # Calculate change percentage if prev_close available
            if spy_decision.prev_close:
                prev = spy_decision.prev_close
                change_pct = ((price - prev) / prev) * 100
                result["spy"]["change_pct"] = round(change_pct, 2)
        else:
            # FALLBACK: Try Polygon.io (more reliable than yfinance)
            logger.info("SPY price_quorum returned NULL, trying Polygon.io fallback...")
            try:
                import os
                import requests
                
                polygon_key = os.getenv("POLYGON_API_KEY")
                if polygon_key:
                    try:
                        # Get last 2 days of SPY data from Polygon
                        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2026-01-10/2026-01-14"
                        headers = {"Authorization": f"Bearer {polygon_key}"}
                        
                        resp = requests.get(url, headers=headers, timeout=3)
                        if resp.status_code == 200:
                            data = resp.json()
                            results = data.get("results", [])
                            
                            if len(results) >= 1:
                                current = results[-1]
                                current_price = float(current.get("c", 0))  # Close price
                                
                                if len(results) >= 2:
                                    prev = results[-2]
                                    prev_close = float(prev.get("c", 0))
                                else:
                                    prev_close = current_price
                                
                                if current_price > 0:
                                    result["spy"]["price"] = round(current_price, 2)
                                    result["spy"]["provider"] = "polygon_fallback"
                                    
                                    if prev_close > 0:
                                        change_pct = ((current_price - prev_close) / prev_close) * 100
                                        result["spy"]["change_pct"] = round(change_pct, 2)
                                        logger.info(f"✅ SPY (Polygon): ${current_price:.2f} ({change_pct:+.2f}%)")
                                    else:
                                        logger.info(f"✅ SPY (Polygon): ${current_price:.2f}")
                        else:
                            logger.warning(f"Polygon SPY request failed: {resp.status_code}")
                    except Exception as poly_err:
                        logger.error(f"❌ Polygon SPY fallback error: {poly_err}")
                else:
                    logger.warning("POLYGON_API_KEY not set, cannot use fallback")
            except Exception as fallback_err:
                logger.error(f"❌ SPY fallback error: {fallback_err}")
                
    except Exception as e:
        logger.warning(f"Could not get SPY price: {e}")
    
    # =========================================================================
    # VIX: Use Yahoo Finance (FREE) as primary source
    # Polygon returns 403 for index data on free plans
    # =========================================================================
    try:
        vix_level, vix_source = get_real_vix()
        result["vix"]["level"] = round(vix_level, 2)
        result["vix"]["source"] = vix_source
        
        # Determine VIX status
        if vix_level < 15:
            result["vix"]["status"] = "calm"
        elif vix_level < 20:
            result["vix"]["status"] = "normal"
        elif vix_level < 30:
            result["vix"]["status"] = "elevated"
        else:
            result["vix"]["status"] = "high-fear"
        
        logger.info(f"[VIX] {vix_level:.2f} ({result['vix']['status']}) via {vix_source}")
        
    except Exception as e:
        logger.warning(f"[VIX] Error getting VIX: {e}, using default 18.0")
        result["vix"]["level"] = 18.0
        result["vix"]["status"] = "normal"
        result["vix"]["source"] = "error_default"
    
    # Calculate market mood based on SPY and VIX
    try:
        spy_price = result["spy"]["price"]
        spy_change = result["spy"]["change_pct"]
        vix_level = result["vix"]["level"]
        
        factors = []
        score = 50.0  # Neutral baseline
        
        if spy_change is not None:
            if spy_change > 1.0:
                score += 20
                factors.append("SPY strong up")
            elif spy_change > 0:
                score += 10
                factors.append("SPY up")
            elif spy_change < -1.0:
                score -= 20
                factors.append("SPY strong down")
            elif spy_change < 0:
                score -= 10
                factors.append("SPY down")
        
        if vix_level is not None:
            if vix_level < 15:
                score += 10
                factors.append("VIX calm")
            elif vix_level > 30:
                score -= 20
                factors.append("VIX high")
            elif vix_level > 20:
                score -= 10
                factors.append("VIX elevated")
        
        # Clamp score to 0-100
        score = max(0.0, min(100.0, score))
        
        # Determine sentiment
        if score >= 70:
            sentiment = "bullish"
        elif score >= 40:
            sentiment = "neutral"
        else:
            sentiment = "bearish"
        
        result["market_mood"] = {
            "sentiment": sentiment,
            "score": round(score, 1),
            "factors": factors
        }
    except Exception as e:
        logger.error(f"Could not calculate market mood: {e}")
    
    # Get news summary (placeholder for now - would integrate with news provider)
    result["news_summary"] = get_news_summary()
    
    return result


def get_news_summary() -> dict[str, Any]:
    """Get market news summary. Placeholder for future news integration."""
    return {
        "total": 0,
        "bullish": 0,
        "bearish": 0,
        "neutral": 0,
        "top_stories": []
    }
