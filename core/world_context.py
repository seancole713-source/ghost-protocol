"""
World Context Module
Aggregates global market context including SPY, VIX, market mood, and news.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


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
    
    # Get VIX level
    try:
        from core.price_quorum import get_price_quorum
        from core.providers.stock_providers import get_stock_providers
        
        quorum = get_price_quorum()
        providers = get_stock_providers("VIX")
        vix_decision = quorum.get_price("VIX", providers, is_market_open=True)
        
        if vix_decision and vix_decision.price:
            vix_level = vix_decision.price
            result["vix"]["level"] = round(vix_level, 2)
            
            # Calculate VIX change if prev_close available
            if vix_decision.prev_close:
                prev = vix_decision.prev_close
                change = vix_level - prev
                result["vix"]["change"] = round(change, 2)
            
            # Determine VIX status
            if vix_level < 15:
                result["vix"]["status"] = "calm"
            elif vix_level < 20:
                result["vix"]["status"] = "normal"
            elif vix_level < 30:
                result["vix"]["status"] = "elevated"
            else:
                result["vix"]["status"] = "high-fear"
        else:
            # FALLBACK: Try Polygon.io for VIX
            logger.info("VIX price_quorum returned NULL, trying Polygon.io fallback...")
            try:
                import os
                import requests
                
                polygon_key = os.getenv("POLYGON_API_KEY")
                if polygon_key:
                    try:
                        # VIX is tracked as I:VIX on Polygon
                        url = f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/2026-01-10/2026-01-14"
                        headers = {"Authorization": f"Bearer {polygon_key}"}
                        
                        resp = requests.get(url, headers=headers, timeout=3)
                        if resp.status_code == 200:
                            data = resp.json()
                            results = data.get("results", [])
                            
                            if len(results) >= 1:
                                current = results[-1]
                                vix_level = float(current.get("c", 0))  # Close level
                                
                                if len(results) >= 2:
                                    prev = results[-2]
                                    prev_close = float(prev.get("c", 0))
                                else:
                                    prev_close = vix_level
                                
                                if vix_level > 0:
                                    result["vix"]["level"] = round(vix_level, 2)
                                    
                                    if prev_close > 0:
                                        change = vix_level - prev_close
                                        result["vix"]["change"] = round(change, 2)
                                    
                                    # Determine VIX status
                                    if vix_level < 15:
                                        result["vix"]["status"] = "calm"
                                    elif vix_level < 20:
                                        result["vix"]["status"] = "normal"
                                    elif vix_level < 30:
                                        result["vix"]["status"] = "elevated"
                                    else:
                                        result["vix"]["status"] = "high-fear"
                                    
                                    logger.info(f"✅ VIX (Polygon): {vix_level:.2f} ({result['vix']['status']}, change: {result['vix'].get('change', 'N/A')})")
                        else:
                            logger.warning(f"Polygon VIX request failed: {resp.status_code}")
                    except Exception as poly_err:
                        logger.error(f"❌ Polygon VIX fallback error: {poly_err}")
                else:
                    logger.warning("POLYGON_API_KEY not set, cannot use VIX fallback")
            except Exception as fallback_err:
                logger.error(f"❌ VIX fallback error: {fallback_err}")
                
    except Exception as e:
        logger.error(f"Could not get VIX level: {e}")
    
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
