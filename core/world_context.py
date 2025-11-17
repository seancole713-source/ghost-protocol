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
