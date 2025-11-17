"""
World Context Module
Provides SPY, VIX, market mood, and news aggregation for cockpit display.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def get_world_context() -> dict[str, Any]:
    """
    Get comprehensive world market context.
    
    Returns:
        {
            "spy": {"price": float, "change_pct": float, "provider": str},
            "vix": {"level": float, "change": float, "status": str},
            "market_mood": {"sentiment": str, "score": float, "factors": list},
            "news_summary": {"total": int, "bullish": int, "bearish": int, "top_stories": list},
            "timestamp": float
        }
    """
    from services.price_quorum import get_price_quorum
    
    result = {
        "spy": {"price": None, "change_pct": None, "provider": "unavailable"},
        "vix": {"level": None, "change": None, "status": "unknown"},
        "market_mood": {"sentiment": "neutral", "score": 50.0, "factors": []},
        "news_summary": {"total": 0, "bullish": 0, "bearish": 0, "top_stories": []},
        "timestamp": time.time()
    }
    
    # Get SPY price
    try:
        spy_data = get_price_quorum("SPY", "stock")
        if spy_data and spy_data.get("price"):
            result["spy"]["price"] = spy_data["price"]
            result["spy"]["provider"] = spy_data.get("provider", "unknown")
            # Calculate change % if we have previous close
            if spy_data.get("prev_close"):
                change_pct = ((spy_data["price"] - spy_data["prev_close"]) / spy_data["prev_close"]) * 100
                result["spy"]["change_pct"] = round(change_pct, 2)
    except Exception as e:
        logger.warning(f"Could not get SPY price: {e}")
    
    # Get VIX level (mock for now - would integrate with real provider)
    try:
        vix_data = get_price_quorum("VIX", "stock")
        if vix_data and vix_data.get("price"):
            vix_level = vix_data["price"]
            result["vix"]["level"] = round(vix_level, 2)
            
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
        logger.warning(f"Could not get VIX level: {e}")
    
    # Calculate market mood
    try:
        spy_change = result["spy"].get("change_pct", 0) or 0
        vix_level = result["vix"].get("level", 20) or 20
        
        # Simple mood calculation
        mood_score = 50.0  # Neutral baseline
        
        # SPY movement impact
        mood_score += spy_change * 5  # +1% SPY = +5 mood points
        
        # VIX impact (inverse - lower VIX = better mood)
        if vix_level < 15:
            mood_score += 10
        elif vix_level > 30:
            mood_score -= 15
        
        # Clamp to 0-100
        mood_score = max(0, min(100, mood_score))
        
        # Determine sentiment
        if mood_score >= 65:
            sentiment = "bullish"
        elif mood_score >= 35:
            sentiment = "neutral"
        else:
            sentiment = "bearish"
        
        result["market_mood"] = {
            "sentiment": sentiment,
            "score": round(mood_score, 1),
            "factors": [
                f"SPY {'+' if spy_change >= 0 else ''}{spy_change:.2f}%" if spy_change else "SPY flat",
                f"VIX {vix_level:.1f} ({result['vix']['status']})" if vix_level else "VIX unavailable"
            ]
        }
    except Exception as e:
        logger.warning(f"Could not calculate market mood: {e}")
    
    return result


def get_news_summary() -> dict[str, Any]:
    """
    Get aggregated news summary with sentiment distribution.
    
    Returns:
        {
            "total": int,
            "bullish": int,
            "bearish": int,
            "neutral": int,
            "top_stories": [{"title": str, "sentiment": str, "source": str, "timestamp": float}, ...]
        }
    """
    # This would integrate with actual news feed
    # For now, return empty structure
    return {
        "total": 0,
        "bullish": 0,
        "bearish": 0,
        "neutral": 0,
        "top_stories": []
    }
