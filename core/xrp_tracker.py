"""
XRP Tracker Module
Specialized tracker for XRP with "bullish eye" indicator and signal generation.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def get_xrp_status() -> dict[str, Any]:
    """
    Get XRP tracking data with bullish eye indicator.
    
    Returns:
        {
            "price": float,
            "change_24h_pct": float,
            "bullish_eye": str ('🟢', '🟡', '🔴'),
            "signal": str ('BUY', 'HOLD', 'SELL', 'WAIT'),
            "confidence": float (0-1),
            "factors": list[str],
            "timestamp": float
        }
    """
    from services.price_quorum import get_price_quorum
    
    result = {
        "price": None,
        "change_24h_pct": None,
        "bullish_eye": "🟡",  # Yellow = neutral
        "signal": "WAIT",
        "confidence": 0.0,
        "factors": [],
        "timestamp": time.time()
    }
    
    try:
        # Get XRP price
        xrp_data = get_price_quorum("XRP", "crypto")
        
        if xrp_data and xrp_data.get("price"):
            price = xrp_data["price"]
            result["price"] = round(price, 4)
            
            # Calculate 24h change if available
            if xrp_data.get("prev_close"):
                change_pct = ((price - xrp_data["prev_close"]) / xrp_data["prev_close"]) * 100
                result["change_24h_pct"] = round(change_pct, 2)
            
            # Simple bullish eye logic
            # Green = strong buy, Yellow = neutral, Red = caution
            confidence = xrp_data.get("confidence", 0.5)
            
            factors = []
            signal_score = 0  # -1 to +1 scale
            
            # Factor 1: Price momentum
            if result["change_24h_pct"]:
                if result["change_24h_pct"] > 5:
                    factors.append("Strong upward momentum (+5%)")
                    signal_score += 0.4
                elif result["change_24h_pct"] > 2:
                    factors.append("Positive momentum (+2%)")
                    signal_score += 0.2
                elif result["change_24h_pct"] < -5:
                    factors.append("Sharp decline (-5%)")
                    signal_score -= 0.4
                elif result["change_24h_pct"] < -2:
                    factors.append("Negative momentum (-2%)")
                    signal_score -= 0.2
            
            # Factor 2: Data confidence
            if confidence > 0.8:
                factors.append("High data confidence")
                signal_score += 0.1
            elif confidence < 0.5:
                factors.append("Low data confidence")
                signal_score -= 0.1
            
            # Factor 3: Price level analysis (simple)
            if price > 1.0:
                factors.append("Above $1 resistance")
                signal_score += 0.1
            elif price < 0.5:
                factors.append("Below $0.50 support")
                signal_score -= 0.1
            
            # Determine bullish eye and signal
            if signal_score > 0.3:
                result["bullish_eye"] = "🟢"
                result["signal"] = "BUY"
                result["confidence"] = min(0.9, 0.5 + signal_score)
            elif signal_score < -0.3:
                result["bullish_eye"] = "🔴"
                result["signal"] = "SELL"
                result["confidence"] = min(0.9, 0.5 + abs(signal_score))
            elif abs(signal_score) > 0.1:
                result["bullish_eye"] = "🟡"
                result["signal"] = "HOLD"
                result["confidence"] = 0.6
            else:
                result["bullish_eye"] = "🟡"
                result["signal"] = "WAIT"
                result["confidence"] = 0.3
            
            result["factors"] = factors
            
    except Exception as e:
        logger.error(f"XRP tracker error: {e}")
        result["factors"] = [f"Error: {str(e)[:50]}"]
    
    return result
