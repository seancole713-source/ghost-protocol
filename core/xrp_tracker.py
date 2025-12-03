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
        from core.providers.turbo_provider import turbo_crypto_price
        
        # Get XRP price using turbo provider
        xrp_price_data = turbo_crypto_price("XRP", max_budget_s=3.0)
        
        if xrp_price_data and xrp_price_data.get("ok") and xrp_price_data.get("price"):
            price = xrp_price_data["price"]
            result["price"] = round(price, 4)
            
            # Calculate 24h change if available from cache metadata
            # Note: turbo_crypto_price doesn't return prev_close, so change_24h_pct stays None
            # This is acceptable - we focus on current price + bullish eye signal
            
            # Calculate bullish eye and signal
            factors = []
            confidence = 0.0
            
            # Factor 1: Price momentum
            if result["change_24h_pct"] is not None:
                change = result["change_24h_pct"]
                if change > 5.0:
                    factors.append("Strong momentum +5%")
                    confidence += 0.3
                elif change > 2.0:
                    factors.append("Positive momentum +2%")
                    confidence += 0.2
                elif change < -5.0:
                    factors.append("Weak momentum -5%")
                    confidence -= 0.3
                elif change < -2.0:
                    factors.append("Negative momentum -2%")
                    confidence -= 0.2
            
            # Factor 2: Data quality (quorum confidence)
            if xrp_decision.quorum_size >= 2:
                factors.append(f"Strong quorum ({xrp_decision.quorum_size} providers)")
                confidence += 0.2
            else:
                factors.append(f"Weak quorum ({xrp_decision.quorum_size} providers)")
                confidence -= 0.1
            
            # Factor 3: Price levels (simple example)
            if price > 2.0:
                factors.append("Above $2.00")
                confidence += 0.1
            elif price < 0.50:
                factors.append("Below $0.50")
                confidence -= 0.1
            
            # Normalize confidence to 0-1 range
            confidence = max(0.0, min(1.0, (confidence + 0.5)))
            result["confidence"] = round(confidence, 2)
            
            # Determine signal based on confidence
            if confidence >= 0.7:
                result["signal"] = "BUY"
                result["bullish_eye"] = "🟢"  # Green = bullish
            elif confidence >= 0.4:
                result["signal"] = "HOLD"
                result["bullish_eye"] = "🟡"  # Yellow = neutral
            elif confidence >= 0.2:
                result["signal"] = "WAIT"
                result["bullish_eye"] = "🟡"  # Yellow = cautious
            else:
                result["signal"] = "SELL"
                result["bullish_eye"] = "🔴"  # Red = bearish
            
            result["factors"] = factors
    
    except Exception as e:
        logger.error(f"XRP tracker error: {e}")
    
    return result
