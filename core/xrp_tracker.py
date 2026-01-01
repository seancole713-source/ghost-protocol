"""
XRP Tracker Module
Specialized tracker for XRP with "bullish eye" indicator and signal generation.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


async def get_xrp_status() -> dict[str, Any]:
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
        # Use the crypto price quorum which is proven to work
        from core.crypto.crypto_providers import get_crypto_price_quorum
        
        # Get XRP price from quorum (returns dict with price, change_24h_pct, etc.)
        xrp_data = await get_crypto_price_quorum("XRP", use_cache=True)
        
        if xrp_data and xrp_data.get("price"):
            price = xrp_data["price"]
            result["price"] = round(price, 4)
            
            # Get 24h change if available
            if xrp_data.get("change_24h_pct") is not None:
                result["change_24h_pct"] = round(xrp_data["change_24h_pct"], 2)
            
            # Calculate bullish eye and signal
            factors = []
            confidence = 0.5  # Start neutral
            
            # Factor 1: Price level analysis
            if price > 2.0:
                factors.append("Above $2.00 resistance")
                confidence += 0.15
            elif price > 1.5:
                factors.append("Mid-range $1.50-$2.00")
                confidence += 0.1
            elif price > 1.0:
                factors.append("Support at $1.00")
                confidence += 0.05
            elif price > 0.50:
                factors.append("Range $0.50-$1.00")
            else:
                factors.append("Below $0.50 support")
                confidence -= 0.1
            
            # Factor 2: Live data available
            factors.append("Live price from Coinbase")
            confidence += 0.05
            
            # Check if we have a recent prediction for XRP
            try:
                from wolf_app import _LATEST_PREDICTIONS
                xrp_pred = _LATEST_PREDICTIONS.get("XRP", {})
                if xrp_pred and xrp_pred.get("confidence"):
                    pred_confidence = xrp_pred.get("confidence", 0)
                    if pred_confidence > 1:
                        pred_confidence = pred_confidence / 100.0
                    confidence = pred_confidence
                    
                    pred_direction = xrp_pred.get("direction", "FLAT")
                    if pred_direction == "UP" and confidence >= 0.4:
                        result["signal"] = "BUY" if confidence >= 0.6 else "HOLD"
                        result["bullish_eye"] = "🟢" if confidence >= 0.6 else "🟡"
                    elif pred_direction == "DOWN" and confidence >= 0.4:
                        result["signal"] = "SELL" if confidence >= 0.6 else "WAIT"
                        result["bullish_eye"] = "🔴" if confidence >= 0.6 else "🟡"
                    else:
                        result["signal"] = "HOLD"
                        result["bullish_eye"] = "🟡"
                    
                    factors.append(f"Ghost prediction: {pred_direction} @ {confidence*100:.0f}%")
            except Exception as e:
                logger.debug(f"No prediction available for XRP: {e}")
            
            # If no prediction, use tracker heuristics
            if result["signal"] == "WAIT":
                confidence = max(0.0, min(1.0, confidence))
                if confidence >= 0.7:
                    result["signal"] = "BUY"
                    result["bullish_eye"] = "🟢"
                elif confidence >= 0.5:
                    result["signal"] = "HOLD"
                    result["bullish_eye"] = "🟡"
                elif confidence >= 0.3:
                    result["signal"] = "WAIT"
                    result["bullish_eye"] = "🟡"
                else:
                    result["signal"] = "SELL"
                    result["bullish_eye"] = "🔴"
            
            result["confidence"] = round(max(0.0, min(1.0, confidence)), 2)
            result["factors"] = factors
        else:
            result["factors"] = ["Unable to fetch XRP price"]
    
    except Exception as e:
        logger.error(f"XRP tracker error: {e}")
        result["factors"] = [f"Error: {str(e)}"]
    
    return result
