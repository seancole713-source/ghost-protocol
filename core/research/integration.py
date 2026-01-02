"""
Ghost Protocol - Research Integration
Integrates deep research into prediction workflow
"""

import os
import logging
import asyncio
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Feature flag for research integration
RESEARCH_ENABLED = os.getenv("GHOST_RESEARCH_ENABLED", "1") == "1"
RESEARCH_TIMEOUT = float(os.getenv("GHOST_RESEARCH_TIMEOUT", "10"))  # 10s default


async def get_research_enhancement(symbol: str) -> Dict:
    """
    Get research data that can enhance a prediction.
    
    Returns:
        {
            "confidence_adjustment": int (-50 to +10),
            "warnings": list[str],
            "earnings_risk": bool,
            "news_sentiment": str,
            "seasonal_outlook": str,
            "research_available": bool
        }
    """
    if not RESEARCH_ENABLED:
        return {
            "confidence_adjustment": 0,
            "warnings": [],
            "earnings_risk": False,
            "news_sentiment": "unknown",
            "seasonal_outlook": "unknown",
            "research_available": False,
            "reason": "Research disabled via GHOST_RESEARCH_ENABLED"
        }
    
    try:
        from core.research import deep_research
        
        # Run with timeout
        research = await asyncio.wait_for(
            deep_research(symbol),
            timeout=RESEARCH_TIMEOUT
        )
        
        return {
            "confidence_adjustment": research.get("total_confidence_adjustment", 0),
            "warnings": research.get("warnings", []),
            "earnings_risk": research.get("summary", {}).get("earnings_risk", False),
            "news_sentiment": research.get("summary", {}).get("news_sentiment", "unknown"),
            "seasonal_outlook": research.get("summary", {}).get("seasonal_outlook", "unknown"),
            "range_position": research.get("summary", {}).get("range_position", 50),
            "research_available": True,
            "recommendation": research.get("recommendation", ""),
            "duration_ms": research.get("duration_ms", 0)
        }
    except asyncio.TimeoutError:
        logger.warning(f"[{symbol}] Research timeout after {RESEARCH_TIMEOUT}s")
        return {
            "confidence_adjustment": 0,
            "warnings": ["Research timeout - proceed with caution"],
            "research_available": False,
            "reason": "timeout"
        }
    except ImportError as e:
        logger.warning(f"Research module not available: {e}")
        return {
            "confidence_adjustment": 0,
            "warnings": [],
            "research_available": False,
            "reason": "module_unavailable"
        }
    except Exception as e:
        logger.error(f"[{symbol}] Research error: {e}")
        return {
            "confidence_adjustment": 0,
            "warnings": [],
            "research_available": False,
            "reason": str(e)
        }


def apply_research_adjustment(
    base_confidence: float,
    research: Dict,
    direction: str
) -> Tuple[float, str]:
    """
    Apply research-based confidence adjustment.
    
    Args:
        base_confidence: Original confidence (0.0 to 1.0)
        research: Research data from get_research_enhancement()
        direction: "UP" or "DOWN"
    
    Returns:
        (adjusted_confidence, adjustment_reason)
    """
    if not research.get("research_available"):
        return base_confidence, "No research data available"
    
    adjustment = research.get("confidence_adjustment", 0) / 100  # Convert to decimal
    
    # Apply adjustment
    adjusted = base_confidence + adjustment
    
    # Clamp to valid range
    adjusted = max(0.35, min(0.95, adjusted))
    
    # Build reason string
    reasons = []
    if research.get("earnings_risk"):
        reasons.append("⚠️ Earnings risk")
    
    news_sentiment = research.get("news_sentiment", "unknown")
    if news_sentiment == "bullish" and direction == "UP":
        reasons.append("📰 News supports direction")
    elif news_sentiment == "bearish" and direction == "DOWN":
        reasons.append("📰 News supports direction")
    elif news_sentiment in ("bullish", "bearish"):
        reasons.append(f"📰 News conflicts ({news_sentiment})")
    
    seasonal = research.get("seasonal_outlook", "unknown")
    if "BULLISH" in str(seasonal).upper():
        reasons.append("📅 Bullish season")
    elif "BEARISH" in str(seasonal).upper():
        reasons.append("📅 Bearish season")
    
    range_pos = research.get("range_position", 50)
    if range_pos > 90:
        reasons.append("📊 Near 52-week high")
    elif range_pos < 10:
        reasons.append("📊 Near 52-week low")
    
    adjustment_str = f"{adjustment*100:+.0f}% adjustment" if adjustment != 0 else "No adjustment"
    reason = f"{adjustment_str}: {', '.join(reasons)}" if reasons else adjustment_str
    
    return adjusted, reason


async def should_skip_prediction(symbol: str) -> Tuple[bool, str]:
    """
    Check if prediction should be skipped due to research warnings.
    
    Returns:
        (should_skip: bool, reason: str)
    """
    if not RESEARCH_ENABLED:
        return False, ""
    
    try:
        from core.research import check_earnings_risk
        
        earnings = await asyncio.wait_for(
            check_earnings_risk(symbol),
            timeout=5
        )
        
        # Skip if earnings are tomorrow
        if earnings.get("risky") and earnings.get("confidence_penalty", 0) >= 50:
            return True, f"Skipping {symbol}: {earnings.get('reason')}"
        
        return False, ""
    except Exception as e:
        logger.warning(f"[{symbol}] Earnings check failed: {e}")
        return False, ""
