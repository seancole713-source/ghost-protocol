"""
Ghost Protocol - The Ghost Brain
Master orchestrator that combines ALL intelligence sources.

This is the "weatherman" for markets.
It doesn't just look at one indicator.
It combines EVERYTHING to make informed predictions.

MICRO SIGNALS + HUMAN BEHAVIOR + HISTORY + EVENTS = INTELLIGENT PREDICTION
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GhostBrain:
    """
    The Ghost Brain - Central Intelligence System
    
    Like a weatherman combines:
    - Satellite imagery (micro signals)
    - Pressure systems (events)
    - Historical patterns (what happened before)
    - Current conditions (human behavior)
    
    To make an intelligent forecast.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 600  # 10 minutes
    
    async def full_analysis(self, symbol: str) -> Dict:
        """
        Perform COMPLETE analysis of a symbol.
        
        This is the main function that combines ALL intelligence.
        """
        logger.info(f"[BRAIN] Starting full analysis for {symbol}")
        start_time = datetime.now()
        
        # Determine if crypto or stock
        crypto_symbols = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", 
                        "DOT", "LINK", "AVAX", "MATIC", "BCH", "LTC", "ZEC"}
        is_crypto = symbol.upper() in crypto_symbols
        
        # Import here to avoid circular imports
        from .micro_signals.micro_aggregator import scan_micro_signals
        from .human_behavior.narrative_detector import detect_narratives
        from .human_behavior.influencer_tracker import check_influencers
        from .historical.event_outcomes import get_event_database
        
        # Run all analysis in parallel
        try:
            results = await asyncio.gather(
                # Micro signals
                scan_micro_signals(symbol, is_crypto),
                
                # Human behavior
                detect_narratives(symbol),
                check_influencers(symbol),
                
                # Historical/seasonal
                get_event_database().get_seasonal_pattern(symbol),
                
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Brain analysis failed for {symbol}: {e}")
            return self._error_report(symbol, str(e))
        
        # Unpack results
        micro_signals = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0]), "warnings": [], "positives": [], "confidence_adjustment": 0}
        narratives = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1]), "confidence_adjustment": 0}
        influencers = results[2] if not isinstance(results[2], Exception) else {"error": str(results[2]), "warnings": [], "positives": [], "confidence_adjustment": 0}
        seasonal = results[3] if not isinstance(results[3], Exception) else {"error": str(results[3]), "confidence_adjustment": 0}
        
        # Calculate total confidence adjustment
        total_adjustment = 0
        total_adjustment += micro_signals.get("confidence_adjustment", 0)
        total_adjustment += narratives.get("confidence_adjustment", 0)
        total_adjustment += influencers.get("confidence_adjustment", 0)
        total_adjustment += seasonal.get("confidence_adjustment", 0)
        
        # Collect all warnings and positives
        all_warnings = []
        all_positives = []
        
        all_warnings.extend(micro_signals.get("warnings", []))
        all_warnings.extend(influencers.get("warnings", []))
        all_positives.extend(micro_signals.get("positives", []))
        all_positives.extend(influencers.get("positives", []))
        
        # Determine overall signal
        overall_signal = self._determine_overall_signal(
            micro_signals, narratives, influencers, seasonal
        )
        
        # Determine alert level
        alert_level = micro_signals.get("alert_level", "SHADOW")
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            symbol, overall_signal, total_adjustment, 
            all_warnings, all_positives, alert_level
        )
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"[BRAIN] Completed {symbol} in {duration_ms:.0f}ms - Signal: {overall_signal}")
        
        return {
            "symbol": symbol,
            "is_crypto": is_crypto,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration_ms),
            
            # Overall assessment
            "overall_signal": overall_signal,
            "alert_level": alert_level,
            "confidence_adjustment": max(-50, min(50, total_adjustment)),
            
            # Individual analyses
            "micro_signals": micro_signals,
            "narratives": narratives,
            "influencers": influencers,
            "seasonal": seasonal,
            
            # Aggregated warnings/positives
            "warnings": all_warnings,
            "positives": all_positives,
            
            # Final recommendation
            "recommendation": recommendation,
            
            # Summary for quick reference
            "summary": {
                "micro_alert": alert_level,
                "dominant_narrative": narratives.get("dominant_narrative", {}).get("name") if isinstance(narratives.get("dominant_narrative"), dict) else None,
                "narrative_sentiment": narratives.get("dominant_narrative", {}).get("sentiment") if isinstance(narratives.get("dominant_narrative"), dict) else None,
                "influencer_activity": influencers.get("has_activity", False),
                "seasonal_tendency": seasonal.get("pattern", {}).get("tendency") if isinstance(seasonal.get("pattern"), dict) else None,
                "total_warnings": len(all_warnings),
                "total_positives": len(all_positives)
            }
        }
    
    def _determine_overall_signal(self, micro: Dict, narrative: Dict, 
                                   influencer: Dict, seasonal: Dict) -> str:
        """Determine overall signal from all sources"""
        signals = []
        
        # Micro signal
        if micro.get("overall_signal"):
            signals.append(micro["overall_signal"])
        
        # Narrative
        if narrative.get("market_mood", {}).get("overall_mood"):
            mood = narrative["market_mood"]["overall_mood"]
            if mood in ["BULLISH", "BEARISH"]:
                signals.append(mood)
        
        # Influencer
        if influencer.get("signal") and influencer["signal"] != "NEUTRAL":
            signals.append(influencer["signal"])
        
        # Seasonal
        if isinstance(seasonal.get("pattern"), dict):
            tendency = seasonal["pattern"].get("tendency")
            if tendency and tendency != "NEUTRAL":
                signals.append(tendency)
        
        # Count
        bullish = signals.count("BULLISH")
        bearish = signals.count("BEARISH")
        
        if bullish >= 3:
            return "STRONG_BULLISH"
        elif bullish >= 2:
            return "BULLISH"
        elif bearish >= 3:
            return "STRONG_BEARISH"
        elif bearish >= 2:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _generate_recommendation(self, symbol: str, signal: str, adjustment: int,
                                   warnings: List, positives: List, alert_level: str) -> str:
        """Generate comprehensive recommendation"""
        parts = []
        
        parts.append("=" * 50)
        parts.append(f"🧠 GHOST BRAIN ANALYSIS: {symbol}")
        parts.append("=" * 50)
        parts.append("")
        
        # Alert level
        alert_emoji = {"SHADOW": "👁️", "WHISPER": "🗣️", "RIPPLE": "🌊", "WAVE": "🌊🌊"}
        parts.append(f"ALERT LEVEL: {alert_emoji.get(alert_level, '?')} {alert_level}")
        parts.append("")
        
        # Overall signal
        signal_emoji = {
            "STRONG_BULLISH": "🟢🟢", "BULLISH": "🟢",
            "STRONG_BEARISH": "🔴🔴", "BEARISH": "🔴",
            "NEUTRAL": "⚪"
        }
        parts.append(f"OVERALL SIGNAL: {signal_emoji.get(signal, '?')} {signal}")
        parts.append(f"CONFIDENCE ADJUSTMENT: {'+' if adjustment > 0 else ''}{adjustment}%")
        parts.append("")
        
        # Key findings
        if warnings:
            parts.append("⚠️ WARNINGS:")
            for w in warnings[:5]:
                parts.append(f"  • {w}")
            parts.append("")
        
        if positives:
            parts.append("✅ POSITIVE SIGNALS:")
            for p in positives[:5]:
                parts.append(f"  • {p}")
            parts.append("")
        
        # Final recommendation
        parts.append("-" * 50)
        parts.append("RECOMMENDATION:")
        parts.append("")
        
        if signal in ["STRONG_BULLISH", "BULLISH"]:
            if alert_level in ["WAVE", "RIPPLE"]:
                parts.append("⚠️ Bullish signals BUT high alert level.")
                parts.append("Consider waiting for pullback entry.")
            else:
                parts.append("✅ Conditions favor LONG positions.")
                parts.append(f"Adjust technical confidence by {'+' if adjustment > 0 else ''}{adjustment}%")
        elif signal in ["STRONG_BEARISH", "BEARISH"]:
            if alert_level in ["WAVE", "RIPPLE"]:
                parts.append("🚨 Bearish signals AND high alert level.")
                parts.append("Consider reducing exposure or hedging.")
            else:
                parts.append("⚠️ Conditions favor SHORT positions or cash.")
                parts.append(f"Adjust technical confidence by {adjustment}%")
        else:
            parts.append("➡️ Mixed signals. No strong edge detected.")
            parts.append("Consider waiting for clearer setup.")
        
        parts.append("")
        parts.append("=" * 50)
        
        return "\n".join(parts)
    
    def _error_report(self, symbol: str, error: str) -> Dict:
        """Return error report"""
        return {
            "symbol": symbol,
            "error": error,
            "overall_signal": "UNKNOWN",
            "confidence_adjustment": 0,
            "warnings": [],
            "positives": [],
            "recommendation": f"Analysis failed: {error}. Proceed with technical analysis only."
        }


# Singleton
_brain = None

def get_ghost_brain() -> GhostBrain:
    global _brain
    if _brain is None:
        _brain = GhostBrain()
    return _brain


async def analyze_with_intelligence(symbol: str) -> Dict:
    """Quick access to full brain analysis"""
    return await get_ghost_brain().full_analysis(symbol)
