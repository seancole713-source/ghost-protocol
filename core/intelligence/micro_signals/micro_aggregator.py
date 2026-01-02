"""
Ghost Protocol - Micro Signal Aggregator
Combines ALL micro signals into unified early warning system.

ALERT LEVELS:
- SHADOW:  Faint signals, only visible to those watching closely
- WHISPER: Growing signals, starting to be noticeable  
- RIPPLE:  Clear signals, about to become mainstream
- WAVE:    The move is happening now
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from .insider_tracker import analyze_insiders
from .whale_detector import analyze_whales
from .options_flow import analyze_options
from .social_velocity import analyze_social_velocity
from .volume_analyzer import analyze_volume

logger = logging.getLogger(__name__)


class MicroSignalAggregator:
    """
    Aggregate all micro signals into unified alert system.
    
    THE GOAL: See the storm forming, not just the storm arriving.
    
    Combines:
    - Insider transactions (who knows?)
    - Whale movements (what are big players doing?)
    - Options flow (how are pros positioning?)
    - Social velocity (what's the crowd sensing?)
    - Volume anomalies (what's hiding in the volume?)
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 900  # 15 minutes
    
    async def scan_all_signals(self, symbol: str, is_crypto: bool = None) -> Dict:
        """
        Scan ALL micro signal sources for a symbol.
        
        Args:
            symbol: Ticker symbol
            is_crypto: Force crypto mode, or auto-detect
        
        Returns:
            Comprehensive micro signal report
        """
        logger.info(f"[MICRO] Scanning all signals for {symbol}")
        start_time = datetime.now()
        
        # Auto-detect if crypto
        if is_crypto is None:
            crypto_symbols = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", 
                           "LINK", "AVAX", "MATIC", "BCH", "LTC", "ZEC"}
            is_crypto = symbol.upper() in crypto_symbols
        
        # Gather all signals in parallel
        tasks = []
        signal_names = []
        
        if not is_crypto:
            # Stock signals
            tasks.append(analyze_insiders(symbol))
            signal_names.append("insider")
            
            tasks.append(analyze_options(symbol))
            signal_names.append("options")
        else:
            # Crypto signals
            tasks.append(analyze_whales(symbol))
            signal_names.append("whale")
        
        # Common signals
        tasks.append(analyze_social_velocity(symbol))
        signal_names.append("social")
        
        tasks.append(analyze_volume(symbol))
        signal_names.append("volume")
        
        # Execute all
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Micro scan failed for {symbol}: {e}")
            return self._empty_report(symbol, str(e))
        
        # Organize results
        signals = {}
        for i, name in enumerate(signal_names):
            if isinstance(results[i], Exception):
                signals[name] = {"error": str(results[i]), "signal": "NEUTRAL", "confidence_adjustment": 0}
            else:
                signals[name] = results[i]
        
        # Aggregate signals
        aggregate = self._aggregate_signals(signals)
        
        # Determine alert level
        alert_level = self._determine_alert_level(aggregate)
        
        # Calculate overall confidence adjustment
        total_adjustment = sum(
            s.get("confidence_adjustment", 0) 
            for s in signals.values() 
            if isinstance(s, dict)
        )
        
        # Collect all warnings and positives
        all_warnings = []
        all_positives = []
        for s in signals.values():
            if isinstance(s, dict):
                all_warnings.extend(s.get("warnings", []))
                all_positives.extend(s.get("positives", []))
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"[MICRO] Completed {symbol} scan in {duration_ms:.0f}ms - Alert: {alert_level}")
        
        return {
            "symbol": symbol,
            "is_crypto": is_crypto,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration_ms),
            
            # Alert level
            "alert_level": alert_level,
            "alert_description": self._get_alert_description(alert_level),
            
            # Overall signal
            "overall_signal": aggregate["overall_signal"],
            "overall_strength": aggregate["overall_strength"],
            "confidence_adjustment": max(-50, min(50, total_adjustment)),
            
            # Individual signals
            "signals": signals,
            
            # Aggregated warnings/positives
            "warnings": all_warnings,
            "positives": all_positives,
            
            # Recommendation
            "recommendation": self._generate_recommendation(
                alert_level, aggregate, all_warnings, all_positives
            )
        }
    
    def _aggregate_signals(self, signals: Dict) -> Dict:
        """Aggregate individual signals into overall assessment"""
        bullish_signals = 0
        bearish_signals = 0
        total_strength = 0
        signal_count = 0
        
        for name, data in signals.items():
            if isinstance(data, dict) and data.get("signal"):
                signal = data.get("signal", "NEUTRAL")
                strength = data.get("signal_strength", 0)
                
                if signal == "BULLISH":
                    bullish_signals += 1
                elif signal == "BEARISH":
                    bearish_signals += 1
                
                total_strength += strength
                signal_count += 1
        
        # Determine overall
        if bullish_signals > bearish_signals and bullish_signals >= 2:
            overall = "BULLISH"
        elif bearish_signals > bullish_signals and bearish_signals >= 2:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"
        
        avg_strength = total_strength / signal_count if signal_count > 0 else 0
        
        return {
            "overall_signal": overall,
            "overall_strength": round(avg_strength),
            "bullish_count": bullish_signals,
            "bearish_count": bearish_signals,
            "signal_count": signal_count
        }
    
    def _determine_alert_level(self, aggregate: Dict) -> str:
        """
        Determine alert level based on signal strength.
        
        SHADOW:  0-20  = Faint signals
        WHISPER: 21-40 = Growing signals
        RIPPLE:  41-70 = Clear signals
        WAVE:    71+   = Move happening
        """
        strength = aggregate.get("overall_strength", 0)
        
        if strength >= 71:
            return "WAVE"
        elif strength >= 41:
            return "RIPPLE"
        elif strength >= 21:
            return "WHISPER"
        else:
            return "SHADOW"
    
    def _get_alert_description(self, level: str) -> str:
        """Get description for alert level"""
        descriptions = {
            "SHADOW": "👁️ Faint signals detected. Only visible to those watching closely.",
            "WHISPER": "🗣️ Growing signals. Something may be brewing.",
            "RIPPLE": "🌊 Clear signals detected. High probability of imminent move.",
            "WAVE": "🌊🌊 The move is happening NOW. Maximum alert."
        }
        return descriptions.get(level, "Unknown")
    
    def _generate_recommendation(self, alert_level: str, aggregate: Dict,
                                   warnings: List, positives: List) -> str:
        """Generate actionable recommendation"""
        parts = []
        
        parts.append(f"MICRO SIGNAL REPORT: {alert_level}")
        parts.append("=" * 40)
        
        if alert_level == "WAVE":
            parts.append("🚨 MAXIMUM ALERT")
            parts.append("Strong micro signals detected across multiple sources.")
            parts.append("A significant move is likely in progress or imminent.")
            parts.append("")
            parts.append("ACTION: Do not enter new positions. If holding, consider your stop-loss.")
            
        elif alert_level == "RIPPLE":
            parts.append("⚠️ HIGH ALERT")
            parts.append("Clear micro signals detected.")
            parts.append("The crowd is noticing. Move likely within 24-48 hours.")
            parts.append("")
            parts.append("ACTION: Reduce position size. Tighten stops.")
            
        elif alert_level == "WHISPER":
            parts.append("📢 ELEVATED ALERT")
            parts.append("Growing micro signals detected.")
            parts.append("Something may be developing. Monitor closely.")
            parts.append("")
            parts.append("ACTION: Proceed with caution. Don't go all-in.")
            
        else:
            parts.append("✅ NORMAL")
            parts.append("No significant micro signals detected.")
            parts.append("")
            parts.append("ACTION: Proceed with technical analysis as normal.")
        
        parts.append("")
        parts.append("KEY FINDINGS:")
        if warnings:
            for w in warnings[:5]:
                parts.append(f"  {w}")
        if positives:
            for p in positives[:5]:
                parts.append(f"  {p}")
        
        return "\n".join(parts)
    
    def _empty_report(self, symbol: str, error: str) -> Dict:
        """Return empty report on error"""
        return {
            "symbol": symbol,
            "error": error,
            "alert_level": "SHADOW",
            "overall_signal": "NEUTRAL",
            "confidence_adjustment": 0,
            "recommendation": "Micro signal scan failed. Proceed with caution."
        }


# Singleton
_aggregator = None

def get_micro_aggregator() -> MicroSignalAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = MicroSignalAggregator()
    return _aggregator


async def scan_micro_signals(symbol: str, is_crypto: bool = None) -> Dict:
    """Quick access to full micro signal scan"""
    return await get_micro_aggregator().scan_all_signals(symbol, is_crypto)
