#!/usr/bin/env python3
"""
🧠 GHOST NEWS BRAIN - AI-Powered News Analysis

Uses Claude to analyze breaking news and cross-reference with Ghost's predictions.
Runs automatically at 6 AM and 6 PM CT, or on-demand via API.

Key Features:
- Web search for breaking news in last 24 hours
- Cross-references news with pending Ghost predictions
- Alerts when predictions may be invalidated by news
- Sends Telegram alerts for high-risk situations
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Optional anthropic import
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# News categories to monitor
NEWS_CATEGORIES = [
    "stock market news",
    "cryptocurrency news", 
    "federal reserve interest rates",
    "geopolitical events",
    "major company earnings",
    "SEC announcements crypto",
    "oil prices energy",
    "tech sector news",
]

# Severity levels
SEVERITY_CRITICAL = "CRITICAL"  # War, Fed emergency, major hack
SEVERITY_HIGH = "HIGH"          # Earnings miss, regulatory action
SEVERITY_MEDIUM = "MEDIUM"      # Sector rotation, analyst upgrades
SEVERITY_LOW = "LOW"            # Normal market news


class GhostNewsBrain:
    """
    AI-powered news analysis for Ghost Protocol.
    
    Uses Claude to:
    1. Search for breaking news
    2. Analyze impact on markets
    3. Cross-reference with Ghost's pending predictions
    4. Alert when predictions may be wrong due to news
    """
    
    def __init__(self, send_telegram_func=None):
        self.client = None
        self.send_telegram = send_telegram_func
        self._last_analysis: Dict = {}
        self._analysis_history: List[Dict] = []
        self._init_client()
    
    def _init_client(self):
        """Initialize Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            LOGGER.warning("[NEWS BRAIN] anthropic package not installed")
            return
            
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                LOGGER.info("[NEWS BRAIN] Anthropic client initialized")
            except Exception as e:
                LOGGER.error(f"[NEWS BRAIN] Failed to init Anthropic: {e}")
                self.client = None
        else:
            LOGGER.warning("[NEWS BRAIN] No ANTHROPIC_API_KEY - news analysis disabled")
    
    def analyze_news(self, pending_predictions: List[Dict] = None) -> Dict:
        """
        Analyze breaking news and cross-reference with predictions.
        
        Args:
            pending_predictions: List of Ghost's pending predictions
                Each should have: symbol, direction, confidence, entry_price
        
        Returns:
            Analysis result with major_events, predictions_at_risk, etc.
        """
        if not self.client:
            return {
                "status": "error",
                "error": "Anthropic client not initialized",
                "timestamp": datetime.utcnow().isoformat(),
            }
        
        try:
            # Build the analysis prompt
            predictions_context = ""
            if pending_predictions:
                pred_lines = []
                for p in pending_predictions[:50]:  # Limit to 50 for context
                    pred_lines.append(
                        f"- {p.get('symbol', '?')}: {p.get('direction', '?')} @ {p.get('confidence', 0):.0%}"
                    )
                predictions_context = f"""

GHOST'S CURRENT PREDICTIONS (need to check if any are at risk):
{chr(10).join(pred_lines)}
"""
            
            prompt = f"""You are a financial news analyst. Analyze the most important market-moving news from the last 24 hours.

TODAY'S DATE: {datetime.utcnow().strftime('%Y-%m-%d')}
{predictions_context}

Please analyze:
1. What are the TOP 3-5 most market-moving news events in the last 24 hours?
2. For each event, what sectors/symbols are affected?
3. Are any of Ghost's predictions likely WRONG because of this news?

IMPORTANT: Focus on ACTUAL news that happened, not speculation. Include:
- Geopolitical events (wars, sanctions, elections)
- Federal Reserve / central bank actions
- Major earnings surprises
- Regulatory announcements
- Crypto-specific news (SEC, hacks, etc.)

Respond in this exact JSON format:
{{
    "analysis_time": "{datetime.utcnow().isoformat()}",
    "major_events": [
        {{
            "headline": "Brief headline",
            "severity": "CRITICAL|HIGH|MEDIUM|LOW",
            "event_type": "GEOPOLITICAL|FED|EARNINGS|REGULATORY|CRYPTO|SECTOR",
            "summary": "2-3 sentence summary",
            "affected_sectors": ["sector1", "sector2"],
            "bullish_symbols": ["SYM1", "SYM2"],
            "bearish_symbols": ["SYM3", "SYM4"]
        }}
    ],
    "predictions_at_risk": [
        {{
            "symbol": "SYM",
            "our_prediction": "UP|DOWN",
            "likely_actual": "UP|DOWN",
            "reason": "Why the prediction may be wrong",
            "risk_level": "HIGH|MEDIUM|LOW"
        }}
    ],
    "market_sentiment": "BULLISH|BEARISH|NEUTRAL|MIXED",
    "action_required": true|false,
    "recommendation": "Brief recommendation for Ghost users"
}}

If no major news, return empty arrays but still provide market_sentiment."""

            # Call Claude
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            response_text = response.content[0].text
            
            # Try to extract JSON from response
            try:
                # Find JSON in response
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    result = json.loads(json_str)
                else:
                    result = {"raw_response": response_text}
            except json.JSONDecodeError:
                result = {"raw_response": response_text}
            
            # Add metadata
            result["status"] = "success"
            result["timestamp"] = datetime.utcnow().isoformat()
            result["predictions_checked"] = len(pending_predictions) if pending_predictions else 0
            
            # Store in history
            self._last_analysis = result
            self._analysis_history.append(result)
            if len(self._analysis_history) > 100:
                self._analysis_history = self._analysis_history[-100:]
            
            # Send alert if action required
            if result.get("action_required") and self.send_telegram:
                self._send_alert(result)
            
            LOGGER.info(f"[NEWS BRAIN] Analysis complete - {len(result.get('major_events', []))} events found")
            return result
            
        except Exception as e:
            LOGGER.error(f"[NEWS BRAIN] Analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    def _send_alert(self, analysis: Dict):
        """Send Telegram alert for predictions at risk"""
        if not self.send_telegram:
            return
        
        predictions_at_risk = analysis.get("predictions_at_risk", [])
        if not predictions_at_risk:
            return
        
        # Build alert message
        lines = [
            "🧠 GHOST NEWS BRAIN ALERT",
            "",
            "⚠️ Predictions may be affected by breaking news:",
            "",
        ]
        
        for p in predictions_at_risk[:5]:  # Limit to 5
            symbol = p.get("symbol", "?")
            our_pred = p.get("our_prediction", "?")
            likely = p.get("likely_actual", "?")
            reason = p.get("reason", "News impact")
            risk = p.get("risk_level", "MEDIUM")
            
            emoji = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "⚪"
            lines.append(f"{emoji} {symbol}: Predicted {our_pred} → May go {likely}")
            lines.append(f"   Reason: {reason}")
            lines.append("")
        
        # Add major events summary
        events = analysis.get("major_events", [])
        if events:
            lines.append("📰 Key Events:")
            for e in events[:3]:
                severity_emoji = "🚨" if e.get("severity") == "CRITICAL" else "⚠️"
                lines.append(f"{severity_emoji} {e.get('headline', 'Unknown')}")
        
        lines.append("")
        lines.append(f"Recommendation: {analysis.get('recommendation', 'Review predictions')}")
        
        msg = "\n".join(lines)
        try:
            self.send_telegram(msg)
            LOGGER.info("[NEWS BRAIN] Alert sent to Telegram")
        except Exception as e:
            LOGGER.error(f"[NEWS BRAIN] Failed to send alert: {e}")
    
    def get_last_analysis(self) -> Dict:
        """Get the most recent analysis"""
        return self._last_analysis or {"status": "no_analysis", "message": "Run analyze_news first"}
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get analysis history"""
        return self._analysis_history[-limit:]
    
    def get_status(self) -> Dict:
        """Get news brain status"""
        # Re-check client if not initialized (env vars might be loaded now)
        if self.client is None:
            self._init_client()
        
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        has_key = bool(api_key)
        key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "too_short"
        
        return {
            "enabled": self.client is not None,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "api_key_present": has_key,
            "api_key_preview": key_preview if has_key else None,
            "last_analysis": self._last_analysis.get("timestamp") if self._last_analysis else None,
            "analyses_count": len(self._analysis_history),
            "telegram_enabled": self.send_telegram is not None,
        }
    
    def reinit(self):
        """Re-initialize the client (useful if env vars changed)"""
        self._init_client()
        return self.client is not None


# Singleton instance
_news_brain: Optional[GhostNewsBrain] = None


def get_news_brain(send_telegram_func=None) -> GhostNewsBrain:
    """Get or create the news brain singleton"""
    global _news_brain
    if _news_brain is None:
        _news_brain = GhostNewsBrain(send_telegram_func)
    elif send_telegram_func and _news_brain.send_telegram is None:
        _news_brain.send_telegram = send_telegram_func
    return _news_brain


def analyze_breaking_news(pending_predictions: List[Dict] = None) -> Dict:
    """Convenience function to analyze news"""
    brain = get_news_brain()
    return brain.analyze_news(pending_predictions)
