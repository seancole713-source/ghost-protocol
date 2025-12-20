"""
Ghost Protocol - Honest Telegram Formatter
Formats predictions with REAL verified accuracy, not hardcoded lies.

Every message must show:
- Actual verified accuracy (not "85%")
- Number of verified predictions
- Win/Loss record
- Disclaimer if accuracy is below threshold or unverified
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class HonestTelegramFormatter:
    """
    Formats Telegram messages with REAL accuracy data.
    No more hardcoded "85% Accuracy" lies.
    """
    
    def __init__(self):
        self.min_verified_for_display = int(os.getenv('MIN_VERIFIED_FOR_ACCURACY', '10'))
    
    def format_prediction(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        target_price: float = None,
        entry_price: float = None,
        stop_loss: float = None,
        predicted_return: float = None,
        horizon: str = "24h",
        accuracy_stats: Dict = None,
        pattern_signals: Dict = None
    ) -> str:
        """
        Format a prediction message with honest accuracy stats.
        """
        # Header
        emoji = "🟢" if direction.upper() == "LONG" else "🔴"
        lines = [
            f"{emoji} **{symbol}** - {direction.upper()}",
            "",
        ]
        
        # Price targets
        if entry_price:
            lines.append(f"📍 Entry: ${entry_price:,.2f}")
        if target_price:
            lines.append(f"🎯 Target: ${target_price:,.2f}")
        if stop_loss:
            lines.append(f"🛡️ Stop Loss: ${stop_loss:,.2f}")
        
        # Predicted return
        if predicted_return:
            lines.append(f"📈 Expected: {predicted_return:+.1%}")
        
        # Confidence
        lines.append(f"🔮 Confidence: {confidence:.0%}")
        lines.append(f"⏱️ Horizon: {horizon}")
        lines.append("")
        
        # HONEST accuracy section
        lines.append(self._format_accuracy_section(accuracy_stats))
        
        # Pattern signals if available
        if pattern_signals:
            signals_str = self._format_pattern_signals(pattern_signals)
            if signals_str:
                lines.append("")
                lines.append(signals_str)
        
        # Timestamp
        lines.append("")
        lines.append(f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_")
        
        return "\n".join(lines)
    
    def _format_accuracy_section(self, accuracy_stats: Dict = None) -> str:
        """
        Format the accuracy section with HONEST data.
        Shows real verified stats or clearly states unverified.
        """
        if not accuracy_stats:
            return "⚠️ **Accuracy: UNVERIFIED** (no tracked predictions yet)"
        
        verified = accuracy_stats.get('verified_predictions', 0)
        wins = accuracy_stats.get('wins', 0)
        losses = accuracy_stats.get('losses', 0)
        accuracy = accuracy_stats.get('accuracy_pct', 0)
        avg_return = accuracy_stats.get('avg_return', 0)
        
        if verified < self.min_verified_for_display:
            return (
                f"⚠️ **Accuracy: BUILDING** ({verified}/{self.min_verified_for_display} verified)\n"
                f"└ Record so far: {wins}W / {losses}L"
            )
        
        # Show real accuracy with record
        if accuracy >= 70:
            emoji = "✅"
            status = "VERIFIED"
        elif accuracy >= 50:
            emoji = "⚡"
            status = "MODERATE"
        else:
            emoji = "⚠️"
            status = "LOW"
        
        return (
            f"{emoji} **Accuracy: {accuracy:.1f}% {status}**\n"
            f"└ Record: {wins}W / {losses}L ({verified} verified)\n"
            f"└ Avg Return: {avg_return:+.1f}%"
        )
    
    def _format_pattern_signals(self, signals: Dict) -> str:
        """Format pattern signals section"""
        if not signals:
            return ""
        
        lines = ["📊 **Pattern Signals:**"]
        
        bullish = signals.get('bullish', [])
        bearish = signals.get('bearish', [])
        
        if bullish:
            lines.append(f"🟢 Bullish: {', '.join(bullish[:3])}")
        if bearish:
            lines.append(f"🔴 Bearish: {', '.join(bearish[:3])}")
        
        return "\n".join(lines)
    
    def format_daily_summary(self, accuracy_stats: Dict, daily_stats: Dict = None) -> str:
        """Format a daily summary message"""
        verified = accuracy_stats.get('verified_predictions', 0)
        wins = accuracy_stats.get('wins', 0)
        losses = accuracy_stats.get('losses', 0)
        accuracy = accuracy_stats.get('accuracy_pct', 0)
        avg_return = accuracy_stats.get('avg_return', 0)
        
        lines = [
            "📊 **Ghost Daily Summary**",
            "",
        ]
        
        if daily_stats:
            sent_today = daily_stats.get('predictions_sent', 0)
            verified_today = daily_stats.get('predictions_verified', 0)
            lines.append(f"Today: {sent_today} predictions sent, {verified_today} verified")
            lines.append("")
        
        lines.extend([
            "**All-Time Stats:**",
            f"• Total Verified: {verified}",
            f"• Record: {wins}W / {losses}L",
            f"• Accuracy: {accuracy:.1f}%",
            f"• Avg Return: {avg_return:+.1f}%",
            "",
            f"_Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_"
        ])
        
        return "\n".join(lines)
    
    def format_accuracy_alert(self, accuracy_stats: Dict, threshold: float = 85.0) -> str:
        """Format an accuracy alert when system falls below threshold"""
        accuracy = accuracy_stats.get('accuracy_pct', 0)
        verified = accuracy_stats.get('verified_predictions', 0)
        
        if accuracy >= threshold:
            return None  # No alert needed
        
        return (
            f"⚠️ **ACCURACY ALERT**\n\n"
            f"System accuracy ({accuracy:.1f}%) has fallen below threshold ({threshold}%).\n"
            f"Predictions are paused until accuracy improves.\n\n"
            f"Verified: {verified} | Min Required: {threshold}%\n\n"
            f"_This is an automated system health alert._"
        )


# Singleton instance
_formatter: HonestTelegramFormatter | None = None


def get_honest_formatter() -> HonestTelegramFormatter:
    """Get the singleton formatter instance"""
    global _formatter
    if _formatter is None:
        _formatter = HonestTelegramFormatter()
    return _formatter


def format_honest_prediction(
    symbol: str,
    direction: str,
    confidence: float,
    accuracy_stats: Dict = None,
    **kwargs
) -> str:
    """Convenience function to format a prediction"""
    return get_honest_formatter().format_prediction(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        accuracy_stats=accuracy_stats,
        **kwargs
    )
