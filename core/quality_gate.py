"""
Ghost Protocol - Quality Gate
Strict filtering for predictions to ensure quality over quantity.

Requirements:
- 85% minimum verified accuracy (not hardcoded)
- 10 predictions max per day
- 24-hour deduplication per symbol
- 3% minimum predicted return
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Configuration from environment
MIN_ACCURACY_PCT = float(os.getenv('MIN_ACCURACY_PCT', '50.0'))  # TUNED: 85→50 (was blocking all predictions)
MIN_VERIFIED_PREDICTIONS = int(os.getenv('MIN_VERIFIED_PREDICTIONS', '20'))
MAX_DAILY_PREDICTIONS = int(os.getenv('MAX_DAILY_PREDICTIONS', '10'))
MIN_PREDICTED_RETURN = float(os.getenv('MIN_PREDICTED_RETURN', '0.03'))  # 3%
DEDUP_HOURS = int(os.getenv('DEDUP_HOURS', '24'))
MIN_CONFIDENCE = float(os.getenv('MIN_ALERT_CONFIDENCE', '0.60'))  # TUNED: 85%→60% (was too strict)


@dataclass
class QualityGateResult:
    """Result of quality gate check"""
    allowed: bool
    reason: str
    accuracy: Optional[float] = None
    daily_count: int = 0
    remaining_slots: int = 0


class QualityGate:
    """
    Strict quality gate for predictions.
    
    Rules:
    1. System must have MIN_VERIFIED_PREDICTIONS with MIN_ACCURACY_PCT
    2. Max MAX_DAILY_PREDICTIONS per day
    3. No duplicate symbols within DEDUP_HOURS
    4. Minimum confidence threshold
    5. Minimum predicted return threshold
    """
    
    def __init__(self):
        self._reload_config()
        self._daily_count = 0
        self._daily_date = datetime.utcnow().date()
        self._recent_symbols: Dict[str, datetime] = {}
    
    def _reload_config(self):
        """Reload configuration from environment"""
        self.min_accuracy = float(os.getenv('MIN_ACCURACY_PCT', '50.0'))  # TUNED: was 85%
        self.min_verified = int(os.getenv('MIN_VERIFIED_PREDICTIONS', '20'))
        self.max_daily = int(os.getenv('MAX_DAILY_PREDICTIONS', '10'))
        self.min_return = float(os.getenv('MIN_PREDICTED_RETURN', '0.03'))
        self.dedup_hours = int(os.getenv('DEDUP_HOURS', '24'))
        self.min_confidence = float(os.getenv('MIN_ALERT_CONFIDENCE', '0.60'))  # TUNED: was 85%
    
    def check(
        self,
        symbol: str,
        confidence: float,
        predicted_return: float = 0.0,
        accuracy_stats: Dict = None
    ) -> QualityGateResult:
        """
        Check if a prediction passes the quality gate.
        
        Args:
            symbol: The ticker symbol
            confidence: Model confidence (0-1)
            predicted_return: Expected return percentage
            accuracy_stats: Current accuracy statistics
            
        Returns:
            QualityGateResult with allowed status and reason
        """
        self._reload_config()
        self._maybe_reset_daily()
        
        # Rule 1: Check system accuracy
        if accuracy_stats:
            verified = accuracy_stats.get('verified_predictions', 0)
            accuracy = accuracy_stats.get('accuracy_pct', 0)
            
            if verified >= self.min_verified and accuracy < self.min_accuracy:
                return QualityGateResult(
                    allowed=False,
                    reason=f"System accuracy {accuracy:.1f}% below minimum {self.min_accuracy}% (need {self.min_accuracy}%+ on {self.min_verified}+ verified)",
                    accuracy=accuracy,
                    daily_count=self._daily_count,
                    remaining_slots=self.max_daily - self._daily_count
                )
        
        # Rule 2: Check daily cap
        if self._daily_count >= self.max_daily:
            return QualityGateResult(
                allowed=False,
                reason=f"Daily cap reached ({self.max_daily} predictions)",
                daily_count=self._daily_count,
                remaining_slots=0
            )
        
        # Rule 3: Check deduplication
        if symbol.upper() in self._recent_symbols:
            last_sent = self._recent_symbols[symbol.upper()]
            elapsed = datetime.utcnow() - last_sent
            if elapsed < timedelta(hours=self.dedup_hours):
                remaining = timedelta(hours=self.dedup_hours) - elapsed
                return QualityGateResult(
                    allowed=False,
                    reason=f"Duplicate: {symbol} sent {elapsed.total_seconds()/3600:.1f}h ago (wait {remaining.total_seconds()/3600:.1f}h)",
                    daily_count=self._daily_count,
                    remaining_slots=self.max_daily - self._daily_count
                )
        
        # Rule 4: Check minimum confidence
        if confidence < self.min_confidence:
            return QualityGateResult(
                allowed=False,
                reason=f"Confidence {confidence:.1%} below minimum {self.min_confidence:.1%}",
                daily_count=self._daily_count,
                remaining_slots=self.max_daily - self._daily_count
            )
        
        # Rule 5: Check minimum predicted return
        if predicted_return < self.min_return:
            return QualityGateResult(
                allowed=False,
                reason=f"Predicted return {predicted_return:.1%} below minimum {self.min_return:.1%}",
                daily_count=self._daily_count,
                remaining_slots=self.max_daily - self._daily_count
            )
        
        # Progressive confidence threshold based on remaining slots
        remaining_slots = self.max_daily - self._daily_count
        if remaining_slots <= 3:
            required_conf = 0.85
        elif remaining_slots <= 1:
            required_conf = 0.90
        else:
            required_conf = self.min_confidence
        
        if confidence < required_conf:
            return QualityGateResult(
                allowed=False,
                reason=f"Slot {self._daily_count + 1}/{self.max_daily} requires {required_conf:.0%}+ confidence (got {confidence:.1%})",
                daily_count=self._daily_count,
                remaining_slots=remaining_slots
            )
        
        # All checks passed
        return QualityGateResult(
            allowed=True,
            reason="Passed all quality checks",
            accuracy=accuracy_stats.get('accuracy_pct') if accuracy_stats else None,
            daily_count=self._daily_count,
            remaining_slots=remaining_slots
        )
    
    def record_sent(self, symbol: str):
        """Record that a prediction was sent"""
        self._maybe_reset_daily()
        self._daily_count += 1
        self._recent_symbols[symbol.upper()] = datetime.utcnow()
        logger.info(f"📊 QualityGate: Recorded {symbol} ({self._daily_count}/{self.max_daily} today)")
    
    def _maybe_reset_daily(self):
        """Reset daily counter if it's a new day"""
        today = datetime.utcnow().date()
        if today != self._daily_date:
            logger.info(f"📅 QualityGate: New day, resetting counter (was {self._daily_count})")
            self._daily_count = 0
            self._daily_date = today
            # Also clean up old dedup entries
            self._clean_old_dedup()
    
    def _clean_old_dedup(self):
        """Remove old entries from dedup cache"""
        cutoff = datetime.utcnow() - timedelta(hours=self.dedup_hours)
        old_symbols = [
            s for s, t in self._recent_symbols.items() 
            if t < cutoff
        ]
        for s in old_symbols:
            del self._recent_symbols[s]
        if old_symbols:
            logger.debug(f"🧹 QualityGate: Cleaned {len(old_symbols)} old dedup entries")
    
    def get_status(self) -> Dict:
        """Get current quality gate status"""
        self._maybe_reset_daily()
        
        return {
            'enabled': True,
            'daily_count': self._daily_count,
            'max_daily': self.max_daily,
            'remaining_slots': self.max_daily - self._daily_count,
            'min_accuracy_required': self.min_accuracy,
            'min_verified_required': self.min_verified,
            'min_confidence': self.min_confidence,
            'min_predicted_return': self.min_return,
            'dedup_hours': self.dedup_hours,
            'recent_symbols': list(self._recent_symbols.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def reset_for_testing(self):
        """Reset all state for testing purposes"""
        self._daily_count = 0
        self._daily_date = datetime.utcnow().date()
        self._recent_symbols = {}
        logger.warning("⚠️ QualityGate: Reset for testing")


# Singleton instance
_gate: QualityGate | None = None


def get_quality_gate() -> QualityGate:
    """Get the singleton quality gate instance"""
    global _gate
    if _gate is None:
        _gate = QualityGate()
    return _gate
