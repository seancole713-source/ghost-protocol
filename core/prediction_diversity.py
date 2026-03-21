"""
Prediction Diversity Checker (Phase 4.3)

Ensures predictions aren't heavily skewed to one direction (UP/DOWN).
Balanced predictions indicate healthy model behavior.

Ghost Protocol v5 — Session 6
"""

import logging
from typing import Dict, List
import time

LOGGER = logging.getLogger("ghost.diversity")


class PredictionDiversityChecker:
    """Monitors and enforces prediction diversity."""
    
    def __init__(self, min_diversity_pct: float = 30.0, lookback_hours: int = 24):
        """
        Args:
            min_diversity_pct: Minimum % of predictions in minority direction (default 30%)
            lookback_hours: How far back to analyze (default 24 hours)
        """
        self.min_diversity_pct = min_diversity_pct
        self.lookback_seconds = lookback_hours * 3600
        self._last_check_time = 0
        self._last_diversity_score = 100.0
        
    def check_diversity(self, predictions: List[Dict]) -> Dict:
        """
        Check if predictions are balanced between UP and DOWN.
        
        Args:
            predictions: List of prediction dicts with 'direction' and 'predicted_at' fields
            
        Returns:
            {
                "ok": bool,
                "diversity_score": float (0-100, where 50 = perfect balance),
                "up_pct": float,
                "down_pct": float,
                "up_count": int,
                "down_count": int,
                "total": int,
                "warning": str | None,
                "recommendation": str | None
            }
        """
        now = time.time()
        cutoff = now - self.lookback_seconds
        
        # Filter recent predictions
        recent = [p for p in predictions if p.get("predicted_at", 0) > cutoff]
        
        if not recent:
            return {
                "ok": True,
                "diversity_score": 100.0,
                "up_pct": 50.0,
                "down_pct": 50.0,
                "up_count": 0,
                "down_count": 0,
                "total": 0,
                "warning": "No recent predictions to analyze",
                "recommendation": None
            }
        
        # Count directions
        up_count = sum(1 for p in recent if p.get("direction", "").upper() == "UP")
        down_count = sum(1 for p in recent if p.get("direction", "").upper() == "DOWN")
        total = len(recent)
        
        up_pct = (up_count / total * 100) if total > 0 else 0
        down_pct = (down_count / total * 100) if total > 0 else 0
        
        # Calculate diversity score (0-100, where 50 = perfect balance)
        # Formula: 100 - abs(up_pct - 50) * 2
        # Examples: 50/50 = 100, 60/40 = 80, 70/30 = 60, 80/20 = 40, 90/10 = 20
        diversity_score = 100 - abs(up_pct - 50) * 2
        
        # Check if minority direction is below threshold
        minority_pct = min(up_pct, down_pct)
        ok = minority_pct >= self.min_diversity_pct
        
        warning = None
        recommendation = None
        
        if not ok:
            majority_dir = "UP" if up_count > down_count else "DOWN"
            minority_dir = "DOWN" if up_count > down_count else "UP"
            warning = (
                f"⚠️ Prediction diversity FAILED: {minority_pct:.1f}% {minority_dir} "
                f"(threshold: {self.min_diversity_pct}%). "
                f"Predictions heavily skewed to {majority_dir} ({100-minority_pct:.1f}%)"
            )
            recommendation = (
                f"Model may be biased toward {majority_dir} predictions. "
                f"Consider: 1) Check if regime detector is stuck, "
                f"2) Review feature distributions, "
                f"3) Add contrarian indicators to rebalance predictions"
            )
            LOGGER.warning(warning)
        
        self._last_check_time = now
        self._last_diversity_score = diversity_score
        
        return {
            "ok": ok,
            "diversity_score": round(diversity_score, 1),
            "up_pct": round(up_pct, 1),
            "down_pct": round(down_pct, 1),
            "up_count": up_count,
            "down_count": down_count,
            "total": total,
            "warning": warning,
            "recommendation": recommendation
        }
    
    def get_last_score(self) -> float:
        """Get last diversity score without re-checking."""
        return self._last_diversity_score


# Global singleton
_DIVERSITY_CHECKER = PredictionDiversityChecker()


def check_prediction_diversity(predictions: List[Dict]) -> Dict:
    """Convenience function for global diversity checker."""
    return _DIVERSITY_CHECKER.check_diversity(predictions)


def get_diversity_score() -> float:
    """Get last diversity score."""
    return _DIVERSITY_CHECKER.get_last_score()
