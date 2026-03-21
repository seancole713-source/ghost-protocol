"""
Prediction Scheduling Monitor (Phase 4.2)

Ensures prediction cycles run on consistent intervals.
Detects missed cycles and scheduling drift.

Ghost Protocol v5 — Session 6
"""

import logging
import time
from typing import Dict, List
from datetime import datetime

LOGGER = logging.getLogger("ghost.scheduler")


class PredictionScheduler:
    """Monitors prediction cycle scheduling and consistency."""
    
    def __init__(self, expected_interval_seconds: int = 3600, drift_tolerance_seconds: int = 300):
        """
        Args:
            expected_interval_seconds: Expected time between prediction cycles (default 60 min)
            drift_tolerance_seconds: Maximum acceptable drift (default 5 min)
        """
        self.expected_interval = expected_interval_seconds
        self.drift_tolerance = drift_tolerance_seconds
        self.cycle_history: List[float] = []  # Timestamps of completed cycles
        self.max_history = 100
        
    def record_cycle(self, timestamp: float = None):
        """
        Record that a prediction cycle completed.
        
        Args:
            timestamp: Cycle completion timestamp (default: now)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.cycle_history.append(timestamp)
        
        # Keep only recent history
        if len(self.cycle_history) > self.max_history:
            self.cycle_history = self.cycle_history[-self.max_history:]
        
        LOGGER.debug(f"Recorded prediction cycle at {datetime.fromtimestamp(timestamp)}")
        
    def get_scheduling_status(self) -> Dict:
        """
        Get scheduling health metrics.
        
        Returns:
            {
                "ok": bool,
                "avg_interval_seconds": float,
                "last_cycle_ago_seconds": float,
                "missed_cycles": int,
                "drift_seconds": float,
                "consistency_score": float (0-100),
                "warnings": List[str]
            }
        """
        if len(self.cycle_history) < 2:
            return {
                "ok": True,
                "avg_interval_seconds": None,
                "last_cycle_ago_seconds": None,
                "missed_cycles": 0,
                "drift_seconds": 0.0,
                "consistency_score": 100.0,
                "warnings": ["Insufficient data - need at least 2 cycles"]
            }
        
        now = time.time()
        warnings = []
        
        # Calculate average interval
        intervals = []
        for i in range(1, len(self.cycle_history)):
            intervals.append(self.cycle_history[i] - self.cycle_history[i-1])
        
        avg_interval = sum(intervals) / len(intervals)
        last_cycle_ago = now - self.cycle_history[-1]
        
        # Calculate drift (deviation from expected interval)
        drift = abs(avg_interval - self.expected_interval)
        
        # Check for missed cycles
        expected_cycles_since_last = last_cycle_ago / self.expected_interval
        missed_cycles = max(0, int(expected_cycles_since_last) - 1)
        
        # Calculate consistency score (0-100)
        # Perfect score = intervals match expected exactly
        # Deduct points for drift and missed cycles
        consistency_score = 100.0
        consistency_score -= min(50, (drift / self.expected_interval) * 100)  # Max 50 points for drift
        consistency_score -= min(30, missed_cycles * 10)  # 10 points per missed cycle, max 30
        consistency_score = max(0, consistency_score)
        
        # Generate warnings
        ok = True
        
        if drift > self.drift_tolerance:
            ok = False
            warnings.append(
                f"⚠️ Scheduling drift detected: {drift:.0f}s "
                f"(expected {self.expected_interval}s, actual avg {avg_interval:.0f}s)"
            )
        
        if missed_cycles > 0:
            ok = False
            warnings.append(
                f"⚠️ Missed {missed_cycles} prediction cycle(s) "
                f"(last cycle was {last_cycle_ago/60:.1f} minutes ago)"
            )
        
        if last_cycle_ago > self.expected_interval * 1.5:
            ok = False
            warnings.append(
                f"⚠️ Prediction cycle stale: {last_cycle_ago/60:.1f} minutes since last cycle "
                f"(expected {self.expected_interval/60:.0f} min)"
            )
        
        for warning in warnings:
            LOGGER.warning(warning)
        
        return {
            "ok": ok,
            "avg_interval_seconds": round(avg_interval, 1),
            "last_cycle_ago_seconds": round(last_cycle_ago, 1),
            "missed_cycles": missed_cycles,
            "drift_seconds": round(drift, 1),
            "consistency_score": round(consistency_score, 1),
            "warnings": warnings
        }


# Global singleton
_SCHEDULER = PredictionScheduler()


def record_prediction_cycle(timestamp: float = None):
    """Record a completed prediction cycle."""
    _SCHEDULER.record_cycle(timestamp)


def get_scheduling_status() -> Dict:
    """Get current scheduling health."""
    return _SCHEDULER.get_scheduling_status()
