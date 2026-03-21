"""
Duplicate Predictions Checker (Phase 5.6)

Detects and prevents duplicate predictions that could skew results.
Ensures data integrity in predictions database.

Ghost Protocol v5 — Session 6
"""

import logging
from typing import Dict, List, Set
import time

LOGGER = logging.getLogger("ghost.duplicates")


class DuplicateChecker:
    """Detects duplicate predictions in database."""
    
    def __init__(self, tolerance_seconds: int = 60):
        """
        Args:
            tolerance_seconds: Window within which duplicates are detected (default 60s)
        """
        self.tolerance_seconds = tolerance_seconds
        self._last_check_time = 0
        self._duplicate_count = 0
        
    def check_for_duplicates(self, predictions: List[Dict]) -> Dict:
        """
        Scan predictions for duplicates.
        
        A duplicate is defined as: same symbol, same direction, within tolerance window.
        
        Args:
            predictions: List of prediction dicts with symbol, direction, predicted_at
            
        Returns:
            {
                "ok": bool (False if duplicates found),
                "duplicate_count": int,
                "duplicates": List[Dict] (details of duplicate sets),
                "total_predictions": int,
                "duplicate_pct": float
            }
        """
        if not predictions:
            return {
                "ok": True,
                "duplicate_count": 0,
                "duplicates": [],
                "total_predictions": 0,
                "duplicate_pct": 0.0
            }
        
        # Sort by timestamp
        sorted_preds = sorted(predictions, key=lambda p: p.get("predicted_at", 0))
        
        duplicates = []
        seen_keys: Dict[str, List[Dict]] = {}  # key -> [predictions]
        
        for pred in sorted_preds:
            symbol = pred.get("symbol", "").upper()
            direction = pred.get("direction", "").upper()
            timestamp = pred.get("predicted_at", 0)
            
            if not symbol or not direction:
                continue
            
            # Create key: symbol + direction
            key = f"{symbol}_{direction}"
            
            if key not in seen_keys:
                seen_keys[key] = [pred]
                continue
            
            # Check if this is a duplicate (within tolerance window)
            for existing in seen_keys[key]:
                existing_ts = existing.get("predicted_at", 0)
                time_diff = abs(timestamp - existing_ts)
                
                if time_diff <= self.tolerance_seconds:
                    # Found a duplicate!
                    dup_entry = {
                        "symbol": symbol,
                        "direction": direction,
                        "timestamps": [existing_ts, timestamp],
                        "time_diff_seconds": int(time_diff),
                        "pred_ids": [existing.get("id"), pred.get("id")]
                    }
                    duplicates.append(dup_entry)
                    LOGGER.warning(
                        f"⚠️ Duplicate prediction: {symbol} {direction} "
                        f"within {time_diff:.0f}s (IDs: {dup_entry['pred_ids']})"
                    )
                    break
            
            seen_keys[key].append(pred)
        
        duplicate_count = len(duplicates)
        total = len(predictions)
        duplicate_pct = (duplicate_count / total * 100) if total > 0 else 0.0
        
        ok = duplicate_count == 0
        
        self._last_check_time = time.time()
        self._duplicate_count = duplicate_count
        
        return {
            "ok": ok,
            "duplicate_count": duplicate_count,
            "duplicates": duplicates,
            "total_predictions": total,
            "duplicate_pct": round(duplicate_pct, 2)
        }
    
    def get_last_count(self) -> int:
        """Get last duplicate count without re-checking."""
        return self._duplicate_count


# Global singleton
_DUPLICATE_CHECKER = DuplicateChecker()


def check_for_duplicates(predictions: List[Dict]) -> Dict:
    """Convenience function for global duplicate checker."""
    return _DUPLICATE_CHECKER.check_for_duplicates(predictions)


def get_duplicate_count() -> int:
    """Get last duplicate count."""
    return _DUPLICATE_CHECKER.get_last_count()
