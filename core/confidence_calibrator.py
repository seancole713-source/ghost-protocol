"""
Ghost Confidence Calibrator
============================
Calibrates prediction confidence scores based on actual accuracy.

Problem:
- Current confidence is heuristic (not based on real outcomes)
- Predictor might say 70% confidence but actual accuracy is 55%
- Need to map predicted confidence → actual accuracy

Solution:
- Analyze historical: "When Ghost says 70%, what's actual accuracy?"
- Build calibration curve: predicted_conf → calibrated_conf
- Only make predictions above quality threshold

Example:
- Ghost says: 75% confidence
- Historical data: 75% predictions are actually 60% accurate
- Calibrated: 60% confidence (more honest)
"""

import logging
import numpy as np
from typing import Any
from collections import defaultdict

LOGGER = logging.getLogger("ghost.confidence_calibrator")


class ConfidenceCalibrator:
    """Calibrate confidence scores based on actual outcomes"""
    
    def __init__(self):
        self.calibration_curve = {}  # confidence_bin -> actual_accuracy
        self.symbol_calibrations = {}  # symbol -> calibration_curve
        
    async def build_calibration(self, min_predictions: int = 50) -> dict[str, Any]:
        """
        Build confidence calibration curves from historical data.
        
        Returns:
            Calibration results and quality thresholds
        """
        LOGGER.info("Building confidence calibration curves...")
        
        # Fetch predictions with confidence + outcomes
        data = await self._fetch_calibration_data()
        
        if len(data) < min_predictions:
            return {
                "ok": False,
                "error": f"Insufficient data: {len(data)} predictions (need {min_predictions}+)",
                "predictions_found": len(data)
            }
        
        LOGGER.info(f"Analyzing {len(data)} predictions for calibration...")
        
        # Build global calibration curve
        global_curve = self._build_calibration_curve(data)
        self.calibration_curve = global_curve
        
        # Build per-symbol calibration
        by_symbol = defaultdict(list)
        for record in data:
            by_symbol[record["symbol"]].append(record)
        
        for symbol, symbol_data in by_symbol.items():
            if len(symbol_data) >= 20:  # Min 20 predictions per symbol
                symbol_curve = self._build_calibration_curve(symbol_data)
                self.symbol_calibrations[symbol] = symbol_curve
        
        # Find quality threshold (where actual accuracy > 65%)
        quality_threshold = self._find_quality_threshold(global_curve)
        
        LOGGER.info(f"✅ Calibration complete. Quality threshold: {quality_threshold:.0%}")
        
        return {
            "ok": True,
            "total_predictions": len(data),
            "calibration_curve": global_curve,
            "symbols_calibrated": len(self.symbol_calibrations),
            "quality_threshold": quality_threshold,
            "recommendation": f"Only make predictions with confidence > {quality_threshold:.0%}"
        }
    
    async def _fetch_calibration_data(self) -> list[dict[str, Any]]:
        """Fetch predictions with confidence and outcomes"""
        import os
        from core.prediction_store import get_prediction_store
        from sqlalchemy import text
        
        store = get_prediction_store()
        
        # Check if PostgreSQL
        is_postgres = os.getenv("DATABASE_URL", "").startswith("postgresql")
        if not is_postgres or not hasattr(store, 'engine'):
            LOGGER.warning("Not using PostgreSQL")
            return []
        
        query = text("""
            SELECT
                symbol,
                confidence,
                was_correct
            FROM ghost_predictions
            WHERE actual_direction IS NOT NULL
              AND was_correct IS NOT NULL
              AND confidence IS NOT NULL
            ORDER BY run_at DESC
            LIMIT 10000
        """)
        
        with store.engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            data = []
            for row in rows:
                data.append({
                    "symbol": row.symbol,
                    "confidence": row.confidence,
                    "was_correct": row.was_correct
                })
            
            return data
    
    def _build_calibration_curve(self, data: list[dict]) -> dict[float, dict]:
        """
        Build calibration curve: predicted_confidence → actual_accuracy
        
        Returns:
            {
                0.5: {"actual_accuracy": 0.48, "count": 120},
                0.6: {"actual_accuracy": 0.55, "count": 98},
                ...
            }
        """
        # Bin confidences into 10% buckets
        bins = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for record in data:
            confidence = record["confidence"]
            
            # Normalize confidence to 0-1 range
            if confidence > 1:
                confidence = confidence / 100
            
            # Bin to nearest 10% (0.5, 0.6, 0.7, etc.)
            bin_val = round(confidence * 10) / 10
            
            bins[bin_val]["total"] += 1
            if record["was_correct"]:
                bins[bin_val]["correct"] += 1
        
        # Calculate actual accuracy per bin
        calibration_curve = {}
        for bin_val, stats in bins.items():
            if stats["total"] >= 5:  # Min 5 predictions per bin
                actual_accuracy = stats["correct"] / stats["total"]
                calibration_curve[bin_val] = {
                    "actual_accuracy": round(actual_accuracy, 3),
                    "count": stats["total"],
                    "calibration_error": round(abs(bin_val - actual_accuracy), 3)
                }
        
        return dict(sorted(calibration_curve.items()))
    
    def _find_quality_threshold(self, calibration_curve: dict) -> float:
        """
        Find confidence threshold where actual accuracy > 65%.
        Only make predictions above this threshold for quality.
        
        Returns:
            Minimum confidence threshold (e.g., 0.70)
        """
        for predicted_conf, stats in sorted(calibration_curve.items(), reverse=True):
            if stats["actual_accuracy"] >= 0.65:
                return predicted_conf
        
        # Fallback: highest confidence bucket
        if calibration_curve:
            return max(calibration_curve.keys())
        
        return 0.70  # Default conservative threshold
    
    def calibrate_confidence(
        self, predicted_confidence: float, symbol: str | None = None
    ) -> dict[str, Any]:
        """
        Calibrate predicted confidence to actual expected accuracy.
        
        Args:
            predicted_confidence: Model's predicted confidence (0-1)
            symbol: Symbol for symbol-specific calibration
            
        Returns:
            {
                "predicted_confidence": 0.75,
                "calibrated_confidence": 0.62,
                "should_predict": True,
                "expected_accuracy": 0.62
            }
        """
        # Normalize to 0-1
        if predicted_confidence > 1:
            predicted_confidence = predicted_confidence / 100
        
        # Use symbol-specific calibration if available
        curve = (
            self.symbol_calibrations.get(symbol, self.calibration_curve)
            if symbol
            else self.calibration_curve
        )
        
        if not curve:
            # No calibration data, return predicted as-is
            return {
                "predicted_confidence": round(predicted_confidence, 3),
                "calibrated_confidence": round(predicted_confidence, 3),
                "should_predict": True,
                "expected_accuracy": round(predicted_confidence, 3),
                "calibrated": False
            }
        
        # Find nearest calibration bin
        bin_val = round(predicted_confidence * 10) / 10
        
        if bin_val in curve:
            calibrated = curve[bin_val]["actual_accuracy"]
        else:
            # Interpolate between nearest bins
            bins = sorted(curve.keys())
            if bin_val < bins[0]:
                calibrated = curve[bins[0]]["actual_accuracy"]
            elif bin_val > bins[-1]:
                calibrated = curve[bins[-1]]["actual_accuracy"]
            else:
                # Linear interpolation
                lower = max(b for b in bins if b <= bin_val)
                upper = min(b for b in bins if b >= bin_val)
                if lower == upper:
                    calibrated = curve[lower]["actual_accuracy"]
                else:
                    weight = (bin_val - lower) / (upper - lower)
                    calibrated = (
                        curve[lower]["actual_accuracy"] * (1 - weight) +
                        curve[upper]["actual_accuracy"] * weight
                    )
        
        # Determine if should predict (quality threshold)
        quality_threshold = self._find_quality_threshold(curve)
        should_predict = calibrated >= 0.65 or predicted_confidence >= quality_threshold
        
        return {
            "predicted_confidence": round(predicted_confidence, 3),
            "calibrated_confidence": round(calibrated, 3),
            "should_predict": should_predict,
            "expected_accuracy": round(calibrated, 3),
            "calibrated": True,
            "quality_threshold": round(quality_threshold, 3)
        }


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get confidence calibrator instance"""
    return ConfidenceCalibrator()
