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


    def calibrate_with_signals(
        self,
        features: dict[str, Any],
        base_direction: str,
        base_confidence: float = 0.45
    ) -> dict[str, Any]:
        """
        SIGNAL-BASED calibration: Adjust confidence based on feature alignment.
        
        This is the CRITICAL method that transforms flat 45% confidence
        into 60-75% by detecting strong technical signals.
        
        Args:
            features: Feature dict from orchestrator
            base_direction: "UP", "DOWN", or "FLAT"
            base_confidence: Starting confidence (default: 0.45)
        
        Returns:
            {
                "calibrated_confidence": float (0.05-0.85),
                "adjustments": dict,
                "signal_count": int,
                "signals_fired": list[str]
            }
        """
        confidence = base_confidence
        adjustments = {}
        signals_fired = []
        
        # Extract features
        rsi = features.get("RSI_14")
        macd_hist = features.get("MACD_HISTOGRAM")
        bb_position = features.get("BOLLINGER_POSITION")
        volume_spike = features.get("VOLUME_SPIKE", 0)
        sentiment = features.get("NEWS_SENTIMENT_SCORE")
        market_trend = features.get("SPY_MOMENTUM")
        
        # RSI SIGNALS
        if rsi is not None:
            if rsi < 30 and base_direction == "UP":
                confidence += 0.10
                adjustments["rsi_oversold"] = 0.10
                signals_fired.append("RSI_OVERSOLD_BUY")
            elif rsi > 70 and base_direction == "DOWN":
                confidence += 0.10
                adjustments["rsi_overbought"] = 0.10
                signals_fired.append("RSI_OVERBOUGHT_SELL")
        
        # MACD SIGNALS
        if macd_hist is not None:
            if macd_hist > 0 and base_direction == "UP":
                confidence += 0.08
                adjustments["macd_bullish"] = 0.08
                signals_fired.append("MACD_BULLISH")
            elif macd_hist < 0 and base_direction == "DOWN":
                confidence += 0.08
                adjustments["macd_bearish"] = 0.08
                signals_fired.append("MACD_BEARISH")
        
        # BOLLINGER BAND SIGNALS
        if bb_position is not None:
            if bb_position < 0.2 and base_direction == "UP":
                confidence += 0.07
                adjustments["bb_bounce_buy"] = 0.07
                signals_fired.append("BB_BOUNCE_BUY")
            elif bb_position > 0.8 and base_direction == "DOWN":
                confidence += 0.07
                adjustments["bb_bounce_sell"] = 0.07
                signals_fired.append("BB_BOUNCE_SELL")
        
        # VOLUME SIGNALS
        if volume_spike is not None and volume_spike > 2.0:
            confidence += 0.08
            adjustments["volume_surge"] = 0.08
            signals_fired.append("VOLUME_SURGE")
        elif volume_spike is not None and volume_spike < 0.5:
            confidence -= 0.05
            adjustments["volume_weak"] = -0.05
        
        # SENTIMENT SIGNALS
        if sentiment is not None:
            if sentiment > 0.5 and base_direction == "UP":
                confidence += 0.07
                adjustments["news_bullish"] = 0.07
                signals_fired.append("NEWS_BULLISH")
            elif sentiment < -0.5 and base_direction == "DOWN":
                confidence += 0.07
                adjustments["news_bearish"] = 0.07
                signals_fired.append("NEWS_BEARISH")
        
        # MARKET CONTEXT
        if market_trend is not None:
            if (market_trend > 0 and base_direction == "UP") or \
               (market_trend < 0 and base_direction == "DOWN"):
                confidence += 0.05
                adjustments["market_aligned"] = 0.05
                signals_fired.append("MARKET_TAILWIND")
        
        # ALIGNMENT BONUS
        if len(signals_fired) >= 5:
            confidence += 0.15
            adjustments["full_alignment"] = 0.15
        elif len(signals_fired) >= 3:
            confidence += 0.08
            adjustments["partial_alignment"] = 0.08
        
        # Clamp to 5%-85%
        confidence = max(0.05, min(0.85, confidence))
        
        LOGGER.debug(
            f"Signal-based calibration: {base_confidence:.2f} → {confidence:.2f} "
            f"({len(signals_fired)} signals)"
        )
        
        return {
            "calibrated_confidence": round(confidence, 3),
            "adjustments": {k: round(v, 3) for k, v in adjustments.items()},
            "signal_count": len(signals_fired),
            "signals_fired": signals_fired,
            "alignment_score": round(len(signals_fired) / 10.0, 2)
        }


def get_confidence_calibrator() -> ConfidenceCalibrator:
    """Get confidence calibrator instance"""
    return ConfidenceCalibrator()


def calibrate_confidence_with_signals(
    features: dict[str, Any],
    base_direction: str,
    base_confidence: float = 0.45
) -> dict[str, Any]:
    """
    Quick signal-based calibration (recommended for production).
    
    Usage:
        from core.confidence_calibrator import calibrate_confidence_with_signals
        
        result = calibrate_confidence_with_signals(features, "UP", 0.45)
        confidence = result["calibrated_confidence"]  # e.g., 0.68
    """
    calibrator = get_confidence_calibrator()
    return calibrator.calibrate_with_signals(features, base_direction, base_confidence)
