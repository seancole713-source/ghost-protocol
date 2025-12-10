"""
GHOST MAXIMUM v2.0 - Prediction Explainer
Explains WHY Ghost made a prediction (transparency + trust)
"""
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class PredictionExplainer:
    """
    Generate human-readable explanations for predictions
    Shows which factors contributed most to the decision
    """
    
    async def explain_prediction(
        self,
        symbol: str,
        final_prediction: dict[str, Any],
        ensemble_result: dict[str, Any],
        volume_result: dict[str, Any],
        regime_result: dict[str, Any],
        timeframe_result: dict[str, Any],
        quality_check: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate comprehensive explanation
        
        Returns:
            {
                "summary": "Ghost predicts UP with 75% confidence",
                "key_factors": [...],
                "supporting_evidence": [...],
                "risks": [...],
                "confidence_breakdown": {...}
            }
        """
        direction = final_prediction.get("direction", "FLAT")
        confidence = final_prediction.get("confidence", 0.5)
        
        # Generate summary
        summary = self._generate_summary(symbol, direction, confidence)
        
        # Extract key factors
        key_factors = self._extract_key_factors(
            ensemble_result, volume_result, regime_result, timeframe_result
        )
        
        # Supporting evidence
        supporting_evidence = self._collect_evidence(
            direction, ensemble_result, volume_result, regime_result, timeframe_result
        )
        
        # Risk factors
        risks = self._identify_risks(
            quality_check, ensemble_result, volume_result, regime_result
        )
        
        # Confidence breakdown
        confidence_breakdown = self._breakdown_confidence(
            ensemble_result, volume_result, regime_result, timeframe_result
        )
        
        return {
            "symbol": symbol,
            "summary": summary,
            "key_factors": key_factors,
            "supporting_evidence": supporting_evidence,
            "risks": risks,
            "confidence_breakdown": confidence_breakdown,
            "quality_score": quality_check.get("quality_score", 0.0)
        }
    
    def _generate_summary(self, symbol: str, direction: str, confidence: float) -> str:
        """Generate one-sentence summary"""
        conf_pct = int(confidence * 100)
        
        if direction == "UP":
            return f"Ghost predicts {symbol} will move UP in the next 6 hours with {conf_pct}% confidence"
        elif direction == "DOWN":
            return f"Ghost predicts {symbol} will move DOWN in the next 6 hours with {conf_pct}% confidence"
        else:
            return f"Ghost advises HOLDING {symbol} - insufficient confidence for directional trade ({conf_pct}%)"
    
    def _extract_key_factors(
        self,
        ensemble: dict[str, Any],
        volume: dict[str, Any],
        regime: dict[str, Any],
        timeframe: dict[str, Any]
    ) -> list[str]:
        """Extract the 3-5 most important factors"""
        factors = []
        
        # Ensemble agreement
        agreement = ensemble.get("agreement", "weak")
        if agreement == "strong":
            factors.append(f"🎯 Strong strategy agreement ({ensemble.get('strategy_breakdown', {}).get('buy', 0) or ensemble.get('strategy_breakdown', {}).get('sell', 0)}/5 strategies)")
        
        # Volume patterns
        volume_pattern = volume.get("pattern", "neutral")
        if volume_pattern in ["accumulation", "distribution", "climax"]:
            factors.append(f"📊 Volume shows {volume_pattern} (strength: {volume.get('strength', 0):.0%})")
        
        # Market regime
        regime_type = regime.get("regime", "sideways")
        regime_conf = regime.get("confidence", 0.5)
        if regime_conf > 0.7:
            factors.append(f"📈 Clear {regime_type} market regime (conf: {regime_conf:.0%})")
        
        # Timeframe alignment
        alignment = timeframe.get("alignment", "weak")
        if alignment in ["strong", "moderate"]:
            factors.append(f"⏰ {alignment.capitalize()} multi-timeframe alignment")
        
        # OBV trend
        obv_trend = volume.get("obv_trend", "flat")
        if obv_trend != "flat":
            factors.append(f"💹 On-Balance Volume trending {obv_trend}")
        
        return factors[:5]  # Top 5 only
    
    def _collect_evidence(
        self,
        direction: str,
        ensemble: dict[str, Any],
        volume: dict[str, Any],
        regime: dict[str, Any],
        timeframe: dict[str, Any]
    ) -> list[str]:
        """Collect supporting evidence for the prediction"""
        evidence = []
        
        # Strategy votes
        votes = ensemble.get("strategy_breakdown", {})
        if direction == "UP":
            buy_votes = votes.get("buy", 0)
            if buy_votes >= 3:
                evidence.append(f"{buy_votes} strategies recommend BUY")
        elif direction == "DOWN":
            sell_votes = votes.get("sell", 0)
            if sell_votes >= 3:
                evidence.append(f"{sell_votes} strategies recommend SELL")
        
        # Volume confirmation
        volume_signal = volume.get("signal", "HOLD")
        if volume_signal == direction or (direction == "UP" and volume_signal == "BUY"):
            evidence.append(f"Volume analysis confirms {direction}")
        
        # Regime support
        regime_type = regime.get("regime", "sideways")
        if (direction == "UP" and regime_type == "bull") or \
           (direction == "DOWN" and regime_type == "bear"):
            evidence.append(f"Market regime supports {direction}")
        
        # Timeframe trends
        tf_signal = timeframe.get("signal", "HOLD")
        if tf_signal == direction or (direction == "UP" and tf_signal == "BUY"):
            trends = timeframe.get("trend_strength", {})
            evidence.append(f"Multiple timeframes confirm (1h: {trends.get('1h', 0):.1f}, 4h: {trends.get('4h', 0):.1f}, 6h: {trends.get('6h', 0):.1f})")
        
        return evidence
    
    def _identify_risks(
        self,
        quality: dict[str, Any],
        ensemble: dict[str, Any],
        volume: dict[str, Any],
        regime: dict[str, Any]
    ) -> list[str]:
        """Identify risk factors that could invalidate the prediction"""
        risks = []
        
        # Quality conflicts
        conflicts = quality.get("conflicts", [])
        if conflicts:
            risks.append(f"⚠️ Signal conflicts: {conflicts[0]}")
        
        # Low quality score
        quality_score = quality.get("quality_score", 1.0)
        if quality_score < 0.6:
            risks.append(f"⚠️ Low quality score ({quality_score:.0%})")
        
        # Weak ensemble agreement
        agreement = ensemble.get("agreement", "strong")
        if agreement == "weak":
            risks.append("⚠️ Strategies disagree on direction")
        
        # Volume divergence
        if volume.get("price_volume_divergence", False):
            risks.append("⚠️ Price-volume divergence detected")
        
        # High volatility
        volatility = regime.get("volatility", "normal")
        if volatility == "high":
            risks.append("⚠️ High volatility increases risk")
        
        # Sideways regime
        regime_type = regime.get("regime", "sideways")
        if regime_type == "sideways":
            risks.append("⚠️ Sideways market - lower predictability")
        
        return risks[:5]  # Top 5 risks
    
    def _breakdown_confidence(
        self,
        ensemble: dict[str, Any],
        volume: dict[str, Any],
        regime: dict[str, Any],
        timeframe: dict[str, Any]
    ) -> dict[str, float]:
        """Break down confidence by component"""
        return {
            "ensemble_confidence": ensemble.get("confidence", 0.5),
            "volume_confidence": volume.get("confidence", 0.5),
            "regime_confidence": regime.get("confidence", 0.5),
            "timeframe_confidence": timeframe.get("confidence", 0.5),
            "average": (
                ensemble.get("confidence", 0.5) +
                volume.get("confidence", 0.5) +
                regime.get("confidence", 0.5) +
                timeframe.get("confidence", 0.5)
            ) / 4.0
        }


# Singleton
_EXPLAINER = None


def get_explainer() -> PredictionExplainer:
    """Get singleton explainer"""
    global _EXPLAINER
    if _EXPLAINER is None:
        _EXPLAINER = PredictionExplainer()
    return _EXPLAINER
