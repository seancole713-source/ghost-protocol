"""
Model Agreement System for Ghost Protocol
Requires consensus from multiple models for high-confidence predictions
"""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelAgreement:
    """Check agreement across multiple prediction models"""
    
    def __init__(self):
        self.enabled = os.getenv("MODEL_AGREEMENT_ENABLED", "1") == "1"
        self.min_agreement = float(os.getenv("MODEL_MIN_AGREEMENT", "0.67"))  # 2/3 agreement
        self.confidence_boost = float(os.getenv("MODEL_AGREEMENT_BOOST", "0.10"))
        self.confidence_penalty = float(os.getenv("MODEL_DISAGREEMENT_PENALTY", "0.15"))
    
    def check_agreement(self, signals: Dict[str, Dict]) -> Dict:
        """
        Check agreement across model signals
        
        Args:
            signals: Dict of model_name -> {direction, confidence, ...}
            
        Returns:
            Dict with agreed_direction, agreement_pct, adjustments
        """
        if not signals:
            return {
                "agreed": False,
                "error": "No signals provided"
            }
        
        directions = []
        weights = []
        
        for model_name, signal in signals.items():
            direction = signal.get("direction")
            confidence = signal.get("confidence", 0.5)
            
            if direction in ["UP", "DOWN"]:
                directions.append(direction)
                weights.append(confidence)
        
        if not directions:
            return {
                "agreed": False,
                "error": "No valid directions in signals"
            }
        
        # Count votes (weighted by confidence)
        up_weight = sum(w for d, w in zip(directions, weights) if d == "UP")
        down_weight = sum(w for d, w in zip(directions, weights) if d == "DOWN")
        total_weight = up_weight + down_weight
        
        # Simple count
        up_count = directions.count("UP")
        down_count = directions.count("DOWN")
        total_count = len(directions)
        
        # Determine consensus
        if up_count > down_count:
            consensus_direction = "UP"
            agreement_pct = up_count / total_count
            weighted_agreement = up_weight / total_weight if total_weight > 0 else 0
        elif down_count > up_count:
            consensus_direction = "DOWN"
            agreement_pct = down_count / total_count
            weighted_agreement = down_weight / total_weight if total_weight > 0 else 0
        else:
            consensus_direction = "MIXED"
            agreement_pct = 0.5
            weighted_agreement = 0.5
        
        # Determine if agreed
        agreed = agreement_pct >= self.min_agreement
        
        # Calculate confidence adjustment
        if agreed and consensus_direction != "MIXED":
            adjustment = self.confidence_boost * (agreement_pct - 0.5)
        else:
            adjustment = -self.confidence_penalty * (1 - agreement_pct)
        
        return {
            "agreed": agreed,
            "consensus_direction": consensus_direction,
            "agreement_pct": round(agreement_pct * 100, 0),
            "weighted_agreement_pct": round(weighted_agreement * 100, 0),
            "models_checked": total_count,
            "up_votes": up_count,
            "down_votes": down_count,
            "confidence_adjustment": round(adjustment, 3),
            "recommendation": "PROCEED" if agreed else "REDUCE_SIZE" if agreement_pct >= 0.5 else "SKIP"
        }
    
    def aggregate_predictions(self, models: List[Dict]) -> Dict:
        """
        Aggregate predictions from multiple models into consensus
        
        Args:
            models: List of model predictions with direction, confidence
            
        Returns:
            Consensus prediction with adjusted confidence
        """
        signals = {f"model_{i}": m for i, m in enumerate(models)}
        agreement = self.check_agreement(signals)
        
        if not agreement.get("agreed"):
            return {
                "direction": agreement.get("consensus_direction", "NEUTRAL"),
                "confidence": 0.4,  # Low confidence for disagreement
                "agreement": agreement,
                "suppressed": agreement.get("recommendation") == "SKIP"
            }
        
        # Calculate consensus confidence
        confidences = [
            m.get("confidence", 0.5) for m in models 
            if m.get("direction") == agreement["consensus_direction"]
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Apply adjustment
        final_confidence = min(0.85, max(0.4, avg_confidence + agreement["confidence_adjustment"]))
        
        return {
            "direction": agreement["consensus_direction"],
            "confidence": round(final_confidence, 3),
            "agreement": agreement,
            "suppressed": False
        }


# Singleton
_agreement: Optional[ModelAgreement] = None


def get_model_agreement() -> ModelAgreement:
    """Get or create ModelAgreement singleton"""
    global _agreement
    if _agreement is None:
        _agreement = ModelAgreement()
    return _agreement


def check_model_agreement(signals: Dict) -> Dict:
    """Check agreement across model signals"""
    return get_model_agreement().check_agreement(signals)


def aggregate_model_predictions(models: List[Dict]) -> Dict:
    """Aggregate predictions from multiple models"""
    return get_model_agreement().aggregate_predictions(models)
