"""
Ghost Feature Diagnostics
=========================
Instruments feature extraction pipeline with visibility into data quality

Usage:
    from core.feature_diagnostics import FeatureStatus, diagnose_features
    
    status = diagnose_features(symbol, price_data, volume_data, context_data)
    logger.info("feature_status", extra={"features_status": status.to_dict()})
    
    if not status.is_usable():
        # Force confidence = 0 due to degraded features
        confidence = 0.0
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
import time

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureStatus:
    """Diagnostic struct for feature extraction pipeline"""
    
    # Identification
    symbol: str
    
    # Component status flags
    price_ok: bool = False
    volume_ok: bool = False
    momentum_ok: bool = False
    context_ok: bool = False
    sentiment_ok: bool = False
    
    # Metadata
    price_source: str = "unknown"
    price_age_seconds: float = 0.0
    num_features: int = 0
    missing_components: list[str] = field(default_factory=list)
    
    # Overall health
    degraded_features: bool = False
    
    def is_usable(self) -> bool:
        """
        Check if feature set is usable for prediction
        
        Minimum requirements:
        - price_ok must be True (critical)
        - At least 2 other components OK
        - num_features >= 3
        
        Returns:
            True if feature set meets minimum quality threshold
        """
        if not self.price_ok:
            return False
        
        components_ok = sum([
            self.volume_ok,
            self.momentum_ok,
            self.context_ok,
            self.sentiment_ok
        ])
        
        return components_ok >= 2 and self.num_features >= 3
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for logging"""
        return {
            "symbol": self.symbol,
            "price_source": self.price_source,
            "price_ok": self.price_ok,
            "volume_ok": self.volume_ok,
            "momentum_ok": self.momentum_ok,
            "context_ok": self.context_ok,
            "sentiment_ok": self.sentiment_ok,
            "num_features": self.num_features,
            "degraded_features": self.degraded_features,
            "missing_components": self.missing_components,
        }


def diagnose_features(
    symbol: str,
    price_data: Optional[Dict[str, Any]],
    volume_data: Optional[Dict[str, Any]] = None,
    momentum_data: Optional[Dict[str, Any]] = None,
    context_data: Optional[Dict[str, Any]] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    price_staleness_threshold_s: float = 300.0  # 5 minutes
) -> FeatureStatus:
    """
    Diagnose feature extraction pipeline status
    
    Args:
        symbol: Trading symbol
        price_data: {price: float, timestamp: float, provider: str}
        volume_data: {volume: float, avg_volume: float}
        momentum_data: {momentum_score: float, trend: str}
        context_data: {market_regime: str, sector_health: float}
        sentiment_data: {sentiment_score: float, news_count: int}
        price_staleness_threshold_s: Maximum age of price data (seconds)
        
    Returns:
        FeatureStatus with diagnostic flags
    """
    import time
    
    status = FeatureStatus(symbol=symbol)
    missing_components = []
    
    # Check price data
    if price_data and price_data.get("price") is not None:
        price = price_data.get("price")
        timestamp = price_data.get("timestamp", time.time())
        provider = price_data.get("provider", "unknown")
        
        age_s = time.time() - timestamp
        status.price_source = provider
        status.price_age_seconds = age_s
        
        # Price is OK if:
        # 1. Price is a valid number > 0
        # 2. Data is fresh (within staleness threshold)
        if price > 0 and age_s <= price_staleness_threshold_s:
            status.price_ok = True
            status.num_features += 1
        else:
            missing_components.append(f"price_stale ({age_s:.0f}s old)")
    else:
        missing_components.append("price_missing")
    
    # Check volume data
    if volume_data and volume_data.get("volume") is not None:
        volume = volume_data.get("volume")
        avg_volume = volume_data.get("avg_volume")
        
        if volume > 0:
            status.volume_ok = True
            status.num_features += 1
    else:
        missing_components.append("volume")
    
    # Check momentum data
    if momentum_data and momentum_data.get("momentum_score") is not None:
        status.momentum_ok = True
        status.num_features += 1
    else:
        missing_components.append("momentum")
    
    # Check context data
    if context_data and context_data.get("market_regime"):
        status.context_ok = True
        status.num_features += 1
    else:
        missing_components.append("context")
    
    # Check sentiment data
    if sentiment_data and sentiment_data.get("sentiment_score") is not None:
        status.sentiment_ok = True
        status.num_features += 1
    else:
        missing_components.append("sentiment")
    
    # Set degraded flag
    status.missing_components = missing_components
    status.degraded_features = not status.is_usable()
    
    return status


def build_confidence_with_diagnostics(
    base_confidence: float,
    feature_status: FeatureStatus,
    min_confidence_when_degraded: float = 0.0
) -> tuple[float, dict[str, Any]]:
    """
    Adjust confidence based on feature status
    
    Args:
        base_confidence: Raw model confidence (0.0-1.0)
        feature_status: Feature diagnostic status
        min_confidence_when_degraded: Confidence floor for degraded features
        
    Returns:
        (adjusted_confidence, metadata)
        
    Examples:
        >>> status = FeatureStatus(symbol="WOLF", price_ok=False, num_features=1)
        >>> conf, meta = build_confidence_with_diagnostics(0.75, status)
        >>> conf
        0.0
        >>> meta["degraded_features"]
        True
    """
    adjusted_confidence = base_confidence
    metadata = {
        "base_confidence": base_confidence,
        "degraded_features": feature_status.degraded_features,
        "num_features": feature_status.num_features,
        "missing_components": feature_status.missing_components,
    }
    
    # Force confidence to 0 if features are degraded
    if feature_status.degraded_features:
        adjusted_confidence = min_confidence_when_degraded
        metadata["confidence_adjustment"] = "forced_to_0_degraded_features"
        
        LOGGER.warning(
            f"[{feature_status.symbol}] Confidence degraded to {adjusted_confidence:.0%} "
            f"due to missing: {', '.join(feature_status.missing_components)}"
        )
    
    # Reduce confidence if some features missing (but not fully degraded)
    elif feature_status.num_features < 5:
        penalty = (5 - feature_status.num_features) * 0.10  # 10% penalty per missing feature
        adjusted_confidence = max(0.0, base_confidence - penalty)
        metadata["confidence_adjustment"] = f"reduced_by_{penalty:.0%}_missing_features"
    
    return adjusted_confidence, metadata


# Example usage
if __name__ == "__main__":
    import json
    
    # Test case 1: All features healthy
    price_data = {"price": 17.51, "timestamp": 1700000000.0, "provider": "polygon"}
    volume_data = {"volume": 1_000_000, "avg_volume": 500_000}
    momentum_data = {"momentum_score": 0.65, "trend": "up"}
    context_data = {"market_regime": "bull", "sector_health": 0.72}
    sentiment_data = {"sentiment_score": 0.55, "news_count": 12}
    
    status = diagnose_features("WOLF", price_data, volume_data, momentum_data, context_data, sentiment_data)
    print("Test 1: All features healthy")
    print(json.dumps(status.to_dict(), indent=2))
    print(f"Usable: {status.is_usable()}")
    print()
    
    # Test case 2: Price missing (critical failure)
    status = diagnose_features("WOLF", None, volume_data, momentum_data, context_data, sentiment_data)
    print("Test 2: Price missing (critical)")
    print(json.dumps(status.to_dict(), indent=2))
    print(f"Usable: {status.is_usable()}")
    print()
    
    # Test case 3: Price OK but minimal features
    status = diagnose_features("WOLF", price_data, None, None, None, None)
    print("Test 3: Price OK, minimal features")
    print(json.dumps(status.to_dict(), indent=2))
    print(f"Usable: {status.is_usable()}")
    print()
    
    # Test case 4: Confidence adjustment
    feature_status = FeatureStatus(symbol="WOLF", price_ok=False, num_features=1, degraded_features=True)
    conf, meta = build_confidence_with_diagnostics(0.75, feature_status)
    print("Test 4: Confidence adjustment with degraded features")
    print(f"Base: 0.75 → Adjusted: {conf}")
    print(json.dumps(meta, indent=2))
