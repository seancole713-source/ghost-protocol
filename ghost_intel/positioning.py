"""
GHOST INTEL - POSITIONING ANALYZER
===================================
Options, gamma exposure, and market positioning analysis.

Layer 8: POSITIONING
- Put/Call Ratio
- VIX Term Structure
- Volume Anomalies
- Crypto Liquidations (proxy for risk cascade)

Author: Ghost AI
Date: 2026-01-26
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ghost.intel")


@dataclass
class PositioningData:
    """Current market positioning snapshot"""
    timestamp: float
    
    # Put/Call
    put_call_ratio: float
    pcr_percentile: float          # Where is current P/C vs history
    
    # VIX
    vix_level: float
    vix_change: float
    vix_term_structure: str        # "contango" or "backwardation"
    vix_percentile: float
    
    # Interpretation
    fear_level: str                # EXTREME_FEAR, FEAR, NEUTRAL, GREED, EXTREME_GREED
    positioning: str               # DEFENSIVE, NEUTRAL, AGGRESSIVE
    fragility: float               # 0-100, how fragile is current positioning
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "put_call_ratio": round(self.put_call_ratio, 2),
            "pcr_percentile": round(self.pcr_percentile, 1),
            "vix_level": round(self.vix_level, 2),
            "vix_change": round(self.vix_change, 2),
            "vix_term_structure": self.vix_term_structure,
            "vix_percentile": round(self.vix_percentile, 1),
            "fear_level": self.fear_level,
            "positioning": self.positioning,
            "fragility": round(self.fragility, 1),
            "warnings": self.warnings,
        }


@dataclass
class GammaExposure:
    """
    Gamma exposure analysis.
    
    When market makers are short gamma (negative GEX):
    - They have to buy highs and sell lows
    - This AMPLIFIES moves
    
    When market makers are long gamma (positive GEX):
    - They have to sell highs and buy lows
    - This DAMPENS moves
    """
    timestamp: float
    estimated_gex: float           # Positive = long gamma, negative = short gamma
    gex_flip_level: Optional[float] = None  # Price where GEX flips sign
    amplification_risk: str = "UNKNOWN"  # HIGH, MEDIUM, LOW
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "estimated_gex": self.estimated_gex,
            "gex_flip_level": self.gex_flip_level,
            "amplification_risk": self.amplification_risk,
        }


class PositioningAnalyzer:
    """
    Analyze market positioning for impact assessment.
    
    Positioning fragility tells us how much the market might move
    when hit with an event - not the direction, but the magnitude.
    """
    
    def __init__(self):
        # Historical data for percentile calculations
        self._vix_history: List[float] = []
        self._pcr_history: List[float] = []
        
        # Load historical ranges (approximations)
        # VIX: 10-80 range historically, median ~17
        # P/C: 0.5-1.5 range, median ~0.85
        self._vix_median = 17.0
        self._vix_10th = 12.0
        self._vix_90th = 28.0
        
        self._pcr_median = 0.85
        self._pcr_10th = 0.65
        self._pcr_90th = 1.15
    
    def analyze(self, data: Dict[str, Any]) -> PositioningData:
        """
        Analyze current positioning from market data.
        
        Args:
            data: Dictionary with vix, put_call_ratio, vix_term_structure
        
        Returns:
            PositioningData analysis
        """
        timestamp = time.time()
        warnings = []
        
        # Extract data - handle both dict and direct value formats
        vix_raw = data.get("vix_level") or data.get("vix", 15)
        if isinstance(vix_raw, dict):
            vix = vix_raw.get("price", 15)
        else:
            vix = float(vix_raw) if vix_raw else 15.0
        
        pcr = data.get("put_call_ratio", 0.9)
        vix_change = data.get("vix_change", 0)
        
        # VIX term structure
        term_structure = "contango"  # Normal
        if "vix_term_structure" in data:
            ts_data = data["vix_term_structure"]
            if isinstance(ts_data, dict) and ts_data.get("backwardation"):
                term_structure = "backwardation"
        
        # Calculate percentiles
        vix_percentile = self._calculate_percentile(
            vix, self._vix_10th, self._vix_median, self._vix_90th
        )
        pcr_percentile = self._calculate_percentile(
            pcr, self._pcr_10th, self._pcr_median, self._pcr_90th
        )
        
        # Determine fear level
        fear_level = self._determine_fear_level(vix, pcr)
        
        # Determine positioning stance
        positioning = self._determine_positioning(pcr, vix)
        
        # Calculate fragility
        fragility = self._calculate_fragility(vix, pcr, term_structure)
        
        # Generate warnings
        if vix > 30:
            warnings.append("VIX > 30: Extreme fear, expect large moves")
        if vix < 12:
            warnings.append("VIX < 12: Complacency, watch for vol spike")
        if pcr > 1.2:
            warnings.append("P/C > 1.2: Heavy put buying, market defensive")
        if pcr < 0.6:
            warnings.append("P/C < 0.6: Heavy call buying, potential reversal")
        if term_structure == "backwardation":
            warnings.append("VIX backwardation: Near-term fear elevated")
        if vix_change > 20:
            warnings.append(f"VIX spike +{vix_change:.0f}%: Risk-off in progress")
        
        return PositioningData(
            timestamp=timestamp,
            put_call_ratio=pcr,
            pcr_percentile=pcr_percentile,
            vix_level=vix,
            vix_change=vix_change,
            vix_term_structure=term_structure,
            vix_percentile=vix_percentile,
            fear_level=fear_level,
            positioning=positioning,
            fragility=fragility,
            warnings=warnings,
        )
    
    def estimate_gamma_exposure(self, spy_price: float, vix: float) -> GammaExposure:
        """
        Estimate market-wide gamma exposure.
        
        This is a simplified model. Real GEX requires options chain data.
        We use VIX and SPY price as proxies.
        
        Heuristics:
        - Low VIX + high SPY = dealers likely long gamma (dampening)
        - High VIX = dealers likely short gamma (amplifying)
        - Near round numbers = higher gamma (more options at strikes)
        """
        timestamp = time.time()
        
        # VIX-based estimation
        # High VIX = dealers short gamma (they sold puts)
        if vix > 25:
            estimated_gex = -50  # Negative = short gamma
            amplification_risk = "HIGH"
        elif vix > 20:
            estimated_gex = -20
            amplification_risk = "MEDIUM"
        elif vix > 15:
            estimated_gex = 10  # Slight positive
            amplification_risk = "LOW"
        else:
            estimated_gex = 30  # Long gamma
            amplification_risk = "LOW"
        
        # GEX flip level estimation (where gamma changes sign)
        # Typically near ATM strikes - estimate around current price
        gex_flip_level = round(spy_price / 5) * 5  # Nearest $5 strike
        
        return GammaExposure(
            timestamp=timestamp,
            estimated_gex=estimated_gex,
            gex_flip_level=gex_flip_level,
            amplification_risk=amplification_risk,
        )
    
    def _calculate_percentile(
        self, value: float, p10: float, p50: float, p90: float
    ) -> float:
        """Calculate approximate percentile"""
        if value <= p10:
            return 10 * (value / p10) if p10 > 0 else 0
        elif value <= p50:
            return 10 + 40 * ((value - p10) / (p50 - p10))
        elif value <= p90:
            return 50 + 40 * ((value - p50) / (p90 - p50))
        else:
            return min(100, 90 + 10 * ((value - p90) / p90))
    
    def _determine_fear_level(self, vix: float, pcr: float) -> str:
        """Determine fear level from VIX and P/C"""
        # Combined score
        vix_score = vix / 20  # 1.0 at VIX 20
        pcr_score = pcr / 0.9  # 1.0 at P/C 0.9
        
        combined = (vix_score + pcr_score) / 2
        
        if combined > 1.5:
            return "EXTREME_FEAR"
        elif combined > 1.2:
            return "FEAR"
        elif combined > 0.8:
            return "NEUTRAL"
        elif combined > 0.5:
            return "GREED"
        else:
            return "EXTREME_GREED"
    
    def _determine_positioning(self, pcr: float, vix: float) -> str:
        """Determine market positioning stance"""
        if pcr > 1.1 or vix > 25:
            return "DEFENSIVE"
        elif pcr < 0.7 and vix < 15:
            return "AGGRESSIVE"
        else:
            return "NEUTRAL"
    
    def _calculate_fragility(
        self, vix: float, pcr: float, term_structure: str
    ) -> float:
        """
        Calculate how fragile current positioning is (0-100).
        Higher = more likely to see big moves.
        """
        fragility = 0.0
        
        # VIX contribution (0-40)
        if vix > 30:
            fragility += 40
        elif vix > 25:
            fragility += 30
        elif vix > 20:
            fragility += 20
        elif vix > 15:
            fragility += 10
        else:
            fragility += 5
        
        # P/C contribution (0-30)
        if pcr > 1.3 or pcr < 0.5:
            fragility += 30  # Extreme positioning
        elif pcr > 1.1 or pcr < 0.6:
            fragility += 20
        elif pcr > 1.0 or pcr < 0.7:
            fragility += 10
        
        # Term structure contribution (0-20)
        if term_structure == "backwardation":
            fragility += 20  # Near-term fear
        
        # Complacency factor (0-10)
        if vix < 12:
            fragility += 10  # Too calm, vol expansion likely
        
        return min(100, fragility)
    
    def get_positioning_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get actionable positioning signal for trading.
        
        Returns:
            {
                "bias": "DEFENSIVE" | "NEUTRAL" | "AGGRESSIVE",
                "confidence_multiplier": 0.5-1.5,
                "position_size_adj": 0.5-1.0,
                "stop_tightness": "TIGHT" | "NORMAL" | "WIDE",
                "rationale": str
            }
        """
        analysis = self.analyze(data)
        
        # Base multipliers
        confidence_mult = 1.0
        size_adj = 1.0
        stop_tightness = "NORMAL"
        rationale_parts = []
        
        # Adjust based on fear level
        if analysis.fear_level == "EXTREME_FEAR":
            confidence_mult *= 0.7
            size_adj *= 0.5
            stop_tightness = "TIGHT"
            rationale_parts.append("Extreme fear: reduce size, tighten stops")
        elif analysis.fear_level == "FEAR":
            confidence_mult *= 0.85
            size_adj *= 0.75
            rationale_parts.append("Elevated fear: caution warranted")
        elif analysis.fear_level == "EXTREME_GREED":
            confidence_mult *= 0.8
            rationale_parts.append("Extreme greed: potential reversal risk")
        
        # Adjust based on fragility
        if analysis.fragility > 70:
            size_adj *= 0.6
            stop_tightness = "TIGHT"
            rationale_parts.append(f"High fragility ({analysis.fragility:.0f}): expect large moves")
        elif analysis.fragility > 50:
            size_adj *= 0.8
            rationale_parts.append(f"Elevated fragility ({analysis.fragility:.0f})")
        
        # VIX term structure
        if analysis.vix_term_structure == "backwardation":
            confidence_mult *= 0.9
            rationale_parts.append("VIX backwardation: near-term risk elevated")
        
        return {
            "bias": analysis.positioning,
            "confidence_multiplier": round(confidence_mult, 2),
            "position_size_adj": round(size_adj, 2),
            "stop_tightness": stop_tightness,
            "rationale": "; ".join(rationale_parts) if rationale_parts else "Normal conditions",
            "analysis": analysis.to_dict(),
        }


# Singleton instance
_positioning_analyzer: Optional[PositioningAnalyzer] = None


def get_positioning_analyzer() -> PositioningAnalyzer:
    """Get singleton analyzer instance"""
    global _positioning_analyzer
    if _positioning_analyzer is None:
        _positioning_analyzer = PositioningAnalyzer()
    return _positioning_analyzer
