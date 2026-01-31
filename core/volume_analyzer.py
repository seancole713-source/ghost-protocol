"""
GHOST MAXIMUM v2.0 - Volume Analysis System
Detects accumulation/distribution patterns
"""
import numpy as np
from typing import Dict, Any, List
from collections import deque
import logging

LOGGER = logging.getLogger(__name__)


class VolumeAnalyzer:
    """
    Analyze volume patterns to detect:
    - Accumulation (smart money buying)
    - Distribution (smart money selling)
    - Volume climax (exhaustion)
    - Volume divergence (price vs volume)
    """
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self.volume_history: Dict[str, deque] = {}
        self.price_history: Dict[str, deque] = {}
    
    async def analyze_volume(
        self,
        symbol: str,
        current_price: float,
        current_volume: int,
        price_change: float
    ) -> Dict[str, Any]:
        """
        Analyze volume patterns
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            current_volume: Current volume
            price_change: Price change % (positive = up, negative = down)
        
        Returns:
            {
                "pattern": "accumulation/distribution/neutral/climax",
                "strength": 0.0-1.0,
                "obv_trend": "up/down/flat",
                "volume_trend": "increasing/decreasing/stable",
                "price_volume_divergence": bool,
                "signal": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0
            }
        """
        # Initialize history if needed
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.lookback_periods)
            self.price_history[symbol] = deque(maxlen=self.lookback_periods)
        
        # Add current data
        self.volume_history[symbol].append(current_volume)
        self.price_history[symbol].append(current_price)
        
        # Need at least 10 periods for analysis
        if len(self.volume_history[symbol]) < 10:
            return self._neutral_result()
        
        # Calculate metrics
        obv_trend = self._calculate_obv_trend(symbol, price_change)
        volume_trend = self._calculate_volume_trend(symbol)
        avg_volume = np.mean(self.volume_history[symbol])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Detect patterns
        pattern, strength = self._detect_pattern(
            symbol, price_change, current_volume, avg_volume, obv_trend
        )
        
        # Check for divergence
        divergence = self._detect_divergence(symbol)
        
        # Generate signal
        signal, confidence = self._generate_signal(
            pattern, strength, obv_trend, volume_trend, divergence
        )
        
        return {
            "pattern": pattern,
            "strength": strength,
            "obv_trend": obv_trend,
            "volume_trend": volume_trend,
            "volume_ratio": volume_ratio,
            "price_volume_divergence": divergence,
            "signal": signal,
            "confidence": confidence,
            "avg_volume": int(avg_volume)
        }
    
    def _calculate_obv_trend(self, symbol: str, price_change: float) -> str:
        """
        On-Balance Volume trend
        Tracks cumulative volume based on price direction
        """
        obv_values = []
        obv = 0
        
        volumes = list(self.volume_history[symbol])
        prices = list(self.price_history[symbol])
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
            obv_values.append(obv)
        
        if len(obv_values) < 5:
            return "flat"
        
        # Simple trend: compare recent OBV to older OBV
        recent_obv = np.mean(obv_values[-5:])
        older_obv = np.mean(obv_values[-10:-5]) if len(obv_values) >= 10 else obv_values[0]
        
        if recent_obv > older_obv * 1.1:
            return "up"
        elif recent_obv < older_obv * 0.9:
            return "down"
        else:
            return "flat"
    
    def _calculate_volume_trend(self, symbol: str) -> str:
        """Calculate if volume is increasing or decreasing"""
        volumes = list(self.volume_history[symbol])
        
        if len(volumes) < 5:
            return "stable"
        
        recent_avg = np.mean(volumes[-5:])
        older_avg = np.mean(volumes[-10:-5]) if len(volumes) >= 10 else volumes[0]
        
        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_pattern(
        self,
        symbol: str,
        price_change: float,
        current_volume: int,
        avg_volume: float,
        obv_trend: str
    ) -> tuple[str, float]:
        """
        Detect accumulation/distribution patterns
        
        Returns: (pattern_name, strength)
        """
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # ACCUMULATION: Price flat/down + volume up + OBV up = Smart money buying
        if abs(price_change) < 1.5 and volume_ratio > 1.3 and obv_trend == "up":
            strength = min(0.9, 0.6 + (volume_ratio - 1.3) * 0.5)
            return "accumulation", strength
        
        # DISTRIBUTION: Price flat/up + volume up + OBV down = Smart money selling
        if abs(price_change) < 1.5 and volume_ratio > 1.3 and obv_trend == "down":
            strength = min(0.9, 0.6 + (volume_ratio - 1.3) * 0.5)
            return "distribution", strength
        
        # CLIMAX: Extreme volume (2x+ avg) = Exhaustion/reversal imminent
        if volume_ratio > 2.0:
            strength = min(0.85, 0.7 + (volume_ratio - 2.0) * 0.3)
            return "climax", strength
        
        # Moderate patterns
        if volume_ratio > 1.5 and obv_trend == "up":
            return "accumulation", 0.55
        
        if volume_ratio > 1.5 and obv_trend == "down":
            return "distribution", 0.55
        
        return "neutral", 0.5
    
    def _detect_divergence(self, symbol: str) -> bool:
        """
        Detect price-volume divergence
        Price making higher highs but volume declining = bearish
        Price making lower lows but volume declining = bullish
        """
        prices = list(self.price_history[symbol])
        volumes = list(self.volume_history[symbol])
        
        if len(prices) < 10:
            return False
        
        # Recent trend
        recent_price_trend = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        recent_volume_trend = (volumes[-1] - np.mean(volumes[-5:-1])) / np.mean(volumes[-5:-1])
        
        # Divergence: price up but volume down, or vice versa
        if abs(recent_price_trend) > 0.02 and abs(recent_volume_trend) > 0.1:
            # Price and volume moving opposite directions
            if (recent_price_trend > 0 and recent_volume_trend < -0.1) or \
               (recent_price_trend < 0 and recent_volume_trend < -0.1):
                return True
        
        return False
    
    def _generate_signal(
        self,
        pattern: str,
        strength: float,
        obv_trend: str,
        volume_trend: str,
        divergence: bool
    ) -> tuple[str, float]:
        """Generate trading signal from volume analysis"""
        
        # ACCUMULATION = BUY signal
        if pattern == "accumulation":
            confidence = 0.65 + strength * 0.25
            return "BUY", confidence
        
        # DISTRIBUTION = SELL signal
        if pattern == "distribution":
            confidence = 0.65 + strength * 0.25
            return "SELL", confidence
        
        # CLIMAX = REVERSAL signal (direction depends on current trend)
        if pattern == "climax":
            # If OBV up, climax likely marks top (SELL)
            if obv_trend == "up":
                return "SELL", 0.7 + strength * 0.2
            # If OBV down, climax likely marks bottom (BUY)
            elif obv_trend == "down":
                return "BUY", 0.7 + strength * 0.2
        
        # DIVERGENCE = Warning signal
        if divergence:
            if obv_trend == "down":
                return "SELL", 0.6
            elif obv_trend == "up":
                return "BUY", 0.6
        
        # OBV trend alone (weak signal)
        if obv_trend == "up" and volume_trend == "increasing":
            return "BUY", 0.55
        elif obv_trend == "down" and volume_trend == "increasing":
            return "SELL", 0.55
        
        return "HOLD", 0.5
    
    def _neutral_result(self) -> Dict[str, Any]:
        """Return neutral result when insufficient data"""
        return {
            "pattern": "neutral",
            "strength": 0.5,
            "obv_trend": "flat",
            "volume_trend": "stable",
            "volume_ratio": 1.0,
            "price_volume_divergence": False,
            "signal": "HOLD",
            "confidence": 0.5,
            "avg_volume": 0
        }


# Singleton
_VOLUME_ANALYZER = None


def get_volume_analyzer() -> VolumeAnalyzer:
    """Get singleton volume analyzer"""
    global _VOLUME_ANALYZER
    if _VOLUME_ANALYZER is None:
        _VOLUME_ANALYZER = VolumeAnalyzer()
    return _VOLUME_ANALYZER
