"""
Ghost Protocol - Volume Analyzer
Detect unusual volume patterns that precede price moves.

INSIGHT: Volume is the fuel for price moves.
         Unusual volume = Something is happening
         Low volume breakout = Likely fake
         High volume breakout = Likely real
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")


class VolumeAnalyzer:
    """
    Analyze volume patterns for trading signals.
    
    KEY PATTERNS:
    - Volume spike before price move = Smart money positioning
    - High volume at support = Accumulation
    - High volume at resistance = Distribution
    - Volume divergence = Warning sign
    """
    
    def __init__(self):
        self.cache = {}
    
    async def get_volume_data(self, symbol: str, days: int = 30) -> Dict:
        """Get volume data for analysis"""
        if POLYGON_API_KEY:
            return await self._polygon_volume(symbol, days)
        return {}
    
    async def _polygon_volume(self, symbol: str, days: int) -> Dict:
        """Fetch volume data from Polygon"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {"apiKey": POLYGON_API_KEY}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        
                        if results:
                            volumes = [r.get("v", 0) for r in results]
                            prices = [r.get("c", 0) for r in results]
                            
                            return {
                                "volumes": volumes,
                                "prices": prices,
                                "avg_volume": sum(volumes) / len(volumes) if volumes else 0,
                                "latest_volume": volumes[-1] if volumes else 0,
                                "latest_price": prices[-1] if prices else 0,
                                "data_points": len(volumes)
                            }
        except Exception as e:
            logger.error(f"Polygon volume error: {e}")
        
        return {}
    
    async def analyze_volume(self, symbol: str) -> Dict:
        """
        Analyze volume patterns for signals.
        
        SIGNALS:
        - Volume > 2x average = Unusual activity
        - Volume > 3x average = Major event
        - Rising price + Rising volume = Strong trend
        - Rising price + Falling volume = Weak trend
        """
        data = await self.get_volume_data(symbol, 30)
        
        if not data or not data.get("volumes"):
            return {
                "symbol": symbol,
                "has_data": False,
                "signal": "NEUTRAL",
                "signal_strength": 0,
                "confidence_adjustment": 0,
                "message": "No volume data available"
            }
        
        volumes = data.get("volumes", [])
        prices = data.get("prices", [])
        avg_volume = data.get("avg_volume", 0)
        latest_volume = data.get("latest_volume", 0)
        
        # Calculate volume ratio
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
        
        # Analyze recent trend
        recent_volumes = volumes[-5:] if len(volumes) >= 5 else volumes
        recent_prices = prices[-5:] if len(prices) >= 5 else prices
        
        volume_trend = "FLAT"
        if len(recent_volumes) >= 2:
            if recent_volumes[-1] > recent_volumes[0] * 1.2:
                volume_trend = "RISING"
            elif recent_volumes[-1] < recent_volumes[0] * 0.8:
                volume_trend = "FALLING"
        
        price_trend = "FLAT"
        if len(recent_prices) >= 2:
            if recent_prices[-1] > recent_prices[0] * 1.02:
                price_trend = "UP"
            elif recent_prices[-1] < recent_prices[0] * 0.98:
                price_trend = "DOWN"
        
        # Determine signal
        signal = "NEUTRAL"
        signal_strength = 0
        confidence_adjustment = 0
        warnings = []
        positives = []
        
        # VOLUME SPIKE
        if volume_ratio >= 3:
            signal_strength += 35
            warnings.append(f"🚨 EXTREME VOLUME: {volume_ratio:.1f}x average - Major event likely")
        elif volume_ratio >= 2:
            signal_strength += 20
            warnings.append(f"⚠️ HIGH VOLUME: {volume_ratio:.1f}x average - Something happening")
        elif volume_ratio >= 1.5:
            signal_strength += 10
            warnings.append(f"📊 ELEVATED VOLUME: {volume_ratio:.1f}x average")
        
        # TREND ANALYSIS
        if price_trend == "UP" and volume_trend == "RISING":
            signal = "BULLISH"
            confidence_adjustment += 10
            positives.append("✅ STRONG UPTREND: Rising price with rising volume")
        elif price_trend == "UP" and volume_trend == "FALLING":
            confidence_adjustment -= 5
            warnings.append("⚠️ WEAK UPTREND: Rising price but falling volume")
        elif price_trend == "DOWN" and volume_trend == "RISING":
            signal = "BEARISH"
            confidence_adjustment -= 10
            warnings.append("⚠️ STRONG DOWNTREND: Falling price with rising volume")
        elif price_trend == "DOWN" and volume_trend == "FALLING":
            confidence_adjustment += 3
            positives.append("📉 Selling pressure decreasing (volume falling)")
        
        return {
            "symbol": symbol,
            "has_data": True,
            "signal": signal,
            "signal_strength": min(signal_strength, 100),
            "confidence_adjustment": max(-15, min(15, confidence_adjustment)),
            
            # Volume metrics
            "metrics": {
                "latest_volume": latest_volume,
                "avg_volume_30d": avg_volume,
                "volume_ratio": round(volume_ratio, 2),
                "volume_trend": volume_trend,
                "price_trend": price_trend
            },
            
            # Signals
            "warnings": warnings,
            "positives": positives,
            
            # Interpretation
            "interpretation": self._generate_interpretation(
                volume_ratio, volume_trend, price_trend, warnings, positives
            )
        }
    
    def _generate_interpretation(self, vol_ratio: float, vol_trend: str,
                                  price_trend: str, warnings: List, 
                                  positives: List) -> str:
        """Generate interpretation"""
        parts = []
        
        parts.append(f"📊 VOLUME ANALYSIS")
        parts.append(f"Current vs Average: {vol_ratio:.1f}x")
        parts.append(f"Volume Trend: {vol_trend}")
        parts.append(f"Price Trend: {price_trend}")
        parts.append("")
        
        if warnings:
            parts.extend(warnings)
        if positives:
            parts.extend(positives)
        
        return "\n".join(parts)


# Singleton
_analyzer = None

def get_volume_analyzer() -> VolumeAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = VolumeAnalyzer()
    return _analyzer


async def analyze_volume(symbol: str) -> Dict:
    """Quick access to volume analysis"""
    return await get_volume_analyzer().analyze_volume(symbol)
