"""
VWAP Signal Generator for Ghost Protocol
Volume Weighted Average Price analysis
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VWAPAnalyzer:
    """Calculate and analyze VWAP signals"""
    
    def __init__(self):
        self.enabled = os.getenv("VWAP_ENABLED", "1") == "1"
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes
    
    def calculate_vwap(self, symbol: str, period_days: int = 1) -> Optional[Dict]:
        """
        Calculate VWAP for a symbol
        
        Args:
            symbol: Trading symbol
            period_days: Number of days for VWAP calculation
            
        Returns:
            Dict with vwap, current_price, position, signal
        """
        import time
        
        cache_key = f"{symbol}_{period_days}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached.get("_ts", 0) < self._cache_ttl:
                return cached
        
        try:
            import yfinance as yf
            import numpy as np
            
            # Handle crypto symbols
            ticker_symbol = symbol if "-" in symbol or len(symbol) > 4 else f"{symbol}-USD"
            
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=f"{period_days}d", interval="1h")
            
            if df.empty or len(df) < 5:
                return None
            
            # Calculate VWAP
            # VWAP = Σ(Price × Volume) / Σ(Volume)
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            
            # Handle zero volume
            total_volume = df["Volume"].sum()
            if total_volume == 0:
                total_volume = 1  # Prevent division by zero
            
            vwap = (typical_price * df["Volume"]).sum() / total_volume
            
            # Current price
            current_price = df["Close"].iloc[-1]
            
            # Calculate standard deviation bands
            squared_diff = ((typical_price - vwap) ** 2 * df["Volume"]).sum() / total_volume
            std_dev = np.sqrt(squared_diff) if squared_diff > 0 else current_price * 0.02
            
            upper_band_1 = vwap + std_dev
            lower_band_1 = vwap - std_dev
            upper_band_2 = vwap + (2 * std_dev)
            lower_band_2 = vwap - (2 * std_dev)
            
            # Determine position relative to VWAP
            deviation_pct = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0
            
            if current_price > upper_band_2:
                position = "FAR_ABOVE"
                signal = "OVERBOUGHT"
                direction = "DOWN"
                strength = 0.8
            elif current_price > upper_band_1:
                position = "ABOVE"
                signal = "BULLISH"
                direction = "UP"
                strength = 0.6
            elif current_price > vwap:
                position = "SLIGHTLY_ABOVE"
                signal = "NEUTRAL_BULLISH"
                direction = "UP"
                strength = 0.4
            elif current_price > lower_band_1:
                position = "SLIGHTLY_BELOW"
                signal = "NEUTRAL_BEARISH"
                direction = "DOWN"
                strength = 0.4
            elif current_price > lower_band_2:
                position = "BELOW"
                signal = "BEARISH"
                direction = "DOWN"
                strength = 0.6
            else:
                position = "FAR_BELOW"
                signal = "OVERSOLD"
                direction = "UP"
                strength = 0.8
            
            result = {
                "symbol": symbol,
                "vwap": round(float(vwap), 6),
                "current_price": round(float(current_price), 6),
                "deviation_pct": round(float(deviation_pct), 2),
                "position": position,
                "signal": signal,
                "direction": direction,
                "strength": strength,
                "bands": {
                    "upper_2": round(float(upper_band_2), 6),
                    "upper_1": round(float(upper_band_1), 6),
                    "vwap": round(float(vwap), 6),
                    "lower_1": round(float(lower_band_1), 6),
                    "lower_2": round(float(lower_band_2), 6)
                },
                "period_hours": period_days * 24,
                "_ts": time.time()
            }
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"VWAP calculation failed for {symbol}: {e}")
            return None
    
    def get_vwap_signal(self, symbol: str) -> Dict:
        """
        Get VWAP-based trading signal
        
        Returns signal suitable for ensemble predictor
        """
        if not self.enabled:
            return {"enabled": False, "symbol": symbol}
        
        # Calculate daily and weekly VWAP
        daily = self.calculate_vwap(symbol, period_days=1)
        weekly = self.calculate_vwap(symbol, period_days=7)
        
        if not daily:
            return {
                "symbol": symbol,
                "available": False,
                "error": "Could not calculate VWAP"
            }
        
        # Combine signals
        signals = [daily]
        if weekly:
            signals.append(weekly)
        
        # Average direction strength
        up_strength = sum(s["strength"] for s in signals if s["direction"] == "UP")
        down_strength = sum(s["strength"] for s in signals if s["direction"] == "DOWN")
        
        if up_strength > down_strength:
            final_direction = "UP"
            final_strength = up_strength / len(signals)
        elif down_strength > up_strength:
            final_direction = "DOWN"
            final_strength = down_strength / len(signals)
        else:
            final_direction = "NEUTRAL"
            final_strength = 0.3
        
        return {
            "symbol": symbol,
            "available": True,
            "direction": final_direction,
            "confidence": round(0.5 + (final_strength * 0.3), 2),  # 0.5 to 0.8 range
            "daily_vwap": daily,
            "weekly_vwap": weekly
        }


# Singleton
_vwap: Optional[VWAPAnalyzer] = None


def get_vwap_analyzer() -> VWAPAnalyzer:
    """Get or create VWAPAnalyzer singleton"""
    global _vwap
    if _vwap is None:
        _vwap = VWAPAnalyzer()
    return _vwap


def get_vwap_signal(symbol: str) -> Dict:
    """Get VWAP signal for a symbol"""
    return get_vwap_analyzer().get_vwap_signal(symbol)


def calculate_vwap(symbol: str, period_days: int = 1) -> Optional[Dict]:
    """Calculate VWAP for a symbol"""
    return get_vwap_analyzer().calculate_vwap(symbol, period_days)
