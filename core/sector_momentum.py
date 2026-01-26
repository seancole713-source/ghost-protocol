#!/usr/bin/env python3
"""
📊 SECTOR MOMENTUM - Don't fight sector rotation

If tech is rotating out, don't buy AAPL.
If energy is hot, consider XOM.

Sector ETFs track institutional money flow:
- XLK (Technology)
- XLF (Financials)
- XLE (Energy)
- XLV (Healthcare)
- XLY (Consumer Discretionary)
- XLP (Consumer Staples)
- XLI (Industrials)
- XLB (Materials)
- XLU (Utilities)
- XLRE (Real Estate)
- XLC (Communication Services)
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

LOGGER = logging.getLogger("ghost.sector_momentum")

# ============================================================================
# SECTOR MAPPING - Stock to Sector ETF
# ============================================================================

SECTOR_MAP: Dict[str, str] = {
    # Technology (XLK)
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", 
    "INTC": "XLK", "CRM": "XLK", "ADBE": "XLK", "ORCL": "XLK",
    "CSCO": "XLK", "IBM": "XLK", "AVGO": "XLK", "TXN": "XLK",
    "QCOM": "XLK", "NOW": "XLK", "INTU": "XLK", "MU": "XLK",
    
    # Communication Services (XLC) - Tech-adjacent
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC",
    "DIS": "XLC", "CMCSA": "XLC", "VZ": "XLC", "T": "XLC",
    "TMUS": "XLC", "CHTR": "XLC",
    
    # Consumer Discretionary (XLY)
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "NKE": "XLY",
    "MCD": "XLY", "SBUX": "XLY", "LOW": "XLY", "TGT": "XLY",
    "BKNG": "XLY", "MAR": "XLY", "GM": "XLY", "F": "XLY",
    "ABNB": "XLY", "EBAY": "XLY", "ETSY": "XLY",
    
    # Financials (XLF)
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF",
    "MS": "XLF", "C": "XLF", "BLK": "XLF", "SCHW": "XLF",
    "AXP": "XLF", "USB": "XLF", "PNC": "XLF", "COF": "XLF",
    "COIN": "XLF", "HOOD": "XLF", "SOFI": "XLF",
    
    # Healthcare (XLV)
    "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "ABBV": "XLV",
    "MRK": "XLV", "LLY": "XLV", "TMO": "XLV", "ABT": "XLV",
    "DHR": "XLV", "BMY": "XLV", "AMGN": "XLV", "GILD": "XLV",
    "MRNA": "XLV", "BIIB": "XLV", "REGN": "XLV",
    
    # Energy (XLE)
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE",
    "EOG": "XLE", "MPC": "XLE", "PSX": "XLE", "VLO": "XLE",
    "OXY": "XLE", "HAL": "XLE", "DVN": "XLE", "FANG": "XLE",
    
    # Industrials (XLI)
    "CAT": "XLI", "BA": "XLI", "HON": "XLI", "UPS": "XLI",
    "RTX": "XLI", "LMT": "XLI", "GE": "XLI", "DE": "XLI",
    "MMM": "XLI", "FDX": "XLI", "UNP": "XLI", "WM": "XLI",
    
    # Consumer Staples (XLP)
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP",
    "WMT": "XLP", "PM": "XLP", "MO": "XLP", "CL": "XLP",
    "MDLZ": "XLP", "KHC": "XLP", "GIS": "XLP", "K": "XLP",
    
    # Utilities (XLU)
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU",
    "AEP": "XLU", "SRE": "XLU", "EXC": "XLU", "XEL": "XLU",
    
    # Materials (XLB)
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "FCX": "XLB",
    "NEM": "XLB", "NUE": "XLB", "DOW": "XLB", "DD": "XLB",
    
    # Real Estate (XLRE)
    "AMT": "XLRE", "PLD": "XLRE", "CCI": "XLRE", "EQIX": "XLRE",
    "PSA": "XLRE", "DLR": "XLRE", "O": "XLRE", "WELL": "XLRE",
    
    # Special cases
    "WOLF": "XLK",  # Wolfspeed - Semiconductors
    "PLTR": "XLK",  # Palantir - Tech
    "PACS": "XLV",  # PACS Group - Healthcare IT
}

# Sector ETF to full name mapping
SECTOR_NAMES = {
    "XLK": "Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}

# Cache for sector trends
_SECTOR_TREND_CACHE: Dict[str, Tuple[float, float]] = {}  # {sector: (trend_pct, timestamp)}
_SECTOR_CACHE_TTL = 300  # 5 minutes


@dataclass
class SectorMomentum:
    """Sector momentum analysis result"""
    sector_etf: str
    sector_name: str
    trend_5d: float  # 5-day trend percentage
    trend_20d: float  # 20-day trend percentage
    relative_strength: float  # vs SPY
    is_rotating_in: bool  # Money flowing INTO sector
    is_rotating_out: bool  # Money flowing OUT of sector
    signal: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float


def get_sector_for_symbol(symbol: str) -> Optional[str]:
    """Get sector ETF for a stock symbol"""
    return SECTOR_MAP.get(symbol.upper())


def get_sector_name(sector_etf: str) -> str:
    """Get full name for sector ETF"""
    return SECTOR_NAMES.get(sector_etf, sector_etf)


def get_sector_trend(sector_etf: str, days: int = 5) -> Optional[float]:
    """
    Get sector trend (percent change over N days).
    
    Returns:
        Percentage change (e.g., 2.5 for +2.5%, -1.3 for -1.3%)
        None if unable to calculate
    """
    # Check cache
    cache_key = f"{sector_etf}_{days}"
    if cache_key in _SECTOR_TREND_CACHE:
        cached_trend, cached_ts = _SECTOR_TREND_CACHE[cache_key]
        if time.time() - cached_ts < _SECTOR_CACHE_TTL:
            return cached_trend
    
    try:
        # Try to get price history from existing price infrastructure
        from wolf_app import _get_price_history_cached
        
        history = _get_price_history_cached(sector_etf, days=days + 5)
        
        if history and len(history) >= 2:
            prices = [h.get("price") or h.get("close") for h in history if h.get("price") or h.get("close")]
            if len(prices) >= 2:
                # Calculate percent change
                old_price = prices[0]
                new_price = prices[-1]
                if old_price > 0:
                    trend_pct = ((new_price - old_price) / old_price) * 100
                    
                    # Cache result
                    _SECTOR_TREND_CACHE[cache_key] = (trend_pct, time.time())
                    return trend_pct
    except Exception as e:
        LOGGER.debug(f"Sector trend calculation failed for {sector_etf}: {e}")
    
    # Fallback: try yfinance directly
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(sector_etf)
        hist = ticker.history(period=f"{days + 5}d")
        
        if len(hist) >= 2:
            old_price = hist['Close'].iloc[0]
            new_price = hist['Close'].iloc[-1]
            
            if old_price > 0:
                trend_pct = ((new_price - old_price) / old_price) * 100
                _SECTOR_TREND_CACHE[cache_key] = (trend_pct, time.time())
                return trend_pct
    except Exception as e:
        LOGGER.warning(f"yfinance fallback failed for {sector_etf}: {e}")
    
    return None


def get_spy_trend(days: int = 20) -> Optional[float]:
    """Get SPY trend for relative strength calculation"""
    return get_sector_trend("SPY", days)


def analyze_sector_momentum(symbol: str) -> Optional[SectorMomentum]:
    """
    Analyze sector momentum for a stock.
    
    Returns:
        SectorMomentum object with full analysis
        None if sector unknown or data unavailable
    """
    sector_etf = get_sector_for_symbol(symbol)
    
    if not sector_etf:
        LOGGER.debug(f"No sector mapping for {symbol}")
        return None
    
    # Get sector trends
    trend_5d = get_sector_trend(sector_etf, days=5)
    trend_20d = get_sector_trend(sector_etf, days=20)
    
    if trend_5d is None or trend_20d is None:
        LOGGER.warning(f"Unable to get sector trends for {sector_etf}")
        return None
    
    # Get SPY for relative strength
    spy_trend = get_spy_trend(days=20) or 0
    
    # Calculate relative strength (sector vs market)
    relative_strength = trend_20d - spy_trend
    
    # Determine rotation status
    # Rotating IN: Short-term momentum exceeds long-term (acceleration)
    # Rotating OUT: Short-term momentum below long-term (deceleration)
    is_rotating_in = trend_5d > trend_20d and trend_5d > 0
    is_rotating_out = trend_5d < trend_20d and trend_5d < 0
    
    # Generate signal
    if trend_5d > 1.0 and trend_20d > 0 and relative_strength > 0:
        signal = "BULLISH"
        confidence = min(0.9, 0.5 + (trend_5d / 10))
    elif trend_5d < -1.0 and trend_20d < 0 and relative_strength < 0:
        signal = "BEARISH"
        confidence = min(0.9, 0.5 + (abs(trend_5d) / 10))
    else:
        signal = "NEUTRAL"
        confidence = 0.5
    
    return SectorMomentum(
        sector_etf=sector_etf,
        sector_name=get_sector_name(sector_etf),
        trend_5d=round(trend_5d, 2),
        trend_20d=round(trend_20d, 2),
        relative_strength=round(relative_strength, 2),
        is_rotating_in=is_rotating_in,
        is_rotating_out=is_rotating_out,
        signal=signal,
        confidence=round(confidence, 2)
    )


def sector_momentum_gate(symbol: str, direction: str = "UP") -> Tuple[bool, str, float]:
    """
    Gate: Check if sector momentum aligns with prediction direction.
    
    Args:
        symbol: Stock symbol
        direction: Prediction direction ("UP" or "DOWN")
    
    Returns:
        (allow: bool, reason: str, confidence_modifier: float)
        
    confidence_modifier:
        > 1.0 = Sector aligns, boost confidence
        1.0 = Neutral
        < 1.0 = Sector opposes, reduce confidence
        0.0 = Block prediction entirely
    """
    momentum = analyze_sector_momentum(symbol)
    
    if momentum is None:
        # Unknown sector - allow with neutral modifier
        return True, "Sector unknown", 1.0
    
    sector_name = momentum.sector_name
    
    # Check alignment
    if direction == "UP":
        if momentum.signal == "BULLISH":
            # Perfect alignment
            modifier = 1.0 + (momentum.confidence - 0.5) * 0.2  # Up to +10%
            return True, f"✅ {sector_name} bullish ({momentum.trend_5d:+.1f}% 5d)", modifier
        
        elif momentum.signal == "BEARISH":
            # Opposing - reduce confidence significantly
            if momentum.is_rotating_out:
                # Active rotation out - BLOCK
                return False, f"🚫 {sector_name} rotating OUT ({momentum.trend_5d:+.1f}% 5d)", 0.0
            else:
                # Bearish but not actively rotating - allow with penalty
                modifier = 0.7  # -30% confidence
                return True, f"⚠️ {sector_name} bearish ({momentum.trend_5d:+.1f}% 5d)", modifier
        
        else:  # NEUTRAL
            return True, f"➖ {sector_name} neutral ({momentum.trend_5d:+.1f}% 5d)", 1.0
    
    elif direction == "DOWN":
        if momentum.signal == "BEARISH":
            # Perfect alignment for short
            modifier = 1.0 + (momentum.confidence - 0.5) * 0.2
            return True, f"✅ {sector_name} bearish (short aligned)", modifier
        
        elif momentum.signal == "BULLISH":
            # Opposing short
            if momentum.is_rotating_in:
                return False, f"🚫 {sector_name} rotating IN (don't short)", 0.0
            else:
                modifier = 0.7
                return True, f"⚠️ {sector_name} bullish (short risky)", modifier
        
        else:
            return True, f"➖ {sector_name} neutral", 1.0
    
    return True, "Unknown direction", 1.0


def get_sector_leaderboard() -> List[Dict[str, Any]]:
    """
    Get all sectors ranked by momentum.
    
    Returns list of sectors sorted by 5-day performance.
    """
    leaderboard = []
    
    for sector_etf, sector_name in SECTOR_NAMES.items():
        trend_5d = get_sector_trend(sector_etf, days=5)
        trend_20d = get_sector_trend(sector_etf, days=20)
        
        if trend_5d is not None:
            leaderboard.append({
                "sector": sector_etf,
                "name": sector_name,
                "trend_5d": round(trend_5d, 2),
                "trend_20d": round(trend_20d, 2) if trend_20d else None,
                "momentum": "BULLISH" if trend_5d > 1 else "BEARISH" if trend_5d < -1 else "NEUTRAL"
            })
    
    # Sort by 5-day trend (best first)
    leaderboard.sort(key=lambda x: x["trend_5d"], reverse=True)
    
    return leaderboard


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("📊 Sector Momentum Test")
    print("=" * 50)
    
    # Test sector lookup
    test_symbols = ["AAPL", "JPM", "XOM", "TSLA", "UNKNOWN"]
    
    for symbol in test_symbols:
        sector = get_sector_for_symbol(symbol)
        print(f"{symbol}: {sector or 'N/A'} ({get_sector_name(sector) if sector else 'Unknown'})")
    
    print("\n" + "=" * 50)
    print("Sector Momentum Analysis:")
    
    for symbol in ["AAPL", "JPM", "XOM"]:
        momentum = analyze_sector_momentum(symbol)
        if momentum:
            print(f"\n{symbol} ({momentum.sector_name}):")
            print(f"  5d trend: {momentum.trend_5d:+.1f}%")
            print(f"  20d trend: {momentum.trend_20d:+.1f}%")
            print(f"  Relative strength: {momentum.relative_strength:+.1f}%")
            print(f"  Signal: {momentum.signal}")
            print(f"  Rotating in: {momentum.is_rotating_in}")
            print(f"  Rotating out: {momentum.is_rotating_out}")
    
    print("\n" + "=" * 50)
    print("Sector Gate Test (AAPL BUY):")
    allow, reason, modifier = sector_momentum_gate("AAPL", "UP")
    print(f"  Allow: {allow}")
    print(f"  Reason: {reason}")
    print(f"  Confidence modifier: {modifier:.2f}x")
