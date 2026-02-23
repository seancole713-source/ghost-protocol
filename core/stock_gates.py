#!/usr/bin/env python3
"""
🚪 STOCK GATES - Stock-specific regime and confirmation gates

Stocks require stricter gates than crypto:
1. Market hours only (9:30 AM - 4:00 PM ET)
2. VIX < 20 (strict fear threshold)
3. SPY above 20MA (bull market required)
4. Economic calendar clear
5. Sector momentum aligned
6. Multi-timeframe confirmation
7. Earnings blackout
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo

from core.economic_calendar import economic_calendar_gate, get_upcoming_events
from core.sector_momentum import sector_momentum_gate, analyze_sector_momentum
from core.multi_timeframe import multi_timeframe_gate, get_timeframe_confirmations

LOGGER = logging.getLogger("ghost.stock_gates")

# Eastern timezone for market hours
ET = ZoneInfo("America/New_York")

# Stock-specific thresholds (stricter than crypto)
STOCK_VIX_MAX = 20.0  # vs 25 for crypto
STOCK_MIN_CONFIRMATIONS = 4  # vs 3 for crypto
STOCK_RSI_OVERSOLD = 35  # vs 30 for crypto
STOCK_RSI_OVERBOUGHT = 65  # vs 70 for crypto


@dataclass
class GateResult:
    """Result from a gate check"""
    passed: bool
    gate_name: str
    reason: str
    confidence_modifier: float = 1.0
    data: Optional[Dict[str, Any]] = None


@dataclass
class StockGateResults:
    """Combined results from all stock gates"""
    symbol: str
    all_passed: bool
    gates: List[GateResult]
    final_confidence_modifier: float
    block_reason: Optional[str]
    confirmations: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "all_passed": self.all_passed,
            "gates": [
                {
                    "name": g.gate_name,
                    "passed": g.passed,
                    "reason": g.reason,
                    "modifier": g.confidence_modifier
                }
                for g in self.gates
            ],
            "final_modifier": self.final_confidence_modifier,
            "block_reason": self.block_reason,
            "confirmations": self.confirmations
        }


class StockMarketHoursGate:
    """
    Gate: Only predict during market hours.
    
    Regular trading: 9:30 AM - 4:00 PM ET
    Extended hours: 4:00 AM - 8:00 PM ET (pre/post market)
    Pre-market card: 7:00 AM - 9:30 AM ET (for 8 AM CT daily card)
    
    For strict mode: Only regular hours
    For extended mode: Include pre/post market
    For premarket_card mode: Allow pre-market predictions using yesterday's close
    """
    
    def __init__(self, strict: bool = True, allow_premarket_card: bool = False):
        self.strict = strict
        self.allow_premarket_card = allow_premarket_card
    
    def check(self, date: datetime = None) -> GateResult:
        if date is None:
            date = datetime.now(ET)
        
        # Convert to ET if needed
        if date.tzinfo is None:
            date = date.replace(tzinfo=ET)
        else:
            date = date.astimezone(ET)
        
        hour = date.hour
        minute = date.minute
        weekday = date.weekday()
        
        # Weekend check
        if weekday >= 5:
            return GateResult(
                passed=False,
                gate_name="MarketHours",
                reason=f"Weekend (day {weekday})",
                confidence_modifier=0.0
            )
        
        # Time check
        time_decimal = hour + minute / 60
        
        if self.strict:
            # Regular hours only: 9:30 AM - 4:00 PM
            market_open = 9.5  # 9:30 AM
            market_close = 16.0  # 4:00 PM
        else:
            # Extended hours: 4:00 AM - 8:00 PM
            market_open = 4.0
            market_close = 20.0
        
        if market_open <= time_decimal <= market_close:
            return GateResult(
                passed=True,
                gate_name="MarketHours",
                reason=f"Market open ({hour}:{minute:02d} ET)",
                confidence_modifier=1.0
            )
        elif self.allow_premarket_card and 7.0 <= time_decimal < 9.5:
            # Pre-market card window: yesterday's close data is valid
            # The stock market doesn't open until 9:30 AM ET, so there
            # is literally no fresher data to wait for.
            return GateResult(
                passed=True,
                gate_name="MarketHours",
                reason=f"Pre-market card ({hour}:{minute:02d} ET, daily bars valid)",
                confidence_modifier=0.95  # Slight discount for pre-market
            )
        else:
            return GateResult(
                passed=False,
                gate_name="MarketHours",
                reason=f"Market closed ({hour}:{minute:02d} ET)",
                confidence_modifier=0.0
            )


class StockVIXGate:
    """
    Gate: VIX must be below threshold.
    
    Stock predictions fail when fear is high.
    VIX > 20: Elevated fear, reduce confidence
    VIX > 25: High fear, BLOCK
    VIX > 30: Extreme fear, definitely BLOCK
    """
    
    def __init__(self, max_vix: float = STOCK_VIX_MAX):
        self.max_vix = max_vix
    
    def check(self, vix_level: float = None) -> GateResult:
        if vix_level is None:
            vix_level = self._get_vix()
        
        if vix_level is None:
            # Can't get VIX - allow with warning
            return GateResult(
                passed=True,
                gate_name="VIX",
                reason="VIX unavailable (allowing)",
                confidence_modifier=0.9
            )
        
        if vix_level <= self.max_vix:
            return GateResult(
                passed=True,
                gate_name="VIX",
                reason=f"VIX {vix_level:.1f} ≤ {self.max_vix} ✅",
                confidence_modifier=1.0 + (self.max_vix - vix_level) / 100,  # Bonus for low VIX
                data={"vix": vix_level}
            )
        elif vix_level <= 25:
            return GateResult(
                passed=True,
                gate_name="VIX",
                reason=f"VIX {vix_level:.1f} elevated (caution)",
                confidence_modifier=0.8,
                data={"vix": vix_level}
            )
        else:
            return GateResult(
                passed=False,
                gate_name="VIX",
                reason=f"VIX {vix_level:.1f} > {self.max_vix} 🚫",
                confidence_modifier=0.0,
                data={"vix": vix_level}
            )
    
    def _get_vix(self) -> Optional[float]:
        """Fetch current VIX level"""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except Exception as e:
            LOGGER.warning(f"Failed to get VIX: {e}")
        return None


class StockSPYRegimeGate:
    """
    Gate: SPY must be above 20-day MA (bull market).
    
    Don't fight the trend - if SPY is bearish, stock predictions suffer.
    """
    
    def check(self) -> GateResult:
        try:
            import yfinance as yf
            
            spy = yf.Ticker("SPY")
            hist = spy.history(period="30d")
            
            if len(hist) < 20:
                return GateResult(
                    passed=True,
                    gate_name="SPYRegime",
                    reason="Insufficient SPY data (allowing)",
                    confidence_modifier=0.9
                )
            
            current_price = hist['Close'].iloc[-1]
            ma20 = hist['Close'].rolling(20).mean().iloc[-1]
            
            pct_vs_ma = ((current_price - ma20) / ma20) * 100
            
            if current_price > ma20:
                return GateResult(
                    passed=True,
                    gate_name="SPYRegime",
                    reason=f"SPY above 20MA ({pct_vs_ma:+.1f}%) ✅",
                    confidence_modifier=1.0 + min(0.1, pct_vs_ma / 100),
                    data={"spy_price": current_price, "spy_ma20": ma20, "pct_vs_ma": pct_vs_ma}
                )
            else:
                # Below MA but check how far
                if pct_vs_ma > -2:
                    # Close to MA - allow with caution
                    return GateResult(
                        passed=True,
                        gate_name="SPYRegime",
                        reason=f"SPY near 20MA ({pct_vs_ma:+.1f}%) ⚠️",
                        confidence_modifier=0.85,
                        data={"spy_price": current_price, "spy_ma20": ma20, "pct_vs_ma": pct_vs_ma}
                    )
                else:
                    # Significantly below - BLOCK
                    return GateResult(
                        passed=False,
                        gate_name="SPYRegime",
                        reason=f"SPY below 20MA ({pct_vs_ma:+.1f}%) 🚫",
                        confidence_modifier=0.0,
                        data={"spy_price": current_price, "spy_ma20": ma20, "pct_vs_ma": pct_vs_ma}
                    )
                    
        except Exception as e:
            LOGGER.warning(f"SPY regime check failed: {e}")
            return GateResult(
                passed=True,
                gate_name="SPYRegime",
                reason=f"SPY check failed: {e}",
                confidence_modifier=0.9
            )


class StockConfirmationCounter:
    """
    Count bullish/bearish confirmations for stocks.
    
    Requires MORE confirmations than crypto (4 vs 3).
    Uses stock-tuned thresholds (RSI 35/65 vs 30/70).
    """
    
    def __init__(
        self,
        min_confirmations: int = STOCK_MIN_CONFIRMATIONS,
        rsi_oversold: float = STOCK_RSI_OVERSOLD,
        rsi_overbought: float = STOCK_RSI_OVERBOUGHT
    ):
        self.min_confirmations = min_confirmations
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    async def count_buy_confirmations(
        self,
        symbol: str,
        metrics: Dict[str, Any],
        spy_above_ma: bool = True,
        vix_level: float = 18.0
    ) -> Tuple[int, List[str]]:
        """
        Count bullish confirmations for stocks.
        
        Confirmations:
        1. RSI < 35 (oversold - stricter than crypto's 30)
        2. MACD bullish crossover
        3. Price near support (Bollinger lower)
        4. SPY above 20MA
        5. VIX < 20
        6. Sector momentum bullish
        7. Multi-timeframe alignment (Daily + 4H + 1H)
        8. Volume spike (institutional buying)
        
        Returns: (count, reasons)
        """
        confirmations = 0
        reasons = []
        
        # 1. RSI Oversold (< 35 for stocks)
        rsi = metrics.get("rsi_14", metrics.get("RSI_14", metrics.get("rsi", 50)))
        if rsi and rsi < self.rsi_oversold:
            confirmations += 1
            reasons.append(f"RSI oversold ({rsi:.0f} < {self.rsi_oversold})")
        
        # 2. MACD Bullish
        macd_hist = metrics.get("macd_histogram", metrics.get("MACD_HISTOGRAM", 0))
        macd_prev = metrics.get("macd_histogram_prev", macd_hist)
        if macd_hist and macd_hist > 0:
            if macd_prev and macd_prev <= 0:
                confirmations += 1.5  # Crossover gets extra credit
                reasons.append("MACD bullish crossover")
            else:
                confirmations += 0.5
                reasons.append("MACD positive")
        
        # 3. Price near support
        bb_lower = metrics.get("bb_lower", metrics.get("BB_LOWER", 0))
        price = metrics.get("current_price", metrics.get("price", 0))
        if bb_lower and price and bb_lower > 0:
            distance = (price - bb_lower) / price
            if distance < 0.02:  # Within 2% of lower band
                confirmations += 1
                reasons.append("Price near support (BB lower)")
        
        # 4. SPY above 20MA
        if spy_above_ma:
            confirmations += 1
            reasons.append("SPY above 20MA (bull market)")
        
        # 5. VIX low
        if vix_level < STOCK_VIX_MAX:
            confirmations += 1
            reasons.append(f"VIX low ({vix_level:.1f})")
        elif vix_level < 25:
            confirmations += 0.5
            reasons.append(f"VIX moderate ({vix_level:.1f})")
        
        # 6. Sector momentum
        try:
            allow, reason, modifier = sector_momentum_gate(symbol, "UP")
            if allow and modifier > 1.0:
                confirmations += 1
                reasons.append(reason)
            elif allow and modifier < 1.0:
                reasons.append(f"⚠️ {reason}")
        except Exception:
            pass
        
        # 7. Multi-timeframe alignment
        try:
            mtf_confirms = get_timeframe_confirmations(symbol, "UP")
            if mtf_confirms >= 3:
                confirmations += 1.5
                reasons.append(f"All 3 timeframes bullish")
            elif mtf_confirms >= 2:
                confirmations += 1
                reasons.append(f"{mtf_confirms}/3 timeframes bullish")
            elif mtf_confirms == 1:
                confirmations += 0.5
                reasons.append(f"1/3 timeframes bullish")
        except Exception:
            pass
        
        # 8. Volume spike
        volume_ratio = metrics.get("volume_ratio", metrics.get("VOLUME_RATIO", 1.0))
        if volume_ratio and volume_ratio > 1.5:
            confirmations += 1
            reasons.append(f"Volume spike ({volume_ratio:.1f}x)")
        elif volume_ratio and volume_ratio > 1.2:
            confirmations += 0.5
            reasons.append(f"Above avg volume ({volume_ratio:.1f}x)")
        
        return int(confirmations), reasons
    
    async def count_sell_confirmations(
        self,
        symbol: str,
        metrics: Dict[str, Any],
        spy_above_ma: bool = True,
        vix_level: float = 18.0
    ) -> Tuple[int, List[str]]:
        """
        Count bearish confirmations for stocks (for SELL/SHORT signals).
        """
        confirmations = 0
        reasons = []
        
        # 1. RSI Overbought (> 65 for stocks)
        rsi = metrics.get("rsi_14", metrics.get("RSI_14", metrics.get("rsi", 50)))
        if rsi and rsi > self.rsi_overbought:
            confirmations += 1
            reasons.append(f"RSI overbought ({rsi:.0f} > {self.rsi_overbought})")
        
        # 2. MACD Bearish
        macd_hist = metrics.get("macd_histogram", metrics.get("MACD_HISTOGRAM", 0))
        if macd_hist and macd_hist < 0:
            confirmations += 1
            reasons.append("MACD bearish")
        
        # 3. Price near resistance
        bb_upper = metrics.get("bb_upper", metrics.get("BB_UPPER", 0))
        price = metrics.get("current_price", metrics.get("price", 0))
        if bb_upper and price and bb_upper > 0:
            distance = (bb_upper - price) / price
            if distance < 0.02:
                confirmations += 1
                reasons.append("Price near resistance (BB upper)")
        
        # 4. SPY below 20MA (bear market)
        if not spy_above_ma:
            confirmations += 1
            reasons.append("SPY below 20MA (bear market)")
        
        # 5. VIX high
        if vix_level > 25:
            confirmations += 1
            reasons.append(f"VIX high ({vix_level:.1f})")
        
        # 6. Sector momentum bearish
        try:
            allow, reason, modifier = sector_momentum_gate(symbol, "DOWN")
            if allow and modifier > 1.0:
                confirmations += 1
                reasons.append(reason)
        except Exception:
            pass
        
        # 7. Multi-timeframe alignment (bearish)
        try:
            mtf_confirms = get_timeframe_confirmations(symbol, "DOWN")
            if mtf_confirms >= 2:
                confirmations += 1
                reasons.append(f"{mtf_confirms}/3 timeframes bearish")
        except Exception:
            pass
        
        return int(confirmations), reasons


async def run_all_stock_gates(
    symbol: str,
    direction: str,
    metrics: Dict[str, Any],
    strict_market_hours: bool = True,
    allow_premarket_card: bool = False
) -> StockGateResults:
    """
    Run all stock gates and return combined result.
    
    Args:
        symbol: Stock symbol
        direction: "UP" or "DOWN"
        metrics: Feature metrics dict
        strict_market_hours: If True, only regular hours. If False, extended hours.
        allow_premarket_card: If True, allow 7-9:30 AM ET for daily card predictions.
    
    Returns:
        StockGateResults with all gate outcomes
    """
    gates = []
    all_passed = True
    block_reason = None
    final_modifier = 1.0
    
    # 1. Market Hours Gate
    market_gate = StockMarketHoursGate(strict=strict_market_hours, allow_premarket_card=allow_premarket_card)
    result = market_gate.check()
    gates.append(result)
    if not result.passed:
        all_passed = False
        block_reason = result.reason
    final_modifier *= result.confidence_modifier
    
    # 2. VIX Gate
    vix_gate = StockVIXGate()
    result = vix_gate.check()
    gates.append(result)
    vix_level = result.data.get("vix", 20) if result.data else 20
    if not result.passed:
        all_passed = False
        if not block_reason:
            block_reason = result.reason
    final_modifier *= result.confidence_modifier
    
    # 3. SPY Regime Gate
    spy_gate = StockSPYRegimeGate()
    result = spy_gate.check()
    gates.append(result)
    spy_above_ma = result.passed
    if not result.passed and result.confidence_modifier == 0:
        all_passed = False
        if not block_reason:
            block_reason = result.reason
    final_modifier *= result.confidence_modifier
    
    # 4. Economic Calendar Gate
    allow, reason = economic_calendar_gate(symbol)
    gates.append(GateResult(
        passed=allow,
        gate_name="EconomicCalendar",
        reason=reason if not allow else "No economic events",
        confidence_modifier=1.0 if allow else 0.0
    ))
    if not allow:
        all_passed = False
        if not block_reason:
            block_reason = reason
    
    # 5. Sector Momentum Gate
    allow, reason, modifier = sector_momentum_gate(symbol, direction)
    gates.append(GateResult(
        passed=allow,
        gate_name="SectorMomentum",
        reason=reason,
        confidence_modifier=modifier
    ))
    if not allow:
        all_passed = False
        if not block_reason:
            block_reason = reason
    final_modifier *= modifier if allow else 1.0
    
    # 6. Multi-Timeframe Gate
    allow, reason, modifier = multi_timeframe_gate(symbol, direction)
    gates.append(GateResult(
        passed=allow,
        gate_name="MultiTimeframe",
        reason=reason,
        confidence_modifier=modifier
    ))
    if not allow:
        all_passed = False
        if not block_reason:
            block_reason = reason
    final_modifier *= modifier if allow else 1.0
    
    # 7. Confirmation Counter
    counter = StockConfirmationCounter()
    if direction == "UP":
        confirmations, reasons = await counter.count_buy_confirmations(
            symbol, metrics, spy_above_ma, vix_level
        )
    else:
        confirmations, reasons = await counter.count_sell_confirmations(
            symbol, metrics, spy_above_ma, vix_level
        )
    
    min_conf = STOCK_MIN_CONFIRMATIONS
    conf_passed = confirmations >= min_conf
    gates.append(GateResult(
        passed=conf_passed,
        gate_name="Confirmations",
        reason=f"{confirmations}/{min_conf} confirmations: " + ", ".join(reasons[:3]),
        confidence_modifier=1.0 if conf_passed else 0.7,
        data={"count": confirmations, "reasons": reasons}
    ))
    if not conf_passed:
        # Don't block, but reduce confidence
        final_modifier *= 0.7
    
    return StockGateResults(
        symbol=symbol,
        all_passed=all_passed,
        gates=gates,
        final_confidence_modifier=max(0.0, min(1.5, final_modifier)),
        block_reason=block_reason,
        confirmations=confirmations
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("🚪 Stock Gates Test")
        print("=" * 50)
        
        # Test individual gates
        print("\n1. Market Hours Gate:")
        gate = StockMarketHoursGate(strict=True)
        result = gate.check()
        print(f"   Passed: {result.passed}")
        print(f"   Reason: {result.reason}")
        
        print("\n2. VIX Gate:")
        gate = StockVIXGate()
        result = gate.check()
        print(f"   Passed: {result.passed}")
        print(f"   Reason: {result.reason}")
        
        print("\n3. SPY Regime Gate:")
        gate = StockSPYRegimeGate()
        result = gate.check()
        print(f"   Passed: {result.passed}")
        print(f"   Reason: {result.reason}")
        
        # Test full gate suite
        print("\n" + "=" * 50)
        print("Full Gate Suite (AAPL BUY):")
        
        metrics = {
            "rsi_14": 38,
            "macd_histogram": 0.5,
            "current_price": 185.0,
            "bb_lower": 180.0,
            "volume_ratio": 1.3
        }
        
        results = await run_all_stock_gates("AAPL", "UP", metrics)
        
        print(f"\nAll Passed: {results.all_passed}")
        print(f"Block Reason: {results.block_reason}")
        print(f"Confirmations: {results.confirmations}")
        print(f"Final Modifier: {results.final_confidence_modifier:.2f}x")
        
        print("\nGate Results:")
        for gate in results.gates:
            status = "✅" if gate.passed else "🚫"
            print(f"  {status} {gate.gate_name}: {gate.reason} ({gate.confidence_modifier:.2f}x)")
    
    asyncio.run(test())
