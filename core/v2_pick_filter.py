#!/usr/bin/env python3
"""
🎯 GHOST PROTOCOL V2 - PICK QUALITY FILTER

Integrates V2 quality system into the prediction selection flow.

BEFORE V2: Predict everything, send top 20
AFTER V2: Filter by quality, send top 3-5 only

Quality Gates:
1. Check whitelist/blacklist status
2. Verify confidence meets requirements
3. Require multi-signal alignment (3+ signals)
4. Check recent performance trend
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

LOGGER = logging.getLogger("ghost.v2_pick_filter")


@dataclass
class FilteredPick:
    """A prediction that passed all quality gates"""
    symbol: str
    direction: str
    confidence: float
    entry_price: float
    target_price: float
    stop_price: float
    signals: List[str]
    signal_alignment: int
    quality_status: str  # "whitelist", "watchlist", "unknown"
    historical_wr: Optional[float]
    conviction_score: float


class V2PickQualityFilter:
    """
    Filters predictions through Ghost Protocol V2 quality gates.
    
    Usage:
        filter = get_pick_filter()
        candidates = generate_all_predictions()
        top_picks = filter.select_daily_picks(candidates, max_picks=5)
    """
    
    def __init__(self):
        from core.v2_quality import get_quality_system
        
        self.quality = get_quality_system()
        
        # V2 Configuration
        self.MIN_SIGNAL_ALIGNMENT = 3  # At least 3 signals must agree
        self.MAX_DAILY_PICKS = 5       # Maximum picks per day
        self.MIN_CONVICTION_SCORE = 0.70  # Minimum conviction to send
        
        LOGGER.info("[V2-FILTER] Pick quality filter initialized")
    
    def evaluate_pick(self, prediction: Dict[str, Any]) -> Optional[FilteredPick]:
        """
        Evaluate a single prediction through all quality gates.
        
        Returns FilteredPick if passed, None if rejected.
        
        Quality Gates:
        1. Symbol quality check (whitelist/blacklist)
        2. Confidence requirement based on status
        3. Signal alignment (3+ signals agree)
        4. Market condition (skip if choppy)
        """
        symbol = prediction.get('symbol', 'UNKNOWN')
        confidence = prediction.get('confidence', 0.0)
        direction = prediction.get('direction', 'UNKNOWN')
        signals = prediction.get('signals', [])
        
        # Gate 1: Symbol Quality
        should_predict, reason = self.quality.should_predict(symbol, confidence)
        if not should_predict:
            LOGGER.debug(f"[V2-FILTER] ❌ {symbol} rejected: {reason}")
            return None
        
        # Gate 2: Signal Alignment
        signal_alignment = self._count_signal_alignment(signals, direction)
        if signal_alignment < self.MIN_SIGNAL_ALIGNMENT:
            LOGGER.debug(f"[V2-FILTER] ❌ {symbol} rejected: only {signal_alignment} signals align (need {self.MIN_SIGNAL_ALIGNMENT})")
            return None
        
        # Gate 3: Market Condition
        market_condition = prediction.get('market_condition', 'unknown')
        if market_condition == 'choppy':
            LOGGER.debug(f"[V2-FILTER] ❌ {symbol} rejected: choppy market conditions")
            return None
        
        # Get historical performance
        metrics = self.quality.get_asset_metrics(symbol)
        historical_wr = metrics.win_rate if metrics else None
        quality_status = metrics.status if metrics else "unknown"
        
        # Calculate conviction score
        conviction_score = self._calculate_conviction(
            confidence=confidence,
            signal_alignment=signal_alignment,
            historical_wr=historical_wr,
            quality_status=quality_status
        )
        
        if conviction_score < self.MIN_CONVICTION_SCORE:
            LOGGER.debug(f"[V2-FILTER] ❌ {symbol} rejected: conviction {conviction_score:.2f} < {self.MIN_CONVICTION_SCORE:.2f}")
            return None
        
        # PASSED all gates!
        pick = FilteredPick(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=prediction.get('entry_price', prediction.get('current', 0)),
            target_price=prediction.get('target_price', 0),
            stop_price=prediction.get('stop_price', 0),
            signals=[s.get('name', str(s)) for s in signals if isinstance(s, dict)],
            signal_alignment=signal_alignment,
            quality_status=quality_status,
            historical_wr=historical_wr,
            conviction_score=conviction_score
        )
        
        LOGGER.info(f"[V2-FILTER] ✅ {symbol} PASSED: conviction {conviction_score:.2f}, {signal_alignment} signals, {quality_status}")
        return pick
    
    def _count_signal_alignment(self, signals: List, target_direction: str) -> int:
        """
        Count how many signals agree with the target direction.
        
        Signals can be:
        - List of dicts: [{"name": "MACD", "direction": "UP"}, ...]
        - List of strings: ["MACD_bullish", "RSI_oversold", ...]
        """
        if not signals:
            return 0
        
        aligned = 0
        
        for signal in signals:
            if isinstance(signal, dict):
                sig_dir = signal.get('direction', '')
                if sig_dir == target_direction:
                    aligned += 1
            elif isinstance(signal, str):
                # Parse string signals
                if target_direction == "UP" and any(x in signal.lower() for x in ['bullish', 'buy', 'long', 'up']):
                    aligned += 1
                elif target_direction == "DOWN" and any(x in signal.lower() for x in ['bearish', 'sell', 'short', 'down']):
                    aligned += 1
        
        return aligned
    
    def _calculate_conviction(
        self,
        confidence: float,
        signal_alignment: int,
        historical_wr: Optional[float],
        quality_status: str
    ) -> float:
        """
        Calculate conviction score (0-1) based on multiple factors.
        
        Factors:
        - Model confidence (0.6 weight)
        - Signal alignment (0.2 weight)
        - Historical win rate (0.2 weight)
        """
        # Base: Model confidence
        score = confidence * 0.6
        
        # Boost for signal alignment (max +0.2)
        alignment_boost = min(signal_alignment / 5, 1.0) * 0.2
        score += alignment_boost
        
        # Adjust for historical performance (max ±0.2)
        if historical_wr is not None:
            # 50% WR = neutral (0), 60% WR = +0.1, 70% WR = +0.2
            historical_factor = (historical_wr - 0.5) * 0.4
            score += historical_factor
        elif quality_status == "whitelist":
            score += 0.1  # Boost for whitelist even without exact WR
        elif quality_status == "blacklist":
            score -= 0.2  # Penalty for blacklist
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def select_daily_picks(
        self,
        candidates: List[Dict[str, Any]],
        max_picks: Optional[int] = None
    ) -> List[FilteredPick]:
        """
        Select the best daily picks from all candidates.
        
        Process:
        1. Evaluate each candidate through quality gates
        2. Sort by conviction score
        3. Return top N picks (default: MAX_DAILY_PICKS)
        4. If fewer than 2 pass, return empty list (no prediction day)
        
        Args:
            candidates: All generated predictions
            max_picks: Maximum picks to return (default: self.MAX_DAILY_PICKS)
        
        Returns:
            List of FilteredPick objects, sorted by conviction
        """
        if max_picks is None:
            max_picks = self.MAX_DAILY_PICKS
        
        LOGGER.info(f"[V2-FILTER] Evaluating {len(candidates)} candidates for daily picks...")
        
        # Evaluate all candidates
        passed = []
        for candidate in candidates:
            pick = self.evaluate_pick(candidate)
            if pick:
                passed.append(pick)
        
        if not passed:
            LOGGER.warning(f"[V2-FILTER] ⚠️  NO picks passed quality gates out of {len(candidates)} candidates")
            return []
        
        # Sort by conviction score
        passed.sort(key=lambda p: p.conviction_score, reverse=True)
        
        # Take top N
        selected = passed[:max_picks]
        
        # Minimum picks threshold (avoid low-quality days)
        if len(selected) < 2:
            LOGGER.warning(f"[V2-FILTER] 🔇 Only {len(selected)} pick(s) passed - skipping predictions today")
            return []
        
        LOGGER.info(f"[V2-FILTER] ✅ Selected {len(selected)}/{len(passed)} top picks (from {len(candidates)} candidates)")
        
        # Log details
        for i, pick in enumerate(selected, 1):
            LOGGER.info(f"[V2-FILTER]   {i}. {pick.symbol} {pick.direction} — conviction {pick.conviction_score:.2f}, "
                       f"{pick.signal_alignment} signals, {pick.quality_status}")
        
        return selected
    
    def should_send_predictions_today(self, picks: List[FilteredPick]) -> tuple[bool, str]:
        """
        Determine if we should send predictions today.
        
        Returns:
            (should_send: bool, reason: str)
        
        Reasons to skip:
        - Fewer than 2 picks
        - All picks are watchlist (not confident enough)
        - Market conditions unfavorable
        """
        if not picks:
            return False, "No picks passed quality gates"
        
        if len(picks) < 2:
            return False, f"Only {len(picks)} pick passed (need minimum 2)"
        
        # Check if we have at least one whitelist pick
        whitelist_count = sum(1 for p in picks if p.quality_status == "whitelist")
        if whitelist_count == 0:
            return False, "No proven performers in picks (all watchlist/unknown)"
        
        # Check average conviction
        avg_conviction = sum(p.conviction_score for p in picks) / len(picks)
        if avg_conviction < 0.75:
            return False, f"Average conviction too low ({avg_conviction:.2f} < 0.75)"
        
        return True, f"{len(picks)} high-conviction picks ready"


# ============================================================================
# Singleton
# ============================================================================

_pick_filter: Optional[V2PickQualityFilter] = None

def get_pick_filter() -> V2PickQualityFilter:
    """Get singleton pick filter"""
    global _pick_filter
    if _pick_filter is None:
        _pick_filter = V2PickQualityFilter()
    return _pick_filter


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("\n" + "=" * 70)
    print("🎯 GHOST PROTOCOL V2 - PICK QUALITY FILTER TEST")
    print("=" * 70)
    
    # First, update quality system
    print("\n📊 Updating quality system...")
    from core.v2_quality import get_quality_system
    quality = get_quality_system()
    quality.update_from_verification(30)
    
    # Create filter
    filter_sys = get_pick_filter()
    
    # Test cases
    test_candidates = [
        {
            "symbol": "BTC",
            "direction": "UP",
            "confidence": 0.85,
            "signals": [
                {"name": "MACD", "direction": "UP"},
                {"name": "RSI", "direction": "UP"},
                {"name": "Volume", "direction": "UP"},
                {"name": "Trend", "direction": "UP"}
            ],
            "market_condition": "trending",
            "current": 45000,
        },
        {
            "symbol": "UNKNOWN_BAD",
            "direction": "UP",
            "confidence": 0.60,
            "signals": [{"name": "MACD", "direction": "UP"}],
            "market_condition": "choppy",
            "current": 100,
        },
        {
            "symbol": "ETH",
            "direction": "DOWN",
            "confidence": 0.78,
            "signals": [
                {"name": "RSI", "direction": "DOWN"},
                {"name": "Momentum", "direction": "DOWN"},
                {"name": "Volume", "direction": "DOWN"}
            ],
            "market_condition": "trending",
            "current": 2500,
        }
    ]
    
    print("\n🧪 TESTING INDIVIDUAL PREDICTIONS:")
    for candidate in test_candidates:
        pick = filter_sys.evaluate_pick(candidate)
        if pick:
            print(f"   ✅ {pick.symbol}: conviction {pick.conviction_score:.2f}")
        else:
            print(f"   ❌ {candidate['symbol']}: REJECTED")
    
    print("\n🎯 SELECTING DAILY PICKS:")
    picks = filter_sys.select_daily_picks(test_candidates, max_picks=5)
    
    should_send, reason = filter_sys.should_send_predictions_today(picks)
    if should_send:
        print(f"   ✅ SEND TODAY: {reason}")
        for i, pick in enumerate(picks, 1):
            print(f"      {i}. {pick.symbol} {pick.direction} @ {pick.confidence*100:.0f}% conf, conviction {pick.conviction_score:.2f}")
    else:
        print(f"   🔇 SKIP TODAY: {reason}")
    
    print("\n" + "=" * 70)
