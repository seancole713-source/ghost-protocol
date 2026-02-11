"""
V3 Filter: Scores and filters predictions using validated strategies.

Passes predictions that:
1. Are in V3_VALIDATED_STRATEGIES (ETH, XRP, LINK) — full V3 scoring, OR
2. Are in the EDGE WHITELIST — scored at 0.55 × confidence (proven paper trade performance)
3. Meet minimum confidence threshold
4. For inverse strategies: Ghost predicted the trigger direction (DOWN)

Based on 52,433 trade backtest analysis + edge whitelist paper trade validation.
"""
import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("ghost")

# Inline constants (avoids config/settings.py → pydantic_settings import chain)
_V3_MIN_CONFIDENCE = float(os.getenv("V3_MIN_CONFIDENCE", "0.70"))
_DEFAULT_TARGET_PCT = 0.066  # 6.6%
_DEFAULT_STOP_PCT = 0.033    # 3.3%

from config.symbols import (
    V3_VALIDATED_STRATEGIES, 
    V3_REMOVED_SYMBOLS,
    V3_BLACKLIST,
    ValidatedStrategy,
    is_blacklisted,
    is_removed,
)
from core.models import Prediction, ScoredPrediction, Direction, FilterResult

# Edge whitelist: symbols with proven paper trade performance
# These bypass V3_VALIDATED_STRATEGIES gate and get scored at 0.55 × confidence
_EDGE_CSV = os.getenv("EDGE_SYMBOLS",
    "T,GME,TURBO,RNDR,ENJ,JUP,BAND,HOOD,IQ,BMBL,HBAR,XPO,"
    "PEPE,IOTX,GIGA,COIN,ILV,BCH,CHZ,ALICE,YFI,ITRI,ICP,BRETT"
)
_EDGE_SET = frozenset(s.strip().upper() for s in _EDGE_CSV.split(",") if s.strip())


class V3Filter:
    """
    Filter predictions using V3 validated strategies.
    
    This is the core quality gate for Ghost Protocol V3.
    Only predictions that pass this filter should be acted upon.
    """
    
    def __init__(self, min_confidence: Optional[float] = None):
        """
        Initialize V3 filter.
        
        Args:
            min_confidence: Minimum confidence threshold (default from settings)
        """
        self.min_confidence = min_confidence or _V3_MIN_CONFIDENCE
        self._stats = {
            'total_processed': 0,
            'passed': 0,
            'rejected_not_validated': 0,
            'rejected_low_confidence': 0,
            'rejected_wrong_direction': 0,
            'rejected_blacklisted': 0,
            'inversed': 0,
        }
    
    @property
    def stats(self) -> dict:
        """Get filter statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset filter statistics."""
        for key in self._stats:
            self._stats[key] = 0
    
    def filter_and_score(
        self, 
        predictions: List[Prediction],
        max_results: int = 10
    ) -> List[ScoredPrediction]:
        """
        Filter predictions through V3 validation.
        
        Args:
            predictions: Raw predictions from engine
            max_results: Maximum number of results to return
            
        Returns:
            List of scored predictions, sorted by score descending
        """
        scored = []
        
        for pred in predictions:
            self._stats['total_processed'] += 1
            result = self._process_prediction(pred)
            
            if result.passed and result.prediction:
                self._stats['passed'] += 1
                scored.append(result.prediction)
        
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        
        return scored[:max_results]
    
    def filter_single(self, pred: Prediction) -> FilterResult:
        """
        Filter a single prediction and return detailed result.
        
        Useful for debugging and understanding why a prediction was filtered.
        """
        return self._process_prediction(pred)
    
    def _process_prediction(self, pred: Prediction) -> FilterResult:
        """Process a single prediction through V3 filter."""
        symbol = pred.symbol.upper()
        
        # Check blacklist first — edge symbols bypass blacklist
        if is_blacklisted(symbol) and symbol not in _EDGE_SET:
            self._stats['rejected_blacklisted'] += 1
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"BLACKLISTED: {symbol} is on the blacklist"
            )
        
        # Check if in removed symbols — edge symbols bypass this check
        if is_removed(symbol) and symbol not in _EDGE_SET:
            self._stats['rejected_not_validated'] += 1
            reason = V3_REMOVED_SYMBOLS.get(symbol, "Not statistically significant")
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"REMOVED: {reason}"
            )
        
        # Must be in validated strategies OR edge whitelist
        if symbol not in V3_VALIDATED_STRATEGIES:
            # EDGE WHITELIST PASSTHROUGH: edge symbols get scored at 0.55 × confidence
            if symbol in _EDGE_SET:
                return self._process_edge(pred)
            
            self._stats['rejected_not_validated'] += 1
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"NOT_VALIDATED: {symbol} not in V3 validated strategies"
            )
        
        strategy = V3_VALIDATED_STRATEGIES[symbol]
        
        # Confidence check applies to ALL strategies
        if pred.confidence < self.min_confidence:
            self._stats['rejected_low_confidence'] += 1
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"LOW_CONFIDENCE: {pred.confidence:.0%} < {self.min_confidence:.0%}"
            )
        
        # Route to appropriate handler
        if strategy.strategy == 'ghost_inverse':
            return self._process_inverse(pred, strategy)
        else:
            return self._process_normal(pred, strategy)
    
    def _process_inverse(
        self, 
        pred: Prediction, 
        strategy: ValidatedStrategy
    ) -> FilterResult:
        """
        Process ghost_inverse strategy.
        
        Only triggers when Ghost predicts the opposite of what we want.
        E.g., ETH: Ghost says DOWN -> we flip to UP (our validated edge)
        """
        symbol = pred.symbol.upper()
        
        # Inverse requires Ghost to predict DOWN
        if pred.direction != Direction.DOWN:
            self._stats['rejected_wrong_direction'] += 1
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"WRONG_DIRECTION: inverse requires DOWN, got {pred.direction}"
            )
        
        # Flip direction
        new_direction = Direction(strategy.direction_override)
        
        # Recalculate targets for flipped direction
        target_price, stop_loss = self._calculate_inverse_targets(pred.current_price)
        
        score = strategy.backtest_win_rate * pred.confidence
        self._stats['inversed'] += 1
        
        logger.info(
            f"[V3] 🔄 INVERSE {symbol}: {pred.direction} → {new_direction} "
            f"(conf={pred.confidence:.0%}, score={score:.3f})"
        )
        
        scored_pred = ScoredPrediction(
            symbol=symbol,
            direction=new_direction,
            confidence=pred.confidence,
            current_price=pred.current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            hold_hours=strategy.hold_hours,
            timestamp=pred.timestamp,
            strategy=strategy.strategy,
            original_direction=pred.direction,
            is_inverse=True,
            backtest_win_rate=strategy.backtest_win_rate,
            score=score,
            news_influenced=pred.news_influenced,
            asset_type=pred.asset_type,
        )
        
        return FilterResult(
            passed=True,
            symbol=symbol,
            reason=f"PASSED: inverse {pred.direction} → {new_direction}",
            prediction=scored_pred,
        )
    
    def _process_normal(
        self, 
        pred: Prediction, 
        strategy: ValidatedStrategy
    ) -> FilterResult:
        """Process normal (non-inverse) strategies like mean_reversion."""
        symbol = pred.symbol.upper()
        
        # Use direction override if specified, otherwise use prediction direction
        direction = (
            Direction(strategy.direction_override) 
            if strategy.direction_override 
            else pred.direction
        )
        
        score = strategy.backtest_win_rate * pred.confidence
        
        logger.info(
            f"[V3] ✅ PASS {symbol}: {direction} "
            f"(strategy={strategy.strategy}, conf={pred.confidence:.0%}, score={score:.3f})"
        )
        
        scored_pred = ScoredPrediction(
            symbol=symbol,
            direction=direction,
            confidence=pred.confidence,
            current_price=pred.current_price,
            target_price=pred.target_price,
            stop_loss=pred.stop_loss,
            hold_hours=strategy.hold_hours,
            timestamp=pred.timestamp,
            strategy=strategy.strategy,
            original_direction=pred.direction,
            is_inverse=False,
            backtest_win_rate=strategy.backtest_win_rate,
            score=score,
            news_influenced=pred.news_influenced,
            asset_type=pred.asset_type,
        )
        
        return FilterResult(
            passed=True,
            symbol=symbol,
            reason=f"PASSED: {strategy.strategy}",
            prediction=scored_pred,
        )
    
    def _calculate_inverse_targets(self, current_price: float) -> Tuple[float, float]:
        """
        Calculate target and stop for inversed (flipped to UP) trade.
        
        Returns:
            Tuple of (target_price, stop_loss)
        """
        target_price = current_price * (1 + _DEFAULT_TARGET_PCT)
        stop_loss = current_price * (1 - _DEFAULT_STOP_PCT)
        return target_price, stop_loss

    def _process_edge(self, pred: Prediction) -> FilterResult:
        """
        Process edge whitelist symbol (not in V3_VALIDATED_STRATEGIES).
        
        Edge symbols have proven paper trade performance but no V3 backtest
        validation. They get scored at 0.55 × confidence (conservative).
        """
        symbol = pred.symbol.upper()
        
        # Minimum confidence check (use lower threshold for edge: 0.50)
        if pred.confidence < 0.50:
            self._stats['rejected_low_confidence'] += 1
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"LOW_CONFIDENCE: {pred.confidence:.0%} < 50% (edge threshold)"
            )
        
        # Edge symbols scored at 0.55 × confidence (conservative)
        score = 0.55 * pred.confidence
        
        logger.info(
            f"[V3] 🎯 EDGE {symbol}: {pred.direction} "
            f"(edge_whitelist, conf={pred.confidence:.0%}, score={score:.3f})"
        )
        
        scored_pred = ScoredPrediction(
            symbol=symbol,
            direction=pred.direction,
            confidence=pred.confidence,
            current_price=pred.current_price,
            target_price=pred.target_price,
            stop_loss=pred.stop_loss,
            hold_hours=48,  # Default 48h hold for edge symbols
            timestamp=pred.timestamp,
            strategy='edge_whitelist',
            original_direction=pred.direction,
            is_inverse=False,
            backtest_win_rate=0.55,  # Conservative estimate
            score=score,
            news_influenced=pred.news_influenced,
            asset_type=pred.asset_type,
        )
        
        return FilterResult(
            passed=True,
            symbol=symbol,
            reason=f"PASSED: edge_whitelist",
            prediction=scored_pred,
        )


# Singleton instance for convenience
v3_filter = V3Filter()
