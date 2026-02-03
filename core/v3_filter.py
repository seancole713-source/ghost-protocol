"""
V3 Filter: Scores and filters predictions using validated strategies.

Only passes predictions that:
1. Are in V3_VALIDATED_STRATEGIES (ETH, XRP, LINK)
2. Meet minimum confidence threshold (70%)
3. For inverse strategies: Ghost predicted the trigger direction (DOWN)

Based on 52,433 trade backtest analysis.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime
from loguru import logger

from config.settings import settings
from config.symbols import (
    V3_VALIDATED_STRATEGIES, 
    V3_REMOVED_SYMBOLS,
    V3_BLACKLIST,
    ValidatedStrategy,
    is_blacklisted,
    is_removed,
)
from core.models import Prediction, ScoredPrediction, Direction, FilterResult


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
        self.min_confidence = min_confidence or settings.V3_MIN_CONFIDENCE
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
        
        # Check blacklist first
        if is_blacklisted(symbol):
            self._stats['rejected_blacklisted'] += 1
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"BLACKLISTED: {symbol} is on the blacklist"
            )
        
        # Check if in removed symbols
        if is_removed(symbol):
            self._stats['rejected_not_validated'] += 1
            reason = V3_REMOVED_SYMBOLS.get(symbol, "Not statistically significant")
            return FilterResult(
                passed=False,
                symbol=symbol,
                reason=f"REMOVED: {reason}"
            )
        
        # Must be in validated strategies
        if symbol not in V3_VALIDATED_STRATEGIES:
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
        target_price = current_price * (1 + settings.DEFAULT_TARGET_PCT)
        stop_loss = current_price * (1 - settings.DEFAULT_STOP_PCT)
        return target_price, stop_loss


# Singleton instance for convenience
v3_filter = V3Filter()
