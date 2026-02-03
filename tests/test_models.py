"""
Tests for core models.
"""
import pytest
from datetime import datetime
from core.models import (
    Direction, 
    TradeOutcome, 
    Prediction, 
    ScoredPrediction,
    PaperTrade,
    ValidationResult,
)


class TestDirection:
    """Direction enum tests."""
    
    def test_direction_values(self):
        """Direction should have UP and DOWN."""
        assert Direction.UP.value == "UP"
        assert Direction.DOWN.value == "DOWN"
    
    def test_direction_opposite(self):
        """opposite() should return the other direction."""
        assert Direction.UP.opposite() == Direction.DOWN
        assert Direction.DOWN.opposite() == Direction.UP


class TestTradeOutcome:
    """TradeOutcome enum tests."""
    
    def test_all_outcomes_exist(self):
        """All expected outcomes should exist."""
        outcomes = [o.value for o in TradeOutcome]
        assert 'PENDING' in outcomes
        assert 'WIN' in outcomes
        assert 'LOSS' in outcomes
        assert 'STOPPED' in outcomes


class TestPrediction:
    """Prediction dataclass tests."""
    
    def test_create_valid_prediction(self):
        """Should be able to create a valid prediction."""
        pred = Prediction(
            symbol='ETH',
            direction=Direction.UP,
            confidence=0.75,
            current_price=2500.0,
            target_price=2650.0,
            stop_loss=2425.0,
            timestamp=datetime.now(),
        )
        assert pred.symbol == 'ETH'
        assert pred.confidence == 0.75
    
    def test_prediction_is_immutable(self):
        """Prediction should be immutable (frozen)."""
        pred = Prediction(
            symbol='ETH',
            direction=Direction.UP,
            confidence=0.75,
            current_price=2500.0,
            target_price=2650.0,
            stop_loss=2425.0,
            timestamp=datetime.now(),
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            pred.symbol = 'BTC'
    
    def test_prediction_invalid_confidence(self):
        """Confidence outside 0-1 should raise error."""
        with pytest.raises(ValueError):
            Prediction(
                symbol='ETH',
                direction=Direction.UP,
                confidence=1.5,  # Invalid!
                current_price=2500.0,
                target_price=2650.0,
                stop_loss=2425.0,
                timestamp=datetime.now(),
            )
    
    def test_prediction_invalid_price(self):
        """Negative price should raise error."""
        with pytest.raises(ValueError):
            Prediction(
                symbol='ETH',
                direction=Direction.UP,
                confidence=0.75,
                current_price=-100.0,  # Invalid!
                target_price=2650.0,
                stop_loss=2425.0,
                timestamp=datetime.now(),
            )


class TestScoredPrediction:
    """ScoredPrediction dataclass tests."""
    
    def test_expected_return_pct(self):
        """expected_return_pct should calculate correctly."""
        pred = ScoredPrediction(
            symbol='ETH',
            direction=Direction.UP,
            confidence=0.75,
            current_price=2500.0,
            target_price=2625.0,  # +5%
            stop_loss=2425.0,
            hold_hours=72,
            timestamp=datetime.now(),
            strategy='ghost_inverse',
            original_direction=Direction.DOWN,
            is_inverse=True,
            backtest_win_rate=0.615,
            score=0.46,
        )
        assert abs(pred.expected_return_pct - 5.0) < 0.1
    
    def test_hold_days(self):
        """hold_days should convert hours to days."""
        pred = ScoredPrediction(
            symbol='XRP',
            direction=Direction.UP,
            confidence=0.75,
            current_price=1.65,
            target_price=1.75,
            stop_loss=1.58,
            hold_hours=168,  # 7 days
            timestamp=datetime.now(),
            strategy='mean_reversion',
            original_direction=Direction.UP,
            is_inverse=False,
            backtest_win_rate=0.565,
            score=0.42,
        )
        assert pred.hold_days == 7


class TestPaperTrade:
    """PaperTrade dataclass tests."""
    
    def test_is_resolved(self):
        """is_resolved should return True when not PENDING."""
        trade = PaperTrade(symbol='ETH', outcome=TradeOutcome.PENDING)
        assert not trade.is_resolved
        
        trade.outcome = TradeOutcome.WIN
        assert trade.is_resolved
    
    def test_is_winner(self):
        """is_winner should return True only for WIN outcome."""
        trade = PaperTrade(symbol='ETH', outcome=TradeOutcome.WIN)
        assert trade.is_winner
        
        trade.outcome = TradeOutcome.LOSS
        assert not trade.is_winner
        
        trade.outcome = TradeOutcome.PENDING
        assert not trade.is_winner


class TestValidationResult:
    """ValidationResult dataclass tests."""
    
    def test_bool_conversion(self):
        """ValidationResult should be truthy when valid."""
        result = ValidationResult(is_valid=True)
        assert bool(result) == True
        
        result = ValidationResult(is_valid=False, reason="test")
        assert bool(result) == False
    
    def test_reason_optional_when_valid(self):
        """reason should be optional when valid."""
        result = ValidationResult(is_valid=True)
        assert result.reason is None
