"""
Tests for data validators.
"""
import pytest
from data.validators import (
    PriceValidator,
    ConfidenceValidator,
    SymbolValidator,
    validate_prediction_data,
    price_validator,
)


class TestPriceValidator:
    """Price validation tests."""
    
    def test_valid_btc_price(self):
        """Valid BTC price should pass."""
        result = price_validator.validate('BTC', 75000.0)
        assert result.is_valid
    
    def test_btc_price_too_low(self):
        """BTC price below minimum should fail."""
        result = price_validator.validate('BTC', 5000.0)
        assert not result.is_valid
        assert 'below minimum' in result.reason
    
    def test_btc_price_too_high(self):
        """BTC price above maximum should fail."""
        result = price_validator.validate('BTC', 1000000.0)
        assert not result.is_valid
        assert 'above maximum' in result.reason
    
    def test_negative_price_fails(self):
        """Negative price should fail."""
        result = price_validator.validate('BTC', -100.0)
        assert not result.is_valid
        assert 'positive' in result.reason
    
    def test_zero_price_fails(self):
        """Zero price should fail."""
        result = price_validator.validate('ETH', 0.0)
        assert not result.is_valid
    
    def test_none_price_fails(self):
        """None price should fail."""
        result = price_validator.validate('ETH', None)
        assert not result.is_valid
    
    def test_unknown_symbol_uses_default_range(self):
        """Unknown symbols should use permissive default range."""
        result = price_validator.validate('UNKNOWN', 100.0)
        assert result.is_valid
    
    def test_validate_prediction_valid(self):
        """Valid prediction prices should pass."""
        result = price_validator.validate_prediction(
            'ETH', 
            current=2500.0, 
            target=2650.0, 
            stop=2425.0
        )
        assert result.is_valid
    
    def test_validate_prediction_target_too_close(self):
        """Target too close to current should fail."""
        result = price_validator.validate_prediction(
            'ETH',
            current=2500.0,
            target=2500.5,  # Only 0.02% difference
            stop=2425.0
        )
        assert not result.is_valid
        assert 'too close' in result.reason
    
    def test_validate_prediction_stop_wrong_side_for_buy(self):
        """BUY signal with stop above current should fail."""
        result = price_validator.validate_prediction(
            'ETH',
            current=2500.0,
            target=2650.0,  # Target above = BUY
            stop=2550.0     # Stop above current - wrong!
        )
        assert not result.is_valid
        assert 'above current' in result.reason


class TestConfidenceValidator:
    """Confidence validation tests."""
    
    def test_valid_confidence(self):
        """Valid confidence should pass."""
        validator = ConfidenceValidator()
        assert validator.validate(0.75).is_valid
        assert validator.validate(0.0).is_valid
        assert validator.validate(1.0).is_valid
    
    def test_confidence_too_high(self):
        """Confidence above 1 should fail."""
        validator = ConfidenceValidator()
        result = validator.validate(1.5)
        assert not result.is_valid
    
    def test_confidence_negative(self):
        """Negative confidence should fail."""
        validator = ConfidenceValidator()
        result = validator.validate(-0.1)
        assert not result.is_valid
    
    def test_confidence_none(self):
        """None confidence should fail."""
        validator = ConfidenceValidator()
        result = validator.validate(None)
        assert not result.is_valid


class TestSymbolValidator:
    """Symbol validation tests."""
    
    def test_valid_symbols(self):
        """Valid symbols should pass."""
        validator = SymbolValidator()
        assert validator.validate('ETH').is_valid
        assert validator.validate('BTC').is_valid
        assert validator.validate('AAPL').is_valid
    
    def test_empty_symbol(self):
        """Empty symbol should fail."""
        validator = SymbolValidator()
        result = validator.validate('')
        assert not result.is_valid
    
    def test_symbol_with_spaces(self):
        """Symbol with spaces should fail."""
        validator = SymbolValidator()
        result = validator.validate('BT C')
        assert not result.is_valid
    
    def test_symbol_too_long(self):
        """Very long symbol should fail."""
        validator = SymbolValidator()
        result = validator.validate('VERYLONGSYMBOL')
        assert not result.is_valid


class TestValidatePredictionData:
    """Comprehensive prediction validation tests."""
    
    def test_valid_prediction(self):
        """Fully valid prediction should pass."""
        result = validate_prediction_data(
            symbol='ETH',
            direction='UP',
            confidence=0.75,
            current_price=2500.0,
            target_price=2650.0,
            stop_loss=2425.0,
        )
        assert result.is_valid
    
    def test_invalid_direction(self):
        """Invalid direction should fail."""
        result = validate_prediction_data(
            symbol='ETH',
            direction='SIDEWAYS',
            confidence=0.75,
            current_price=2500.0,
            target_price=2650.0,
            stop_loss=2425.0,
        )
        assert not result.is_valid
        assert 'direction' in result.reason.lower()
