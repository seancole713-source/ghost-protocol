"""
Data quality validation.

Prevents corrupted or unreasonable data from entering the system.
All validation logic centralized here.
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from loguru import logger

from config.settings import settings
from core.models import ValidationResult


class PriceValidator:
    """
    Validate price data quality.
    
    Ensures prices are within reasonable historical ranges
    to catch API errors or data corruption.
    """
    
    # Reasonable price ranges by symbol
    # Updated periodically based on market conditions
    PRICE_RANGES: Dict[str, Tuple[float, float]] = {
        'BTC': (settings.MIN_PRICE_BTC, settings.MAX_PRICE_BTC),
        'ETH': (settings.MIN_PRICE_ETH, settings.MAX_PRICE_ETH),
        'XRP': (settings.MIN_PRICE_XRP, settings.MAX_PRICE_XRP),
        'LINK': (settings.MIN_PRICE_LINK, settings.MAX_PRICE_LINK),
        'SOL': (settings.MIN_PRICE_SOL, settings.MAX_PRICE_SOL),
    }
    
    # Default range for unknown symbols (very permissive)
    DEFAULT_RANGE = (0.0001, 1_000_000.0)
    
    def get_range(self, symbol: str) -> Tuple[float, float]:
        """Get the valid price range for a symbol."""
        return self.PRICE_RANGES.get(symbol.upper(), self.DEFAULT_RANGE)
    
    def validate(self, symbol: str, price: float) -> ValidationResult:
        """
        Check if price is within reasonable range.
        
        Args:
            symbol: The trading symbol
            price: The price to validate
            
        Returns:
            ValidationResult with is_valid and optional reason
        """
        if price is None:
            return ValidationResult(False, "Price is None")
        
        if price <= 0:
            return ValidationResult(False, f"Price must be positive: {price}")
        
        low, high = self.get_range(symbol)
        
        if price < low:
            return ValidationResult(
                False,
                f"{symbol} price ${price:,.2f} below minimum ${low:,.2f}"
            )
        
        if price > high:
            return ValidationResult(
                False,
                f"{symbol} price ${price:,.2f} above maximum ${high:,.2f}"
            )
        
        return ValidationResult(True)
    
    def validate_prediction(
        self, 
        symbol: str, 
        current: float, 
        target: float, 
        stop: float
    ) -> ValidationResult:
        """
        Validate a complete prediction's prices.
        
        Checks:
        1. Current price is valid
        2. Target is sufficiently different from current
        3. Stop is sufficiently different from current
        4. Target and stop are on correct sides for direction
        
        Args:
            symbol: Trading symbol
            current: Current price
            target: Target price
            stop: Stop loss price
            
        Returns:
            ValidationResult
        """
        # Validate current price
        result = self.validate(symbol, current)
        if not result.is_valid:
            return result
        
        # Target must be meaningfully different from current (0.1% minimum)
        if abs(target - current) / current < 0.001:
            return ValidationResult(
                False, 
                f"Target ${target:,.2f} too close to current ${current:,.2f}"
            )
        
        # Stop must be meaningfully different from current
        if abs(stop - current) / current < 0.001:
            return ValidationResult(
                False, 
                f"Stop ${stop:,.2f} too close to current ${current:,.2f}"
            )
        
        # For BUY (target > current): stop should be below current
        if target > current and stop > current:
            return ValidationResult(
                False,
                f"BUY signal but stop ${stop:,.2f} above current ${current:,.2f}"
            )
        
        # For SELL (target < current): stop should be above current
        if target < current and stop < current:
            return ValidationResult(
                False,
                f"SELL signal but stop ${stop:,.2f} below current ${current:,.2f}"
            )
        
        return ValidationResult(True)


class ConfidenceValidator:
    """Validate confidence scores."""
    
    def validate(self, confidence: float) -> ValidationResult:
        """Check if confidence is in valid range [0, 1]."""
        if confidence is None:
            return ValidationResult(False, "Confidence is None")
        
        if not 0 <= confidence <= 1:
            return ValidationResult(
                False,
                f"Confidence must be 0-1, got {confidence}"
            )
        
        return ValidationResult(True)


class SymbolValidator:
    """Validate trading symbols."""
    
    # Characters that should never appear in a symbol
    INVALID_CHARS = set(' \t\n\r!@#$%^&*()+=[]{}|\\:;"\'<>,?/')
    
    def validate(self, symbol: str) -> ValidationResult:
        """Check if symbol is valid format."""
        if not symbol:
            return ValidationResult(False, "Symbol is empty")
        
        if len(symbol) > 10:
            return ValidationResult(False, f"Symbol too long: {symbol}")
        
        if any(c in self.INVALID_CHARS for c in symbol):
            return ValidationResult(False, f"Symbol has invalid characters: {symbol}")
        
        return ValidationResult(True)


# Singleton instances
price_validator = PriceValidator()
confidence_validator = ConfidenceValidator()
symbol_validator = SymbolValidator()


def validate_prediction_data(
    symbol: str,
    direction: str,
    confidence: float,
    current_price: float,
    target_price: float,
    stop_loss: float,
) -> ValidationResult:
    """
    Comprehensive validation of all prediction data.
    
    Use this as a single entry point for validation.
    """
    # Validate symbol
    result = symbol_validator.validate(symbol)
    if not result:
        return result
    
    # Validate confidence
    result = confidence_validator.validate(confidence)
    if not result:
        return result
    
    # Validate direction
    if direction not in ('UP', 'DOWN'):
        return ValidationResult(False, f"Invalid direction: {direction}")
    
    # Validate prices
    result = price_validator.validate_prediction(
        symbol, current_price, target_price, stop_loss
    )
    if not result:
        return result
    
    return ValidationResult(True)
