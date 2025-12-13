#!/usr/bin/env python3
"""
Ghost Protocol - Dynamic Position Sizing & Risk Management
==========================================================
Kelly Criterion + ATR-based stops for optimal capital allocation

Protects capital during drawdowns, compounds wins faster
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing recommendation"""
    symbol: str
    shares: int
    dollar_amount: float
    risk_per_share: float
    stop_loss_price: float
    take_profit_price: float
    position_pct: float  # % of portfolio
    kelly_fraction: float
    max_loss_dollar: float
    risk_reward_ratio: float


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_capital: float
    total_risk_dollar: float
    total_risk_pct: float
    portfolio_heat: float  # % of capital at risk
    available_capital: float
    position_count: int
    correlation_risk: float


class PositionSizer:
    """Kelly Criterion + ATR position sizing"""
    
    def __init__(self, total_capital: float = 100000.0):
        self.total_capital = total_capital
        self.max_portfolio_heat = 0.20  # Max 20% of capital at risk
        self.max_position_size = 0.10  # Max 10% per position
        self.kelly_fraction = 0.25  # Use 1/4 Kelly for safety
        
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        confidence: float,
        atr: float,
        win_rate: float = 0.55,
        avg_win_pct: float = 5.0,
        avg_loss_pct: float = 2.0
    ) -> PositionSize:
        """
        Calculate optimal position size using Kelly Criterion + ATR
        
        Args:
            symbol: Ticker
            entry_price: Entry price
            confidence: Model confidence (0.0-1.0)
            atr: Average True Range (volatility measure)
            win_rate: Historical win rate
            avg_win_pct: Average winning trade %
            avg_loss_pct: Average losing trade %
        
        Returns:
            PositionSize with shares, dollar amount, stops
        """
        # Kelly Criterion: f* = (p*b - q) / b
        # Where:
        # p = win rate
        # q = 1 - p (loss rate)
        # b = avg_win / avg_loss
        
        p = win_rate
        q = 1 - p
        b = avg_win_pct / avg_loss_pct
        
        # Full Kelly
        kelly_full = (p * b - q) / b
        
        # Apply fractional Kelly (more conservative)
        kelly_pct = max(kelly_full * self.kelly_fraction, 0.01)
        kelly_pct = min(kelly_pct, self.max_position_size)
        
        # Adjust by confidence (lower confidence = smaller size)
        confidence_adjusted_pct = kelly_pct * confidence
        
        # Calculate dollar amount
        position_dollar = self.total_capital * confidence_adjusted_pct
        
        # ATR-based stop loss (2 ATR below entry)
        atr_multiplier = 2.0
        stop_loss_price = entry_price - (atr * atr_multiplier)
        risk_per_share = entry_price - stop_loss_price
        
        # Calculate shares based on risk
        # Don't risk more than 2% of capital per trade
        max_risk_dollar = self.total_capital * 0.02
        shares = int(min(
            position_dollar / entry_price,  # Kelly-based shares
            max_risk_dollar / risk_per_share  # Risk-based shares
        ))
        
        # Final dollar amount
        actual_dollar = shares * entry_price
        actual_position_pct = actual_dollar / self.total_capital
        
        # Take profit target (3x risk for 3:1 RR)
        take_profit_price = entry_price + (risk_per_share * 3)
        
        # Max loss
        max_loss_dollar = shares * risk_per_share
        
        # Risk/reward ratio
        risk_reward_ratio = (take_profit_price - entry_price) / (entry_price - stop_loss_price)
        
        logger.info(
            f"Position size for {symbol}: {shares} shares "
            f"(${actual_dollar:,.0f}, {actual_position_pct:.1%} of portfolio)"
        )
        
        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_amount=actual_dollar,
            risk_per_share=risk_per_share,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            position_pct=actual_position_pct,
            kelly_fraction=kelly_pct,
            max_loss_dollar=max_loss_dollar,
            risk_reward_ratio=risk_reward_ratio
        )
    
    def calculate_portfolio_risk(
        self,
        open_positions: list[dict[str, Any]]
    ) -> PortfolioRisk:
        """
        Calculate total portfolio risk exposure
        
        Args:
            open_positions: List of position dicts with symbol, shares, entry, stop
        
        Returns:
            PortfolioRisk with total exposure metrics
        """
        total_risk_dollar = 0.0
        total_capital_used = 0.0
        
        for position in open_positions:
            shares = position.get("shares", 0)
            entry = position.get("entry_price", 0)
            stop = position.get("stop_loss_price", entry * 0.98)
            
            risk_per_share = entry - stop
            position_risk = shares * risk_per_share
            position_value = shares * entry
            
            total_risk_dollar += position_risk
            total_capital_used += position_value
        
        portfolio_heat = total_risk_dollar / self.total_capital
        total_risk_pct = (total_risk_dollar / total_capital_used) if total_capital_used > 0 else 0
        available_capital = self.total_capital - total_capital_used
        
        # Correlation risk approximation (increases with position count)
        # Note: Full correlation matrix requires historical price data
        correlation_risk = min(len(open_positions) / 10, 1.0)
        
        return PortfolioRisk(
            total_capital=self.total_capital,
            total_risk_dollar=total_risk_dollar,
            total_risk_pct=total_risk_pct,
            portfolio_heat=portfolio_heat,
            available_capital=available_capital,
            position_count=len(open_positions),
            correlation_risk=correlation_risk
        )
    
    def can_add_position(
        self,
        new_position_risk: float,
        open_positions: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """
        Check if new position would exceed risk limits
        
        Returns:
            (can_add, reason)
        """
        portfolio_risk = self.calculate_portfolio_risk(open_positions)
        
        # Check portfolio heat
        new_total_risk = portfolio_risk.total_risk_dollar + new_position_risk
        new_heat = new_total_risk / self.total_capital
        
        if new_heat > self.max_portfolio_heat:
            return False, f"Portfolio heat {new_heat:.1%} exceeds limit {self.max_portfolio_heat:.1%}"
        
        # Check available capital
        if portfolio_risk.available_capital < new_position_risk * 10:  # Rough estimate
            return False, f"Insufficient capital (available: ${portfolio_risk.available_capital:,.0f})"
        
        return True, "OK"


# Global instance
_position_sizer = None


def get_position_sizer(capital: float = 100000.0) -> PositionSizer:
    """Get or create global position sizer"""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizer(capital)
        logger.info(f"✅ Position sizer initialized (capital: ${capital:,.0f})")
    return _position_sizer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("💰 Testing Position Sizer")
    print("=" * 60)
    
    sizer = get_position_sizer(capital=100000)
    
    # Test position sizing
    position = sizer.calculate_position_size(
        symbol="AAPL",
        entry_price=180.0,
        confidence=0.75,
        atr=3.50,
        win_rate=0.60,
        avg_win_pct=6.0,
        avg_loss_pct=2.5
    )
    
    print("\n📊 Position Sizing:")
    print(f"  Symbol: {position.symbol}")
    print(f"  Shares: {position.shares}")
    print(f"  Dollar Amount: ${position.dollar_amount:,.2f}")
    print(f"  Position %: {position.position_pct:.1%}")
    print(f"  Entry: ${position.stop_loss_price:.2f} → ${position.take_profit_price:.2f}")
    print(f"  Risk per Share: ${position.risk_per_share:.2f}")
    print(f"  Max Loss: ${position.max_loss_dollar:,.2f}")
    print(f"  Risk/Reward: {position.risk_reward_ratio:.2f}:1")
    print(f"  Kelly Fraction: {position.kelly_fraction:.1%}")
    
    # Test portfolio risk
    open_positions = [
        {"shares": 100, "entry_price": 180, "stop_loss_price": 173},
        {"shares": 50, "entry_price": 500, "stop_loss_price": 485},
    ]
    
    portfolio_risk = sizer.calculate_portfolio_risk(open_positions)
    
    print("\n📈 Portfolio Risk:")
    print(f"  Total Capital: ${portfolio_risk.total_capital:,.0f}")
    print(f"  Total Risk: ${portfolio_risk.total_risk_dollar:,.2f}")
    print(f"  Portfolio Heat: {portfolio_risk.portfolio_heat:.1%}")
    print(f"  Available Capital: ${portfolio_risk.available_capital:,.0f}")
    print(f"  Position Count: {portfolio_risk.position_count}")
    
    # Test risk limit
    can_add, reason = sizer.can_add_position(2000, open_positions)
    print(f"\n✅ Can add position: {can_add} ({reason})")
    
    print("\n✅ Position sizer test complete")
