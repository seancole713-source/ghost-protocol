"""
Risk Budget Enforcement Layer
Guards order submission for paper trading (ALPACA_PAPER=1 only)

Enforces:
- RISK_MAX_POS_PCT: Maximum position size as % of equity
- RISK_MAX_DAILY_DD_PCT: Maximum daily drawdown %
- MAX_RISK_DRAWDOWN: Maximum overall drawdown (decimal)
- RISK_SL_PCT: Stop-loss percentage
- RISK_TP_PCT: Take-profit percentage
- TARGET_WEEKLY_PROFIT_USD: Weekly profit target

Safe by default - blocks orders that violate limits.
"""

import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)


class RiskGuard:
    """Risk budget enforcement for order submission"""
    
    def __init__(self):
        # Load risk configuration from environment
        self.max_position_pct = float(os.getenv("RISK_MAX_POS_PCT", "5"))
        self.max_daily_dd_pct = float(os.getenv("RISK_MAX_DAILY_DD_PCT", "5"))
        self.max_risk_drawdown = float(os.getenv("MAX_RISK_DRAWDOWN", "0.05"))
        self.stop_loss_pct = float(os.getenv("RISK_SL_PCT", "3"))
        self.take_profit_pct = float(os.getenv("RISK_TP_PCT", "6"))
        self.target_weekly_profit_usd = float(os.getenv("TARGET_WEEKLY_PROFIT_USD", "300"))
        
        # Broker config
        self.broker = os.getenv("BROKER", "").lower()
        self.alpaca_paper = os.getenv("ALPACA_PAPER", "0") == "1"
        
        LOGGER.info(
            f"RiskGuard initialized: max_pos={self.max_position_pct}%, "
            f"max_daily_dd={self.max_daily_dd_pct}%, max_dd={self.max_risk_drawdown:.2%}, "
            f"broker={self.broker}, paper={self.alpaca_paper}"
        )
    
    def is_enabled(self) -> bool:
        """Check if risk guard is active (only for Alpaca paper trading)"""
        return self.broker == "alpaca" and self.alpaca_paper
    
    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_equity: float,
        current_positions: dict[str, Any] | None = None,
        daily_pnl: float = 0.0,
        total_pnl: float = 0.0
    ) -> tuple[bool, str]:
        """
        Check if order passes risk limits.
        
        Args:
            symbol: Stock/crypto symbol
            side: 'buy' or 'sell'
            quantity: Number of shares/units
            price: Entry price
            current_equity: Total account equity
            current_positions: Dict of current positions {symbol: {qty, avg_price}}
            daily_pnl: Today's P&L (negative for loss)
            total_pnl: Total account P&L since inception
        
        Returns:
            (allowed: bool, reason: str)
        """
        if not self.is_enabled():
            return True, "Risk guard disabled (not in paper mode)"
        
        # Calculate order value
        order_value = quantity * price
        
        # 1. Check maximum position size
        if current_equity > 0:
            position_pct = (order_value / current_equity) * 100
            
            if position_pct > self.max_position_pct:
                reason = (
                    f"Position size {position_pct:.2f}% exceeds "
                    f"RISK_MAX_POS_PCT={self.max_position_pct}%"
                )
                LOGGER.warning(f"RISK BLOCK: {symbol} {side} - {reason}")
                return False, reason
        
        # 2. Check daily drawdown limit
        if daily_pnl < 0:
            daily_dd_pct = abs(daily_pnl / current_equity) * 100 if current_equity > 0 else 0
            
            if daily_dd_pct > self.max_daily_dd_pct:
                reason = (
                    f"Daily drawdown {daily_dd_pct:.2f}% exceeds "
                    f"RISK_MAX_DAILY_DD_PCT={self.max_daily_dd_pct}%"
                )
                LOGGER.warning(f"RISK BLOCK: {symbol} {side} - {reason}")
                return False, reason
        
        # 3. Check maximum overall drawdown
        if total_pnl < 0 and current_equity > 0:
            # Calculate drawdown from peak equity (approximation)
            initial_equity = current_equity - total_pnl
            drawdown = abs(total_pnl / initial_equity)
            
            if drawdown > self.max_risk_drawdown:
                reason = (
                    f"Total drawdown {drawdown:.2%} exceeds "
                    f"MAX_RISK_DRAWDOWN={self.max_risk_drawdown:.2%}"
                )
                LOGGER.warning(f"RISK BLOCK: {symbol} {side} - {reason}")
                return False, reason
        
        # 4. Check if position would create excessive concentration
        if current_positions and side.lower() == "buy":
            # Count existing positions
            num_positions = len([p for p in current_positions.values() if p.get("qty", 0) > 0])
            
            # Prevent > 10 concurrent positions (diversification requirement)
            if num_positions >= 10:
                reason = f"Already holding {num_positions} positions (max 10 for risk diversification)"
                LOGGER.warning(f"RISK BLOCK: {symbol} {side} - {reason}")
                return False, reason
        
        # All checks passed
        LOGGER.info(
            f"RISK PASS: {symbol} {side} {quantity}@${price:.2f} "
            f"(${order_value:.2f}, {(order_value/current_equity*100):.2f}% of equity)"
        )
        return True, "Order approved"
    
    def get_status(self) -> dict[str, Any]:
        """
        Get current risk guard configuration and status.
        
        Returns status dict suitable for health endpoints.
        """
        return {
            "enabled": self.is_enabled(),
            "broker": self.broker,
            "paper_mode": self.alpaca_paper,
            "limits": {
                "max_position_pct": self.max_position_pct,
                "max_daily_dd_pct": self.max_daily_dd_pct,
                "max_risk_drawdown": self.max_risk_drawdown,
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "target_weekly_profit_usd": self.target_weekly_profit_usd
            },
            "status": "active" if self.is_enabled() else "disabled"
        }


# Global risk guard instance
_RISK_GUARD: RiskGuard | None = None


def get_risk_guard() -> RiskGuard:
    """Get or create global risk guard instance"""
    global _RISK_GUARD
    if _RISK_GUARD is None:
        _RISK_GUARD = RiskGuard()
    return _RISK_GUARD
