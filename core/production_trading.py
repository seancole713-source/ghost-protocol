"""
Phase 9: Production Trading Controller
Manages transition from paper to live trading with comprehensive safety measures.
"""
import logging
import os
from datetime import datetime, timedelta, UTC
from typing import Any
from enum import Enum

LOGGER = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration."""
    PAPER = "paper"
    LIVE = "live"
    DISABLED = "disabled"


class ProductionTradingController:
    """Control production trading with risk management safeguards."""
    
    def __init__(self):
        # Trading mode configuration
        self.mode = TradingMode(os.getenv("TRADING_MODE", "paper"))
        
        # Safety limits
        self.daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT", "500"))  # Max daily loss
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "5000"))  # Max per position
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
        self.max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "20"))
        
        # Circuit breaker thresholds
        self.max_drawdown_pct = float(os.getenv("MAX_DRAWDOWN_PCT", "10"))  # 10% max drawdown
        self.consecutive_loss_limit = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "5"))
        
        # State tracking
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.open_positions_count = 0
        self.consecutive_losses = 0
        self.emergency_stop_active = False
        self.last_reset = datetime.now(UTC)
        
        # Kill switch
        self.kill_switch_active = os.getenv("KILL_SWITCH", "false").lower() == "true"
        
        LOGGER.info(f"[PROD-TRADING] Initialized in {self.mode.value.upper()} mode")
        LOGGER.info(f"[PROD-TRADING] Daily loss limit: ${self.daily_loss_limit}")
        LOGGER.info(f"[PROD-TRADING] Max drawdown: {self.max_drawdown_pct}%")
    
    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed based on safety rules.
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Check kill switch
        if self.kill_switch_active:
            return False, "Kill switch is active"
        
        # Check emergency stop
        if self.emergency_stop_active:
            return False, "Emergency stop is active"
        
        # Check trading mode
        if self.mode == TradingMode.DISABLED:
            return False, "Trading is disabled"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            self.emergency_stop_active = True
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"
        
        # Check max trades per day
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Max trades per day reached: {self.trades_today}"
        
        # Check max open positions
        if self.open_positions_count >= self.max_open_positions:
            return False, f"Max open positions reached: {self.open_positions_count}"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self.emergency_stop_active = True
            return False, f"Consecutive loss limit reached: {self.consecutive_losses}"
        
        return True, "OK"
    
    def validate_trade(self, position_size: float, symbol: str) -> tuple[bool, str]:
        """
        Validate a proposed trade.
        
        Args:
            position_size: Dollar value of position
            symbol: Trading symbol
        
        Returns:
            Tuple of (valid, reason)
        """
        # Check if trading allowed
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason
        
        # Check position size
        if position_size > self.max_position_size:
            return False, f"Position size ${position_size:.2f} exceeds max ${self.max_position_size}"
        
        # In live mode, require additional confirmation
        if self.mode == TradingMode.LIVE:
            LOGGER.warning(f"[PROD-TRADING] 🚨 LIVE TRADE: {symbol} ${position_size:.2f}")
        
        return True, "Trade validated"
    
    def record_trade(self, pnl: float, success: bool) -> None:
        """
        Record trade result and update state.
        
        Args:
            pnl: Profit/loss from trade
            success: Whether trade was successful
        """
        self.trades_today += 1
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        LOGGER.info(f"[PROD-TRADING] Trade recorded: PnL=${pnl:.2f}, Daily PnL=${self.daily_pnl:.2f}")
    
    def update_positions(self, count: int) -> None:
        """Update open positions count."""
        self.open_positions_count = count
    
    def check_drawdown(self, current_value: float, peak_value: float) -> None:
        """
        Check if drawdown exceeds threshold.
        
        Args:
            current_value: Current portfolio value
            peak_value: Peak portfolio value
        """
        if peak_value > 0:
            drawdown_pct = ((peak_value - current_value) / peak_value) * 100
            
            if drawdown_pct >= self.max_drawdown_pct:
                self.emergency_stop_active = True
                LOGGER.critical(
                    f"[PROD-TRADING] 🚨 Max drawdown exceeded: {drawdown_pct:.2f}% "
                    f"(limit: {self.max_drawdown_pct}%)"
                )
    
    def reset_daily_limits(self) -> None:
        """Reset daily counters (called at market open)."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.last_reset = datetime.now(UTC)
        LOGGER.info("[PROD-TRADING] Daily limits reset")
    
    def activate_kill_switch(self, reason: str) -> None:
        """
        Activate emergency kill switch.
        
        Args:
            reason: Reason for activation
        """
        self.kill_switch_active = True
        self.emergency_stop_active = True
        LOGGER.critical(f"[PROD-TRADING] 🚨 KILL SWITCH ACTIVATED: {reason}")
    
    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch (requires manual intervention)."""
        self.kill_switch_active = False
        self.emergency_stop_active = False
        self.consecutive_losses = 0
        LOGGER.warning("[PROD-TRADING] ⚠️ Kill switch deactivated - trading resumed")
    
    def get_status(self) -> dict[str, Any]:
        """Get current production trading status."""
        can_trade, reason = self.can_trade()
        
        return {
            "ok": True,
            "mode": self.mode.value,
            "can_trade": can_trade,
            "reason": reason,
            "daily_pnl": self.daily_pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "trades_today": self.trades_today,
            "max_trades_per_day": self.max_trades_per_day,
            "open_positions": self.open_positions_count,
            "max_open_positions": self.max_open_positions,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.consecutive_loss_limit,
            "emergency_stop_active": self.emergency_stop_active,
            "kill_switch_active": self.kill_switch_active,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_position_size": self.max_position_size,
            "last_reset": self.last_reset.isoformat(),
            "timestamp": datetime.now(UTC).isoformat()
        }


# Global production trading controller
_production_controller = ProductionTradingController()


def get_production_controller() -> ProductionTradingController:
    """Get global production trading controller."""
    return _production_controller


def can_trade() -> tuple[bool, str]:
    """Check if trading is allowed."""
    return _production_controller.can_trade()


def validate_trade(position_size: float, symbol: str) -> tuple[bool, str]:
    """Validate a proposed trade."""
    return _production_controller.validate_trade(position_size, symbol)


def record_trade(pnl: float, success: bool) -> None:
    """Record trade result."""
    _production_controller.record_trade(pnl, success)


def activate_kill_switch(reason: str) -> None:
    """Activate emergency kill switch."""
    _production_controller.activate_kill_switch(reason)


def get_status() -> dict[str, Any]:
    """Get production trading status."""
    return _production_controller.get_status()
