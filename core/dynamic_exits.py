"""
Dynamic Exit Management for Ghost Protocol
Implements trailing stops and early exits
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DynamicExitManager:
    """Manage dynamic exits for predictions"""
    
    def __init__(self):
        self.enabled = os.getenv("DYNAMIC_EXITS_ENABLED", "1") == "1"
        
        # Exit parameters
        self.profit_target_pct = float(os.getenv("EXIT_PROFIT_TARGET_PCT", "5.0"))
        self.stop_loss_pct = float(os.getenv("EXIT_STOP_LOSS_PCT", "3.0"))
        self.trailing_stop_pct = float(os.getenv("EXIT_TRAILING_STOP_PCT", "2.0"))
        self.max_hold_hours = int(os.getenv("EXIT_MAX_HOLD_HOURS", "48"))
        
        # Early exit thresholds
        self.early_exit_profit_pct = float(os.getenv("EARLY_EXIT_PROFIT_PCT", "3.0"))
        self.early_exit_hours = int(os.getenv("EARLY_EXIT_HOURS", "12"))
    
    def calculate_exit_levels(
        self,
        entry_price: float,
        direction: str,
        confidence: float
    ) -> Dict:
        """
        Calculate dynamic exit levels based on entry and direction
        
        Args:
            entry_price: Entry price
            direction: "UP" or "DOWN"
            confidence: Prediction confidence (0-1)
            
        Returns:
            Dict with target, stop_loss, trailing_stop levels
        """
        # Adjust targets based on confidence
        confidence_multiplier = 0.8 + (confidence * 0.4)  # 0.8x to 1.2x
        
        target_pct = self.profit_target_pct * confidence_multiplier
        stop_pct = self.stop_loss_pct / confidence_multiplier  # Tighter stop for low confidence
        
        if direction == "UP":
            target_price = entry_price * (1 + target_pct / 100)
            stop_loss = entry_price * (1 - stop_pct / 100)
            trailing_activation = entry_price * (1 + self.early_exit_profit_pct / 100)
        else:  # DOWN
            target_price = entry_price * (1 - target_pct / 100)
            stop_loss = entry_price * (1 + stop_pct / 100)
            trailing_activation = entry_price * (1 - self.early_exit_profit_pct / 100)
        
        return {
            "entry_price": entry_price,
            "direction": direction,
            "target_price": round(target_price, 6),
            "stop_loss": round(stop_loss, 6),
            "target_pct": round(target_pct, 2),
            "stop_loss_pct": round(stop_pct, 2),
            "trailing_stop_pct": self.trailing_stop_pct,
            "trailing_activation_price": round(trailing_activation, 6),
            "max_hold_hours": self.max_hold_hours
        }
    
    def check_exit_condition(
        self,
        entry_price: float,
        current_price: float,
        high_since_entry: float,
        low_since_entry: float,
        direction: str,
        hours_held: float,
        exit_levels: Dict
    ) -> Dict:
        """
        Check if exit condition is met
        
        Returns:
            Dict with should_exit, reason, exit_type
        """
        target = exit_levels["target_price"]
        stop = exit_levels["stop_loss"]
        trailing_pct = exit_levels["trailing_stop_pct"]
        trailing_activation = exit_levels["trailing_activation_price"]
        
        # Calculate current P&L
        if not entry_price or entry_price <= 0:
            return {"should_exit": False, "reason": "Invalid entry_price", "exit_type": None}
        
        if direction == "UP":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            hit_target = current_price >= target
            hit_stop = current_price <= stop
            
            # Trailing stop calculation
            trailing_active = high_since_entry >= trailing_activation
            if trailing_active:
                trailing_stop = high_since_entry * (1 - trailing_pct / 100)
                hit_trailing = current_price <= trailing_stop
            else:
                trailing_stop = None
                hit_trailing = False
                
        else:  # DOWN
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
            hit_target = current_price <= target
            hit_stop = current_price >= stop
            
            # Trailing stop for short
            trailing_active = low_since_entry <= trailing_activation
            if trailing_active:
                trailing_stop = low_since_entry * (1 + trailing_pct / 100)
                hit_trailing = current_price >= trailing_stop
            else:
                trailing_stop = None
                hit_trailing = False
        
        # Check time-based exit
        time_expired = hours_held >= self.max_hold_hours
        
        # Early exit with profit
        early_exit_eligible = (
            hours_held >= self.early_exit_hours and 
            pnl_pct >= self.early_exit_profit_pct
        )
        
        # Determine exit
        if hit_target:
            return {
                "should_exit": True,
                "exit_type": "TARGET_HIT",
                "reason": f"Price hit target ${target:.4f}",
                "pnl_pct": round(pnl_pct, 2)
            }
        elif hit_stop:
            return {
                "should_exit": True,
                "exit_type": "STOP_LOSS",
                "reason": f"Price hit stop loss ${stop:.4f}",
                "pnl_pct": round(pnl_pct, 2)
            }
        elif hit_trailing:
            return {
                "should_exit": True,
                "exit_type": "TRAILING_STOP",
                "reason": f"Trailing stop triggered at ${trailing_stop:.4f}",
                "pnl_pct": round(pnl_pct, 2)
            }
        elif early_exit_eligible:
            return {
                "should_exit": True,
                "exit_type": "EARLY_PROFIT",
                "reason": f"Early exit with {pnl_pct:.1f}% profit after {hours_held:.0f}h",
                "pnl_pct": round(pnl_pct, 2)
            }
        elif time_expired:
            return {
                "should_exit": True,
                "exit_type": "TIME_EXPIRED",
                "reason": f"Max hold time of {self.max_hold_hours}h reached",
                "pnl_pct": round(pnl_pct, 2)
            }
        else:
            return {
                "should_exit": False,
                "exit_type": None,
                "reason": "No exit condition met",
                "pnl_pct": round(pnl_pct, 2),
                "trailing_active": trailing_active,
                "trailing_stop": trailing_stop,
                "hours_remaining": round(self.max_hold_hours - hours_held, 1)
            }


# Singleton
_exit_manager: Optional[DynamicExitManager] = None


def get_exit_manager() -> DynamicExitManager:
    """Get or create exit manager singleton"""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = DynamicExitManager()
    return _exit_manager


def calculate_exits(entry_price: float, direction: str, confidence: float) -> Dict:
    """Calculate exit levels for a new prediction"""
    return get_exit_manager().calculate_exit_levels(entry_price, direction, confidence)


def check_exit(
    entry_price: float,
    current_price: float,
    high_since_entry: float,
    low_since_entry: float,
    direction: str,
    hours_held: float,
    exit_levels: Dict
) -> Dict:
    """Check if exit condition is met"""
    return get_exit_manager().check_exit_condition(
        entry_price, current_price, high_since_entry, low_since_entry,
        direction, hours_held, exit_levels
    )
