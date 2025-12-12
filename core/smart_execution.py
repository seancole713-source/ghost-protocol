"""
🎯 SMART EXECUTION ENGINE
Limit order ladders, TWAP/VWAP execution, trail stops, profit-taking scales
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

# Execution strategies
ExecutionStrategy = Literal["MARKET", "LIMIT_LADDER", "TWAP", "VWAP", "ICEBERG"]


# ============================================================================
# LIMIT ORDER LADDER
# ============================================================================

def calculate_limit_ladder(
    direction: str,
    entry_price: float,
    total_shares: int,
    num_orders: int = 5,
    price_spread: float = 0.5
) -> list[dict]:
    """
    Create limit order ladder (buy/sell in increments)
    """
    try:
        orders = []
        shares_per_order = total_shares // num_orders
        
        if direction == "LONG":
            # Buy ladder: place orders below current price
            for i in range(num_orders):
                limit_price = entry_price * (1 - (price_spread / 100) * (i + 1) / num_orders)
                
                orders.append({
                    "order_id": f"BUY_{i+1}",
                    "side": "BUY",
                    "shares": shares_per_order,
                    "limit_price": limit_price,
                    "status": "PENDING"
                })
        
        else:  # SHORT
            # Sell ladder: place orders above current price
            for i in range(num_orders):
                limit_price = entry_price * (1 + (price_spread / 100) * (i + 1) / num_orders)
                
                orders.append({
                    "order_id": f"SELL_{i+1}",
                    "side": "SELL",
                    "shares": shares_per_order,
                    "limit_price": limit_price,
                    "status": "PENDING"
                })
        
        return orders
        
    except Exception as e:
        LOGGER.error(f"Limit ladder calculation failed: {e}")
        return []


# ============================================================================
# TWAP (Time-Weighted Average Price)
# ============================================================================

def calculate_twap_schedule(
    total_shares: int,
    duration_minutes: int = 60,
    interval_minutes: int = 5
) -> list[dict]:
    """
    Calculate TWAP execution schedule (even distribution over time)
    """
    try:
        num_intervals = duration_minutes // interval_minutes
        shares_per_interval = total_shares // num_intervals
        
        schedule = []
        
        for i in range(num_intervals):
            schedule.append({
                "interval": i + 1,
                "time_offset_minutes": i * interval_minutes,
                "shares": shares_per_interval,
                "execution_type": "MARKET"
            })
        
        # Add remainder to last interval
        remainder = total_shares - (shares_per_interval * num_intervals)
        if remainder > 0:
            schedule[-1]["shares"] += remainder
        
        return schedule
        
    except Exception as e:
        LOGGER.error(f"TWAP schedule calculation failed: {e}")
        return []


# ============================================================================
# VWAP (Volume-Weighted Average Price)
# ============================================================================

def calculate_vwap_schedule(
    total_shares: int,
    historical_volume_profile: list[dict]
) -> list[dict]:
    """
    Calculate VWAP execution schedule (weighted by volume)
    """
    try:
        if not historical_volume_profile:
            # Fallback to TWAP
            return calculate_twap_schedule(total_shares)
        
        # Calculate total volume
        total_volume = sum(bar["volume"] for bar in historical_volume_profile)
        
        schedule = []
        
        for i, bar in enumerate(historical_volume_profile):
            volume_pct = bar["volume"] / total_volume
            shares_for_bar = int(total_shares * volume_pct)
            
            schedule.append({
                "interval": i + 1,
                "time": bar["time"],
                "shares": shares_for_bar,
                "volume_pct": volume_pct * 100
            })
        
        return schedule
        
    except Exception as e:
        LOGGER.error(f"VWAP schedule calculation failed: {e}")
        return []


# ============================================================================
# TRAIL STOP MANAGEMENT
# ============================================================================

class TrailStopManager:
    """
    Dynamic trail stop that tightens as profit increases
    """
    
    def __init__(
        self,
        entry_price: float,
        initial_stop_pct: float = 6.0,
        profit_threshold_pct: float = 10.0,
        tight_stop_pct: float = 8.0
    ):
        self.entry_price = entry_price
        self.initial_stop_pct = initial_stop_pct
        self.profit_threshold_pct = profit_threshold_pct
        self.tight_stop_pct = tight_stop_pct
        self.highest_price = entry_price
        self.current_stop = entry_price * (1 - initial_stop_pct / 100)
    
    def update(self, current_price: float) -> dict:
        """
        Update trail stop based on current price
        """
        try:
            # Update highest price
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Calculate profit %
            profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            
            # Tighten stop if profit > threshold
            if profit_pct >= self.profit_threshold_pct:
                # Trail stop 8% below highest price
                new_stop = self.highest_price * (1 - self.tight_stop_pct / 100)
            else:
                # Fixed stop below entry
                new_stop = self.entry_price * (1 - self.initial_stop_pct / 100)
            
            # Only move stop up, never down
            if new_stop > self.current_stop:
                self.current_stop = new_stop
            
            return {
                "current_stop": self.current_stop,
                "highest_price": self.highest_price,
                "profit_pct": profit_pct,
                "stop_hit": current_price <= self.current_stop
            }
            
        except Exception as e:
            LOGGER.error(f"Trail stop update failed: {e}")
            return {"current_stop": self.current_stop, "stop_hit": False}


# ============================================================================
# PROFIT-TAKING SCALE
# ============================================================================

def calculate_profit_scale(
    entry_price: float,
    total_shares: int,
    target_gain_pct: float = 15.0
) -> list[dict]:
    """
    Calculate profit-taking scale (sell in stages)
    """
    try:
        scales = [
            {"pct_gain": target_gain_pct * 0.5, "shares_pct": 33, "label": "First Target"},
            {"pct_gain": target_gain_pct * 0.75, "shares_pct": 33, "label": "Second Target"},
            {"pct_gain": target_gain_pct * 1.0, "shares_pct": 34, "label": "Final Target"}
        ]
        
        profit_orders = []
        
        for scale in scales:
            profit_price = entry_price * (1 + scale["pct_gain"] / 100)
            shares = int(total_shares * scale["shares_pct"] / 100)
            
            profit_orders.append({
                "label": scale["label"],
                "target_price": profit_price,
                "shares": shares,
                "gain_pct": scale["pct_gain"],
                "status": "PENDING"
            })
        
        return profit_orders
        
    except Exception as e:
        LOGGER.error(f"Profit scale calculation failed: {e}")
        return []


# ============================================================================
# EXECUTION OPTIMIZER
# ============================================================================

def optimize_execution_strategy(
    direction: str,
    total_shares: int,
    entry_price: float,
    urgency: str = "NORMAL",
    market_condition: str = "NEUTRAL"
) -> dict:
    """
    Recommend optimal execution strategy
    """
    try:
        # HIGH urgency = market orders
        if urgency == "HIGH":
            return {
                "strategy": "MARKET",
                "reason": "High urgency - execute immediately",
                "orders": [{
                    "type": "MARKET",
                    "shares": total_shares,
                    "side": "BUY" if direction == "LONG" else "SELL"
                }]
            }
        
        # VOLATILE market = limit ladder
        if market_condition == "VOLATILE":
            orders = calculate_limit_ladder(direction, entry_price, total_shares)
            return {
                "strategy": "LIMIT_LADDER",
                "reason": "Volatile market - use limit ladder for better fills",
                "orders": orders
            }
        
        # Large position = TWAP
        if total_shares > 1000:
            schedule = calculate_twap_schedule(total_shares, duration_minutes=60)
            return {
                "strategy": "TWAP",
                "reason": "Large position - spread execution over 1 hour",
                "schedule": schedule
            }
        
        # Default = limit order near market
        return {
            "strategy": "LIMIT",
            "reason": "Normal execution - limit order 0.1% from market",
            "orders": [{
                "type": "LIMIT",
                "shares": total_shares,
                "side": "BUY" if direction == "LONG" else "SELL",
                "limit_price": entry_price * (0.999 if direction == "LONG" else 1.001)
            }]
        }
        
    except Exception as e:
        LOGGER.error(f"Execution optimizer failed: {e}")
        return {"strategy": "MARKET", "orders": []}


# ============================================================================
# EXECUTION MONITOR
# ============================================================================

class ExecutionMonitor:
    """
    Monitor execution progress and adjust strategy
    """
    
    def __init__(self):
        self.active_orders = {}
    
    def track_order(self, order_id: str, order_details: dict):
        """Add order to tracking"""
        self.active_orders[order_id] = {
            **order_details,
            "submitted_at": time.time(),
            "status": "PENDING"
        }
    
    def update_order_status(self, order_id: str, status: str, filled_shares: int = 0):
        """Update order status"""
        if order_id in self.active_orders:
            self.active_orders[order_id]["status"] = status
            self.active_orders[order_id]["filled_shares"] = filled_shares
            self.active_orders[order_id]["updated_at"] = time.time()
    
    def get_fill_rate(self) -> float:
        """Calculate % of orders filled"""
        if not self.active_orders:
            return 0.0
        
        total_shares = sum(o.get("shares", 0) for o in self.active_orders.values())
        filled_shares = sum(o.get("filled_shares", 0) for o in self.active_orders.values())
        
        return (filled_shares / total_shares * 100) if total_shares > 0 else 0.0
    
    def get_execution_summary(self) -> dict:
        """Get execution summary"""
        return {
            "total_orders": len(self.active_orders),
            "filled": len([o for o in self.active_orders.values() if o["status"] == "FILLED"]),
            "pending": len([o for o in self.active_orders.values() if o["status"] == "PENDING"]),
            "cancelled": len([o for o in self.active_orders.values() if o["status"] == "CANCELLED"]),
            "fill_rate": self.get_fill_rate()
        }
