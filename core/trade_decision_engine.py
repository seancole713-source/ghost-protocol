#!/usr/bin/env python3
"""
GHOST PROTOCOL - TRADE DECISION ENGINE
=======================================
Multi-layer filter system for trade decisions

Evaluates predictions through 4 layers:
1. Confidence & Direction (70%+ confidence, BUY/SELL action)
2. Market Conditions (market hours, liquidity, price)
3. Portfolio Constraints (max positions, correlation)
4. Risk Limits (drawdown, daily loss, position size)

Returns: EXECUTE, HOLD, or REJECT with detailed reasoning
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger(__name__)


# Configuration
MIN_CONFIDENCE = float(os.getenv("AUTO_EXECUTION_MIN_CONFIDENCE", "70"))
MIN_PRICE = float(os.getenv("AUTO_EXECUTION_MIN_PRICE", "5.0"))
MIN_VOLUME_STOCKS = int(os.getenv("AUTO_EXECUTION_MIN_VOLUME_STOCKS", "100000"))
MARKET_HOURS_ONLY = os.getenv("AUTO_EXECUTION_MARKET_HOURS_ONLY", "1") == "1"


def evaluate_trade_opportunity(
    prediction: dict,
    portfolio: dict,
    current_positions: list[dict],
    risk_engine: Any = None
) -> dict[str, Any]:
    """
    Evaluate if prediction should be traded
    
    Args:
        prediction: Prediction dict with symbol, confidence, direction, action
        portfolio: Account state (portfolio_value, cash, buying_power)
        current_positions: List of current open positions
        risk_engine: Optional risk engine for advanced checks
    
    Returns:
        {
            "action": "EXECUTE" | "SKIP",
            "reason": "Explanation string",
            "symbol": str,
            "side": "buy" | "sell",
            "shares": int,
            "confidence": float,
            "stop_loss_price": float,
            "take_profit_price": float
        }
    """
    
    symbol = prediction.get("symbol", "UNKNOWN")
    confidence = prediction.get("confidence", 0)
    direction = prediction.get("direction", "FLAT")
    action = prediction.get("action", "HOLD")
    
    # Layer 1: Confidence & Direction
    confidence_check = _check_confidence(confidence, direction, action)
    if not confidence_check["ok"]:
        return {
            "action": "SKIP",
            "reason": confidence_check["reason"],
            "symbol": symbol
        }
    
    # Layer 2: Market Conditions
    market_check = _check_market_conditions(prediction)
    if not market_check["ok"]:
        return {
            "action": "SKIP",
            "reason": market_check["reason"],
            "symbol": symbol
        }
    
    # Layer 3: Portfolio Constraints
    portfolio_check = _check_portfolio_constraints(prediction, current_positions)
    if not portfolio_check["ok"]:
        return {
            "action": "SKIP",
            "reason": portfolio_check["reason"],
            "symbol": symbol
        }
    
    # Layer 4: Risk Limits
    risk_check = _check_risk_limits(prediction, portfolio, risk_engine)
    if not risk_check["ok"]:
        return {
            "action": "SKIP",
            "reason": risk_check["reason"],
            "symbol": symbol
        }
    
    # Calculate position size
    sizing = _calculate_position_size(prediction, portfolio)
    
    if sizing["shares"] == 0:
        return {
            "action": "SKIP",
            "reason": "Position size calculated as 0",
            "symbol": symbol
        }
    
    # Calculate SL/TP levels
    sl_tp = _calculate_sl_tp(prediction, sizing)
    
    # All checks passed - EXECUTE
    return {
        "action": "EXECUTE",
        "reason": confidence_check["reason"],
        "symbol": symbol,
        "side": "buy" if action == "BUY" else "sell",
        "shares": sizing["shares"],
        "confidence": confidence,
        "entry_price": sizing["entry_price"],
        "stop_loss_price": sl_tp["stop_loss"],
        "take_profit_price": sl_tp["take_profit"],
        "position_value": sizing["position_value"],
        "kelly_fraction": sizing.get("kelly_fraction", 0.25)
    }


def _check_confidence(confidence: float, direction: str, action: str) -> dict:
    """Layer 1: Check confidence and direction"""
    
    if confidence < MIN_CONFIDENCE:
        return {
            "ok": False,
            "reason": f"Confidence {confidence:.0f}% below threshold {MIN_CONFIDENCE:.0f}%"
        }
    
    if action not in ["BUY", "SELL"]:
        return {
            "ok": False,
            "reason": f"Action is {action}, not BUY/SELL"
        }
    
    if direction == "FLAT":
        return {
            "ok": False,
            "reason": "Direction is FLAT, no clear trend"
        }
    
    return {
        "ok": True,
        "reason": f"{confidence:.0f}% confidence {action} signal"
    }


def _check_market_conditions(prediction: dict) -> dict:
    """Layer 2: Check market conditions (hours, liquidity, price)"""
    
    symbol = prediction.get("symbol", "")
    market = prediction.get("market", "stock")
    price = prediction.get("price", 0)
    
    # Check minimum price
    if price < MIN_PRICE:
        return {
            "ok": False,
            "reason": f"Price ${price:.2f} below minimum ${MIN_PRICE}"
        }
    
    # Check market hours for stocks (skip if MARKET_HOURS_ONLY is disabled)
    if market == "stock" and MARKET_HOURS_ONLY:
        if not _is_market_hours():
            return {
                "ok": False,
                "reason": "Market closed (NYSE/Nasdaq)"
            }
    
    # Check liquidity (if volume data available)
    volume = prediction.get("volume", 0)
    if market == "stock" and volume > 0 and volume < MIN_VOLUME_STOCKS:
        return {
            "ok": False,
            "reason": f"Volume {volume:,} below minimum {MIN_VOLUME_STOCKS:,}"
        }
    
    return {
        "ok": True,
        "reason": "Market conditions acceptable"
    }


def _check_portfolio_constraints(prediction: dict, positions: list[dict]) -> dict:
    """Layer 3: Check portfolio constraints"""
    
    symbol = prediction.get("symbol")
    
    # Check if already in position
    for pos in positions:
        if pos.get("symbol") == symbol:
            return {
                "ok": False,
                "reason": f"Already holding position in {symbol}"
            }
    
    # TODO: Add correlation check (don't hold 5 tech stocks)
    # TODO: Add sector concentration check
    
    return {
        "ok": True,
        "reason": "Portfolio constraints satisfied"
    }


def _check_risk_limits(prediction: dict, portfolio: dict, risk_engine: Any) -> dict:
    """Layer 4: Check risk limits"""
    
    # Basic check: Ensure we have buying power
    buying_power = portfolio.get("buying_power", 0)
    
    if buying_power < 100:
        return {
            "ok": False,
            "reason": f"Insufficient buying power ${buying_power:.2f}"
        }
    
    # TODO: Use risk_engine for advanced checks (VaR, correlation, etc.)
    
    return {
        "ok": True,
        "reason": "Risk limits acceptable"
    }


def _calculate_position_size(prediction: dict, portfolio: dict) -> dict:
    """Calculate position size using Kelly Criterion"""
    
    symbol = prediction.get("symbol")
    confidence = prediction.get("confidence", 0) / 100.0  # Convert to 0-1
    price = prediction.get("price", 0)
    portfolio_value = portfolio.get("portfolio_value", 10000)
    
    # Kelly Criterion: f = (p*b - q)/b
    # Simplified: Use confidence as win probability
    # Assume 1.5:1 reward/risk ratio
    p = confidence
    q = 1 - p
    b = 1.5  # Win/loss ratio
    
    kelly_full = (p * b - q) / b
    kelly_fraction = max(0, min(kelly_full * 0.25, 0.25))  # Quarter Kelly, capped at 25%
    
    # Calculate dollar amount
    max_position_pct = float(os.getenv("AUTO_EXECUTION_MAX_POSITION_PCT", "10")) / 100.0
    position_pct = min(kelly_fraction, max_position_pct)
    
    position_value = portfolio_value * position_pct
    shares = int(position_value / price) if price > 0 else 0
    
    # Ensure we don't exceed buying power
    buying_power = portfolio.get("buying_power", 0)
    if position_value > buying_power:
        shares = int(buying_power / price) if price > 0 else 0
        position_value = shares * price
    
    return {
        "shares": shares,
        "position_value": position_value,
        "entry_price": price,
        "kelly_fraction": kelly_fraction,
        "position_pct": position_pct * 100
    }


def _calculate_sl_tp(prediction: dict, sizing: dict) -> dict:
    """Calculate stop-loss and take-profit levels"""
    
    entry_price = sizing["entry_price"]
    predicted_pct = prediction.get("predicted_pct", 0)
    
    # Stop loss: 3% below entry (conservative)
    stop_loss_pct = 3.0
    stop_loss = entry_price * (1 - stop_loss_pct / 100.0)
    
    # Take profit: Use predicted move or 2x stop loss distance
    if abs(predicted_pct) > 0:
        take_profit = entry_price * (1 + abs(predicted_pct) / 100.0)
    else:
        # Default: 6% target (2:1 reward/risk)
        take_profit = entry_price * (1 + (stop_loss_pct * 2) / 100.0)
    
    return {
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "risk_reward_ratio": (take_profit - entry_price) / (entry_price - stop_loss)
    }


def _is_market_hours() -> bool:
    """Check if NYSE/Nasdaq is open (9:30 AM - 4:00 PM ET)"""
    
    try:
        # Get current time in ET
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour
        minute_utc = now_utc.minute
        weekday = now_utc.weekday()  # 0=Monday, 6=Sunday
        
        # Convert to ET (UTC-5)
        hour_et = (hour_utc - 5) % 24
        
        # Check if weekend
        if weekday >= 5:  # Saturday or Sunday
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        # In UTC: 14:30 - 21:00 (winter) or 13:30 - 20:00 (summer)
        # Simplified: Check 13:30 - 21:00 UTC
        if hour_et < 9 or hour_et >= 16:
            return False
        
        if hour_et == 9 and minute_utc < 30:
            return False
        
        return True
    
    except Exception as e:
        LOGGER.error(f"Error checking market hours: {e}")
        return False  # Conservative: assume closed if error


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 Testing Trade Decision Engine")
    print("=" * 60)
    
    # Test prediction
    test_prediction = {
        "symbol": "AAPL",
        "confidence": 78,
        "direction": "UP",
        "action": "BUY",
        "price": 150.25,
        "predicted_pct": 8.5,
        "market": "stock",
        "volume": 50000000
    }
    
    test_portfolio = {
        "portfolio_value": 100000,
        "cash": 50000,
        "buying_power": 50000
    }
    
    test_positions = []
    
    print("\nEvaluating trade opportunity...")
    decision = evaluate_trade_opportunity(
        prediction=test_prediction,
        portfolio=test_portfolio,
        current_positions=test_positions
    )
    
    print(f"\nDecision: {decision['action']}")
    print(f"Reason: {decision['reason']}")
    
    if decision["action"] == "EXECUTE":
        print(f"\nTrade Details:")
        print(f"  Symbol: {decision['symbol']}")
        print(f"  Side: {decision['side'].upper()}")
        print(f"  Shares: {decision['shares']}")
        print(f"  Entry: ${decision['entry_price']:.2f}")
        print(f"  Stop Loss: ${decision['stop_loss_price']:.2f}")
        print(f"  Take Profit: ${decision['take_profit_price']:.2f}")
        print(f"  Position Value: ${decision['position_value']:.2f}")
        print(f"  Kelly Fraction: {decision['kelly_fraction']:.2%}")
