"""
Momentum shift detection system.
Tracks momentum changes and triggers alerts when dramatic shifts occur.
"""

import time
from typing import Any

# Store momentum history
_MOMENTUM_HISTORY: dict[str, list[dict[str, Any]]] = {}


def track_momentum(symbol: str, momentum: float, timestamp: float | None = None) -> None:
    """
    Track momentum value for a symbol.
    
    Args:
        symbol: Stock/crypto ticker
        momentum: Current momentum score (-1 to +1)
        timestamp: Unix timestamp (defaults to now)
    """
    if symbol not in _MOMENTUM_HISTORY:
        _MOMENTUM_HISTORY[symbol] = []
    
    ts = timestamp or time.time()
    
    _MOMENTUM_HISTORY[symbol].append({
        "momentum": momentum,
        "timestamp": ts
    })
    
    # Keep only last 100 data points to prevent memory bloat
    if len(_MOMENTUM_HISTORY[symbol]) > 100:
        _MOMENTUM_HISTORY[symbol] = _MOMENTUM_HISTORY[symbol][-100:]


def detect_momentum_shift(
    symbol: str,
    current_momentum: float,
    lookback_minutes: int = 60
) -> dict[str, Any]:
    """
    Detect if momentum has shifted dramatically.
    
    Args:
        symbol: Stock/crypto ticker
        current_momentum: Current momentum score
        lookback_minutes: How far back to compare (default 60 min)
    
    Returns:
        {
            "shift_detected": bool,
            "shift_magnitude": float (% change),
            "shift_direction": "BULLISH" | "BEARISH" | None,
            "previous_momentum": float,
            "current_momentum": float,
            "alert_priority": "HIGH" | "MEDIUM" | "LOW"
        }
    """
    if symbol not in _MOMENTUM_HISTORY or len(_MOMENTUM_HISTORY[symbol]) < 2:
        return {
            "shift_detected": False,
            "shift_magnitude": 0,
            "shift_direction": None,
            "current_momentum": current_momentum,
            "alert_priority": "LOW"
        }
    
    # Find momentum from lookback period
    now = time.time()
    lookback_timestamp = now - (lookback_minutes * 60)
    
    # Get closest historical momentum
    history = _MOMENTUM_HISTORY[symbol]
    previous_momentum = None
    
    for entry in reversed(history):
        if entry["timestamp"] <= lookback_timestamp:
            previous_momentum = entry["momentum"]
            break
    
    # If no history in lookback window, use oldest available
    if previous_momentum is None and history:
        previous_momentum = history[0]["momentum"]
    
    if previous_momentum is None:
        return {
            "shift_detected": False,
            "shift_magnitude": 0,
            "shift_direction": None,
            "current_momentum": current_momentum,
            "alert_priority": "LOW"
        }
    
    # Calculate shift magnitude
    if previous_momentum == 0:
        shift_pct = 0
    else:
        shift_pct = ((current_momentum - previous_momentum) / abs(previous_momentum)) * 100
    
    # Determine if shift is significant
    shift_detected = abs(shift_pct) > 30  # >30% change
    
    # Determine direction
    if current_momentum > previous_momentum and current_momentum > 0:
        shift_direction = "BULLISH"
    elif current_momentum < previous_momentum and current_momentum < 0:
        shift_direction = "BEARISH"
    else:
        shift_direction = None
    
    # Alert priority based on magnitude
    if abs(shift_pct) > 50:
        alert_priority = "HIGH"
    elif abs(shift_pct) > 30:
        alert_priority = "MEDIUM"
    else:
        alert_priority = "LOW"
    
    return {
        "shift_detected": shift_detected,
        "shift_magnitude": round(shift_pct, 2),
        "shift_direction": shift_direction,
        "previous_momentum": round(previous_momentum, 3),
        "current_momentum": round(current_momentum, 3),
        "alert_priority": alert_priority,
        "lookback_minutes": lookback_minutes
    }


def get_momentum_history(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    """Get recent momentum history for a symbol."""
    if symbol not in _MOMENTUM_HISTORY:
        return []
    
    return _MOMENTUM_HISTORY[symbol][-limit:]


def clear_momentum_history(symbol: str | None = None) -> dict[str, Any]:
    """Clear momentum history (all symbols or specific symbol)."""
    if symbol:
        if symbol in _MOMENTUM_HISTORY:
            del _MOMENTUM_HISTORY[symbol]
            return {"ok": True, "message": f"Cleared history for {symbol}"}
        return {"ok": False, "error": "Symbol not found in history"}
    else:
        _MOMENTUM_HISTORY.clear()
        return {"ok": True, "message": "Cleared all momentum history"}
