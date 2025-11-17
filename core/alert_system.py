"""
Real-time price alert system.
Monitors price movements and triggers alerts when conditions are met.
"""

import time
from typing import Any

# Alert storage
_ALERTS: dict[str, list[dict[str, Any]]] = {}
_ALERT_HISTORY: list[dict[str, Any]] = []


def create_alert(
    symbol: str,
    alert_type: str,
    target_price: float | None = None,
    trigger_condition: str | None = None,
    message: str | None = None
) -> dict[str, Any]:
    """
    Create a price alert.

    Alert types:
    - "price_above": Trigger when price goes above target
    - "price_below": Trigger when price goes below target
    - "gain_pct": Trigger when gain % threshold hit
    - "confidence_spike": Trigger when confidence jumps significantly
    - "momentum_shift": Trigger when momentum changes dramatically

    Args:
        symbol: Stock/crypto ticker
        alert_type: Type of alert
        target_price: Price threshold (for price alerts)
        trigger_condition: Custom condition string
        message: Custom alert message

    Returns:
        Alert configuration
    """
    if symbol not in _ALERTS:
        _ALERTS[symbol] = []

    alert = {
        "id": f"{symbol}_{alert_type}_{int(time.time())}",
        "symbol": symbol,
        "alert_type": alert_type,
        "target_price": target_price,
        "trigger_condition": trigger_condition,
        "message": message or f"{symbol} {alert_type} alert triggered",
        "created_at": time.time(),
        "triggered": False,
        "active": True
    }

    _ALERTS[symbol].append(alert)
    return alert


def check_alerts(symbol: str, current_price: float, forecast_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Check if any alerts should trigger for a symbol.

    Args:
        symbol: Stock/crypto ticker
        current_price: Current price
        forecast_data: Latest forecast with confidence, momentum, etc.

    Returns:
        List of triggered alerts
    """
    if symbol not in _ALERTS:
        return []

    triggered_alerts = []

    for alert in _ALERTS[symbol]:
        if not alert["active"] or alert["triggered"]:
            continue

        should_trigger = False

        if alert["alert_type"] == "price_above":
            if current_price >= alert["target_price"]:
                should_trigger = True

        elif alert["alert_type"] == "price_below":
            if current_price <= alert["target_price"]:
                should_trigger = True

        elif alert["alert_type"] == "confidence_spike":
            confidence = forecast_data.get("confidence", 0)
            if confidence >= 0.85:  # High confidence threshold
                should_trigger = True

        elif alert["alert_type"] == "momentum_shift":
            gain_pct = forecast_data.get("gain_potential_pct", 0)
            if abs(gain_pct) > 5.0:  # >5% move
                should_trigger = True

        if should_trigger:
            alert["triggered"] = True
            alert["triggered_at"] = time.time()
            alert["trigger_price"] = current_price
            triggered_alerts.append(alert)
            _ALERT_HISTORY.append(alert.copy())

    return triggered_alerts


def get_active_alerts(symbol: str | None = None) -> list[dict[str, Any]]:
    """Get all active alerts (optionally filtered by symbol)."""
    if symbol:
        return [a for a in _ALERTS.get(symbol, []) if a["active"] and not a["triggered"]]

    all_alerts = []
    for alerts_list in _ALERTS.values():
        all_alerts.extend([a for a in alerts_list if a["active"] and not a["triggered"]])
    return all_alerts


def get_alert_history(limit: int = 50) -> list[dict[str, Any]]:
    """Get recent triggered alerts."""
    return _ALERT_HISTORY[-limit:]


def delete_alert(alert_id: str) -> dict[str, Any]:
    """Delete/deactivate an alert."""
    for symbol, alerts in _ALERTS.items():
        for alert in alerts:
            if alert["id"] == alert_id:
                alert["active"] = False
                return {"ok": True, "message": f"Alert {alert_id} deleted"}

    return {"ok": False, "error": "Alert not found"}
