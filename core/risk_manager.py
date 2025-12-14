"""Risk manager.

The orchestrator expects `monitor_risk_loop()`.

This module adapts `core.risk_engine.RiskEngine` to continuously update risk
state from the configured broker account, and emits alerts when key thresholds
are breached (drawdown, kill-switch).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

LOGGER = logging.getLogger(__name__)

_LAST_ALERT_TS: dict[str, int] = {}


def _throttle(key: str, min_interval_s: int) -> bool:
    now = int(time.time())
    last = _LAST_ALERT_TS.get(key, 0)
    if now - last < min_interval_s:
        return False
    _LAST_ALERT_TS[key] = now
    return True


def _send_message(text: str) -> None:
    try:
        from core.telegram_hunter import send_telegram_message

        send_telegram_message(text)
        return
    except Exception:
        pass

    try:
        from core import telegram_alerts

        send = getattr(telegram_alerts, "send_text", None)
        if callable(send):
            send(text)
    except Exception:
        pass


async def monitor_risk_loop() -> None:
    """Periodically update portfolio risk state and alert on breaches."""
    interval_s = int(os.getenv("RISK_MANAGER_INTERVAL_S", "300"))
    max_drawdown_pct = float(os.getenv("AUTO_EXECUTION_MAX_DRAWDOWN_PCT", "15.0"))

    while True:
        try:
            from core.alpaca_broker import get_broker
            from core.risk_engine import get_risk_engine

            broker = get_broker()
            risk_engine = get_risk_engine()

            if not getattr(broker, "enabled", False):
                LOGGER.debug("risk_manager_broker_disabled")
                await asyncio.sleep(max(10, interval_s))
                continue

            account = broker.get_account() or {}
            portfolio_value = float(account.get("portfolio_value", 0) or 0)
            if portfolio_value > 0:
                risk_engine.update_portfolio_value(portfolio_value)

            status = risk_engine.get_status()
            drawdown = float(status.get("current", {}).get("drawdown_pct", 0) or 0)
            kill = bool(status.get("kill_switch"))

            if kill and _throttle("risk_kill", 900):
                _send_message("🚨 RISK KILL SWITCH ACTIVE (RISK_KILL=1). Trading should be halted.")

            if drawdown >= max_drawdown_pct and _throttle("drawdown", 900):
                _send_message(
                    f"🚨 DRAWDOWN ALERT: {drawdown:.1f}% (limit {max_drawdown_pct:.1f}%). Portfolio risk elevated."
                )

            LOGGER.info(
                "risk_manager_updated",
                extra={
                    "portfolio_value": portfolio_value,
                    "drawdown_pct": drawdown,
                    "kill_switch": kill,
                },
            )

            try:
                from wolf_app import _add_event

                _add_event(
                    "risk_update",
                    "Risk manager update",
                    {"portfolio_value": portfolio_value, "drawdown_pct": drawdown, "kill_switch": kill},
                )
            except Exception:
                pass

        except Exception:
            LOGGER.exception("risk_manager_error")

        await asyncio.sleep(max(10, interval_s))
