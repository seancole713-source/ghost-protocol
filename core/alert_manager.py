"""Alert manager.

The orchestrator expects `alert_processor_loop()`.

This module centralizes high-signal operational alerts (circuit breaker, risk
kill switch, regime changes) and sends them via Telegram when configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

LOGGER = logging.getLogger(__name__)

_LAST_SENT: dict[str, int] = {}


def _throttle(key: str, min_interval_s: int) -> bool:
    now = int(time.time())
    last = _LAST_SENT.get(key, 0)
    if now - last < min_interval_s:
        return False
    _LAST_SENT[key] = now
    return True


def _send_message(text: str) -> bool:
    try:
        from core.telegram_hunter import send_telegram_message

        send_telegram_message(text)
        return True
    except Exception:
        pass

    try:
        from core import telegram_alerts

        send = getattr(telegram_alerts, "send_text", None)
        if callable(send):
            send(text)
            return True
    except Exception:
        pass

    return False


def send_daily_briefing_alert(text: str) -> bool:
    """Send a daily briefing alert if Telegram is configured."""
    try:
        from core import telegram_alerts

        # Prefer a module-level helper if present.
        send = getattr(telegram_alerts, "send_text", None)
        if callable(send):
            send(text)
            return True

        # Fall back to a best-effort attribute.
        enqueue = getattr(telegram_alerts, "enqueue_alert_text", None)
        if callable(enqueue):
            enqueue(text)
            return True

        return False
    except Exception:
        LOGGER.exception("send_daily_briefing_alert_failed")
        return False


async def alert_processor_loop() -> None:
    """Background loop for operational alert processing."""
    interval_s = int(os.getenv("ALERT_MANAGER_INTERVAL_S", "60"))

    while True:
        try:
            # 1) Circuit breaker state
            try:
                from core.autonomous_execution_engine import get_execution_status

                exec_status = get_execution_status() or {}
                if exec_status.get("circuit_breaker_active") and _throttle("circuit_breaker", 300):
                    reason = exec_status.get("circuit_breaker_reason", "")
                    _send_message(f"🚨 Circuit breaker active. Reason: {reason}")
            except Exception:
                pass

            # 2) Risk kill / drawdown info
            try:
                from core.risk_engine import get_risk_engine

                r = get_risk_engine().get_status()
                if r.get("kill_switch") and _throttle("risk_kill", 300):
                    _send_message("🚨 RISK_KILL is active. Trading should be halted.")
            except Exception:
                pass

            # 3) Regime change notifications (throttled)
            try:
                from core.market_regime import get_current_regime

                regime = get_current_regime() or {}
                key = f"regime:{regime.get('regime','unknown')}"
                if _throttle(key, 3600):
                    _send_message(
                        f"📊 Market regime: {regime.get('regime')} (conf {float(regime.get('confidence',0))*100:.0f}%)"
                    )
            except Exception:
                pass

            LOGGER.debug("alert_manager_tick", extra={"ts": int(time.time()), "interval_s": interval_s})
        except Exception:
            LOGGER.exception("alert_manager_error")

        await asyncio.sleep(max(10, interval_s))
