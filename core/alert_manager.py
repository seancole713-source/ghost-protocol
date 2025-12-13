"""Alert manager (lightweight implementation).

The orchestrator expects `alert_processor_loop()`.
Legacy code may call `send_daily_briefing_alert()`.

This is intentionally minimal and safe; it delegates to `core.telegram_alerts`
when available.
"""

from __future__ import annotations

import asyncio
import os
import time
import logging

LOGGER = logging.getLogger(__name__)


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
    """Background loop for alert processing.

    The system already has multiple alert queues/workers elsewhere; this loop is
    a safe placeholder to satisfy orchestrator wiring.
    """
    interval_s = int(os.getenv("ALERT_MANAGER_INTERVAL_S", "60"))

    while True:
        try:
            LOGGER.debug(
                "alert_manager_tick",
                extra={"ts": int(time.time()), "interval_s": interval_s},
            )
        except Exception:
            LOGGER.exception("alert_manager_error")

        await asyncio.sleep(max(10, interval_s))
