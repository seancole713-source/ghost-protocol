"""Performance tracker.

The orchestrator expects `performance_monitor_loop()`.

This module periodically refreshes and logs the existing dashboard metrics from
`core.performance_dashboard`, providing a single place to monitor prediction
accuracy/win-rate over time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

LOGGER = logging.getLogger(__name__)

_LAST_ALERT_TS = 0


def _send_message(text: str) -> None:
    try:
        from core.telegram_hunter import send_telegram_message

        send_telegram_message(text)
    except Exception:
        return


async def performance_monitor_loop() -> None:
    """Periodically refresh and log performance dashboard metrics."""
    interval_s = int(os.getenv("PERFORMANCE_TRACKER_INTERVAL_S", "3600"))
    alert_win_rate_below = float(os.getenv("PERFORMANCE_ALERT_WIN_RATE_BELOW", "0"))  # 0 disables
    alert_min_predictions = int(os.getenv("PERFORMANCE_ALERT_MIN_PREDICTIONS", "50"))

    while True:
        try:
            from core.performance_dashboard import get_dashboard_metrics

            metrics = get_dashboard_metrics() or {}
            overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}

            preds = int(overall.get("predictions", 0) or 0)
            win_rate = float(overall.get("win_rate", 0) or 0)
            avg_conf = float(overall.get("avg_confidence", 0) or 0)

            LOGGER.info(
                "performance_tracker_metrics",
                extra={"predictions": preds, "win_rate": win_rate, "avg_confidence": avg_conf},
            )

            global _LAST_ALERT_TS
            if (
                alert_win_rate_below > 0
                and preds >= alert_min_predictions
                and win_rate > 0
                and win_rate < alert_win_rate_below
                and (time.time() - _LAST_ALERT_TS) > 6 * 3600
            ):
                _LAST_ALERT_TS = time.time()
                _send_message(
                    f"⚠️ Performance alert: win-rate {win_rate:.1f}% below {alert_win_rate_below:.1f}% (n={preds})."
                )

        except Exception:
            LOGGER.exception("performance_tracker_error")

        await asyncio.sleep(max(60, interval_s))
