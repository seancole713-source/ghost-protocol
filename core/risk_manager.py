"""Risk manager (lightweight stub).

The orchestrator expects `monitor_risk_loop()`.
"""

from __future__ import annotations

import asyncio
import os
import time
import logging

LOGGER = logging.getLogger(__name__)


async def monitor_risk_loop() -> None:
    """Periodically check portfolio risk.

    This is intentionally minimal: it logs a heartbeat and yields.
    """
    interval_s = int(os.getenv("RISK_MANAGER_INTERVAL_S", "300"))

    while True:
        try:
            LOGGER.debug(
                "risk_manager_tick",
                extra={"ts": int(time.time()), "interval_s": interval_s},
            )
        except Exception:
            LOGGER.exception("risk_manager_error")

        await asyncio.sleep(max(10, interval_s))
