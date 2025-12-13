"""Performance tracker (lightweight stub).

The orchestrator expects `performance_monitor_loop()`.
"""

from __future__ import annotations

import asyncio
import os
import time
import logging

LOGGER = logging.getLogger(__name__)


async def performance_monitor_loop() -> None:
    """Periodically emit performance tracking heartbeat."""
    interval_s = int(os.getenv("PERFORMANCE_TRACKER_INTERVAL_S", "3600"))

    while True:
        try:
            LOGGER.debug(
                "performance_tracker_tick",
                extra={"ts": int(time.time()), "interval_s": interval_s},
            )
        except Exception:
            LOGGER.exception("performance_tracker_error")

        await asyncio.sleep(max(60, interval_s))
