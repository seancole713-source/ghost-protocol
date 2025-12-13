"""Live recalculator (lightweight stub).

The orchestrator expects `live_recalculator_loop()`.
This module is intentionally minimal and non-blocking.
"""

from __future__ import annotations

import asyncio
import os
import time
import logging

LOGGER = logging.getLogger(__name__)


async def live_recalculator_loop() -> None:
    """Periodically recalculate live position guidance.

    Designed to be safe in constrained environments (Railway free tier).
    """
    interval_s = int(os.getenv("LIVE_RECALCULATOR_INTERVAL_S", "300"))

    while True:
        try:
            # Placeholder: a real implementation would pull open positions,
            # refresh prices, and update targets/trailing stops.
            LOGGER.debug(
                "live_recalculator_tick",
                extra={"ts": int(time.time()), "interval_s": interval_s},
            )
        except Exception:
            LOGGER.exception("live_recalculator_error")

        await asyncio.sleep(max(5, interval_s))
