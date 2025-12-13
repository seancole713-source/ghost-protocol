"""Market regime detector (lightweight implementation).

The orchestrator expects `regime_detector_loop()`.
Some legacy code references `get_current_regime()`.

This implementation is intentionally conservative: it avoids external API calls
and keeps state in-memory.
"""

from __future__ import annotations

import asyncio
import os
import time
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    regime: str
    ts: int
    confidence: float


_CURRENT: MarketRegime = MarketRegime(regime="unknown", ts=0, confidence=0.0)


def get_current_regime() -> dict:
    """Return current market regime snapshot."""
    return {"regime": _CURRENT.regime, "ts": _CURRENT.ts, "confidence": _CURRENT.confidence}


def _infer_regime_from_env() -> MarketRegime:
    # Allow operators to override via env without deploying code.
    override = os.getenv("MARKET_REGIME_OVERRIDE", "").strip().lower()
    if override:
        return MarketRegime(regime=override, ts=int(time.time()), confidence=1.0)

    # Default to neutral when we don't have enough signal.
    return MarketRegime(regime="neutral", ts=int(time.time()), confidence=0.25)


async def regime_detector_loop() -> None:
    """Background loop to refresh regime."""
    interval_s = int(os.getenv("MARKET_REGIME_INTERVAL_S", "900"))  # 15 min

    while True:
        try:
            global _CURRENT
            _CURRENT = _infer_regime_from_env()
            LOGGER.debug(
                "market_regime_updated",
                extra={"regime": _CURRENT.regime, "confidence": _CURRENT.confidence, "ts": _CURRENT.ts},
            )
        except Exception:
            LOGGER.exception("market_regime_error")

        await asyncio.sleep(max(60, interval_s))
