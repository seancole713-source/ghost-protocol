"""Market regime detector.

The orchestrator expects `regime_detector_loop()`.

This module adapts the existing `core.regime_detector.RegimeDetector` to a
background loop. It reads recent realized prices from `data/forecasts.db`
(`realized_prices` table) and periodically records the current regime.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from core.regime_detector import RegimeDetector

LOGGER = logging.getLogger(__name__)

FORECASTS_DB = Path(__file__).parent.parent / "data" / "forecasts.db"


@dataclass
class MarketRegime:
    regime: str
    ts: int
    confidence: float
    symbol: str


_CURRENT: MarketRegime = MarketRegime(regime="unknown", ts=0, confidence=0.0, symbol="SPY")
_DETECTOR: RegimeDetector | None = None


def _get_detector() -> RegimeDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = RegimeDetector()
    return _DETECTOR


def _get_recent_prices(symbol: str, points: int = 60, since_s: int = 7 * 86400) -> list[float]:
    if not FORECASTS_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(FORECASTS_DB))
        cur = conn.cursor()
        since = int(time.time()) - int(since_s)
        cur.execute(
            "SELECT price FROM realized_prices WHERE symbol=? AND ts>=? ORDER BY ts DESC LIMIT ?",
            (symbol.upper(), since, int(points)),
        )
        rows = cur.fetchall()
        conn.close()
        prices = [float(r[0]) for r in rows if r and r[0] is not None]
        prices.reverse()
        return prices
    except Exception:
        return []


def get_current_regime() -> dict:
    """Return current market regime snapshot."""
    return {
        "regime": _CURRENT.regime,
        "ts": _CURRENT.ts,
        "confidence": _CURRENT.confidence,
        "symbol": _CURRENT.symbol,
    }


def detect_market_regime_once(symbol: str | None = None) -> dict:
    """Compute and update market regime once."""
    sym = (symbol or os.getenv("MARKET_REGIME_SYMBOL", "SPY")).upper()

    override = os.getenv("MARKET_REGIME_OVERRIDE", "").strip()
    if override:
        global _CURRENT
        _CURRENT = MarketRegime(regime=override, ts=int(time.time()), confidence=1.0, symbol=sym)
        return get_current_regime()

    prices = _get_recent_prices(sym)
    if len(prices) < 10:
        LOGGER.debug("market_regime_insufficient_data", extra={"symbol": sym, "points": len(prices)})
        return get_current_regime()

    detector = _get_detector()
    res = detector.detect_regime(prices=prices, spy_price=prices[-1], vix_level=None)

    global _CURRENT
    _CURRENT = MarketRegime(
        regime=str(res.get("regime", "unknown")),
        ts=int(time.time()),
        confidence=float(res.get("confidence", 0.0) or 0.0),
        symbol=sym,
    )
    return {**get_current_regime(), "details": res}


async def regime_detector_loop() -> None:
    """Background loop to refresh regime."""
    interval_s = int(os.getenv("MARKET_REGIME_INTERVAL_S", "900"))  # 15 min

    while True:
        try:
            detect_market_regime_once()
            LOGGER.info(
                "market_regime_updated",
                extra={"regime": _CURRENT.regime, "confidence": _CURRENT.confidence, "symbol": _CURRENT.symbol},
            )
        except Exception:
            LOGGER.exception("market_regime_error")

        await asyncio.sleep(max(60, interval_s))
