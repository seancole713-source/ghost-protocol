"""
PILLAR 5: World Context Engine
===============================

Tracks global market regime and macro signals.

Signals:
- SPY_PRICE, SPY_CHANGE
- VIX_LEVEL
- MARKET_REGIME (bull/bear/neutral)
- QQQ_CHANGE
- DXY_LEVEL (dollar index)

Author: Ghost AI
Date: November 21, 2025
"""

import logging
import time
from typing import Any

from core.data_pillars.base_pillar import BasePillar, DataSignal, PillarResponse

logger = logging.getLogger(__name__)


class WorldContextEngine(BasePillar):
    """Global market context and regime tracking."""

    def __init__(self):
        super().__init__(pillar_name="world_context_engine")

    def get_signals(self, symbol: str = "SPY", **kwargs) -> PillarResponse:
        """
        Fetch world market context signals.
        
        Note: Symbol parameter ignored - world context is global.
        
        Returns:
            Signals: SPY_PRICE, VIX_LEVEL, MARKET_REGIME, QQQ_CHANGE
        """
        self._start_timer()
        signals = []
        errors = []

        try:
            # Use existing world_context module
            from core.world_context import get_world_context

            context = get_world_context()

            if context:
                signals = self._parse_context_signals(context)
            else:
                errors.append("World context unavailable")
                signals = self._create_unavailable_signals()

        except Exception as e:
            logger.error(f"World context engine failed: {e}")
            errors.append(f"World context exception: {str(e)}")
            signals = self._create_unavailable_signals()

        return PillarResponse(
            pillar_name=self.pillar_name,
            symbol="WORLD",
            signals=signals,
            errors=errors,
            execution_time_ms=self._get_execution_time_ms(),
            timestamp=time.time(),
            cached=False,
        )

    def _parse_context_signals(self, context: dict) -> list[DataSignal]:
        """Parse world context into signals"""
        signals = []
        ts = time.time()

        try:
            # SPY price and change
            spy = context.get("spy", {})
            if spy.get("price"):
                signals.append(
                    DataSignal(
                        name="SPY_PRICE",
                        value=round(float(spy["price"]), 2),
                        confidence=1.0,
                        data_available=True,
                        source=spy.get("provider", "unknown"),
                        timestamp=ts,
                        metadata={},
                    )
                )

            if spy.get("change_pct") is not None:
                signals.append(
                    DataSignal(
                        name="SPY_CHANGE",
                        value=round(float(spy["change_pct"]), 2),
                        confidence=1.0,
                        data_available=True,
                        source="calculated",
                        timestamp=ts,
                        metadata={},
                    )
                )

            # VIX level
            vix = context.get("vix", {})
            if vix.get("level"):
                signals.append(
                    DataSignal(
                        name="VIX_LEVEL",
                        value=round(float(vix["level"]), 2),
                        confidence=1.0,
                        data_available=True,
                        source="quorum",
                        timestamp=ts,
                        metadata={"status": vix.get("status", "unknown")},
                    )
                )

            # Market mood/sentiment
            mood = context.get("market_mood", {})
            if mood.get("sentiment"):
                signals.append(
                    DataSignal(
                        name="MARKET_REGIME",
                        value=mood["sentiment"],  # "bull", "bear", "neutral"
                        confidence=mood.get("score", 50.0) / 100.0,
                        data_available=True,
                        source="calculated",
                        timestamp=ts,
                        metadata={"factors": mood.get("factors", [])},
                    )
                )

        except Exception as e:
            logger.error(f"Context signal parsing failed: {e}")

        return signals

    def _create_unavailable_signals(self) -> list[DataSignal]:
        """Create unavailable signals when data missing"""
        return [
            self._create_unavailable_signal(name, "World context unavailable")
            for name in self.get_signal_names()
        ]

    def get_signal_names(self) -> list[str]:
        """Get list of world context signal names"""
        return [
            "SPY_PRICE",
            "SPY_CHANGE",
            "VIX_LEVEL",
            "MARKET_REGIME",
        ]

    def health_check(self) -> dict[str, Any]:
        """Verify world context engine can fetch data"""
        results = {
            "ok": True,
            "pillar": self.pillar_name,
            "providers": [],
            "errors": [],
        }

        try:
            response = self.get_signals()

            if response.available_signal_count() >= 2:
                results["providers"].append(
                    {
                        "name": "world_context",
                        "status": "ok",
                        "latency_ms": response.execution_time_ms,
                        "signals_computed": response.available_signal_count(),
                    }
                )
            else:
                results["ok"] = False
                results["errors"].append("World context engine failed health check")

        except Exception as e:
            results["ok"] = False
            results["errors"].append(f"World context health check failed: {e}")

        return results
