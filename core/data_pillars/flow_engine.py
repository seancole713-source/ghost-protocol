"""
PILLAR 6: Flow & Orderbook Engine
==================================

Tracks order flow and on-chain metrics (crypto).

Signals:
- BID_ASK_SPREAD
- ORDER_IMBALANCE
- WHALE_ACTIVITY (crypto)
- ON_CHAIN_VOLUME (crypto)

Note: Stock orderbook data requires expensive market data subscriptions.
      For now, provides basic bid/ask spread only.

Author: Ghost AI
Date: November 21, 2025
"""

import logging
import time
from typing import Any

from core.data_pillars.base_pillar import BasePillar, DataSignal, PillarResponse

logger = logging.getLogger(__name__)


class FlowEngine(BasePillar):
    """Order flow and orderbook analysis engine."""

    def __init__(self):
        super().__init__(pillar_name="flow_engine")

    def get_signals(self, symbol: str, **kwargs) -> PillarResponse:
        """
        Fetch order flow signals for a symbol.
        
        Note: Full orderbook data requires Level 2 subscriptions.
              Currently provides basic signals only.
        
        Returns:
            Signals: BID_ASK_SPREAD, ORDER_IMBALANCE (limited)
        """
        self._start_timer()
        signals = []
        errors = []

        try:
            # Detect crypto vs stock
            is_crypto = self._is_crypto_symbol(symbol)

            if is_crypto:
                signals = self._fetch_crypto_flow(symbol)
            else:
                signals = self._fetch_stock_flow(symbol)

        except Exception as e:
            logger.error(f"Flow engine failed for {symbol}: {e}")
            errors.append(f"Flow exception: {str(e)}")
            signals = self._create_unavailable_signals()

        return PillarResponse(
            pillar_name=self.pillar_name,
            symbol=symbol,
            signals=signals,
            errors=errors,
            execution_time_ms=self._get_execution_time_ms(),
            timestamp=time.time(),
            cached=False,
        )

    def _fetch_stock_flow(self, symbol: str) -> list[DataSignal]:
        """Fetch basic stock flow signals (bid/ask spread)"""
        signals = []
        ts = time.time()

        # For now, return placeholder unavailable signal
        # Full implementation requires Level 2 market data subscription
        signals.append(
            self._create_unavailable_signal(
                "BID_ASK_SPREAD",
                "Level 2 data subscription required for orderbook access"
            )
        )

        return signals

    def _fetch_crypto_flow(self, symbol: str) -> list[DataSignal]:
        """Fetch crypto on-chain and orderbook signals"""
        signals = []
        ts = time.time()

        # Placeholder for crypto flow signals
        # Would integrate with Binance/Coinbase orderbook APIs
        signals.append(
            self._create_unavailable_signal(
                "WHALE_ACTIVITY",
                "Crypto orderbook integration pending"
            )
        )

        return signals

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Detect if symbol is crypto"""
        from core.asset_classification import is_crypto_symbol

        return is_crypto_symbol(symbol)

    def _create_unavailable_signals(self) -> list[DataSignal]:
        """Create unavailable signals when data missing"""
        return [
            self._create_unavailable_signal(name, "Flow data unavailable")
            for name in self.get_signal_names()
        ]

    def get_signal_names(self) -> list[str]:
        """Get list of flow signal names"""
        return [
            "BID_ASK_SPREAD",
            "ORDER_IMBALANCE",
            "WHALE_ACTIVITY",
            "ON_CHAIN_VOLUME",
        ]

    def health_check(self) -> dict[str, Any]:
        """Verify flow engine status"""
        results = {
            "ok": True,
            "pillar": self.pillar_name,
            "providers": [],
            "errors": [],
        }

        # Flow engine is currently degraded (requires Level 2 data)
        results["ok"] = False
        results["errors"].append(
            "Flow engine degraded - Level 2 market data required for full functionality"
        )
        results["providers"].append(
            {
                "name": "orderbook_api",
                "status": "degraded",
                "reason": "Level 2 subscription required",
            }
        )

        return results
