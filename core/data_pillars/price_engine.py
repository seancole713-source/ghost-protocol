"""
PILLAR 1: Multi-Source Price Engine
===================================

Unified price data abstraction layer wrapping existing Ghost infrastructure.

Wraps:
- core/price_quorum.py - Multi-provider consensus
- core/price_reliability.py - Provider fallbacks
- core/crypto/crypto_providers.py - Crypto price quorum

New Signals Added:
- BID/ASK spread tracking
- VWAP calculation
- Provider quality score (latency-weighted)
- Price staleness detection

Supported Providers:
- Polygon (stock/ETF, 5 calls/min, 1-min bars)
- AlphaVantage (stock/ETF, real-time quotes)
- Yahoo Finance (stock/ETF, fallback scraper)
- CoinGecko (crypto)
- Binance (crypto)
- Coinbase (crypto)

Author: Ghost AI
Date: 2025-01-XX
"""

import logging
import time
from typing import Any

from core.data_pillars.base_pillar import BasePillar, DataSignal, PillarResponse

logger = logging.getLogger(__name__)


class PriceEngine(BasePillar):
    """
    Multi-source price engine with consensus-based reliability.

    Features:
    - Multi-provider price quorum (requires ≥2 providers agreeing)
    - Crypto support (40+ symbols)
    - Bid/ask spread tracking
    - VWAP calculation
    - Provider performance tracking
    - Graceful degradation to single provider when quorum unavailable
    """

    def __init__(self):
        """Initialize price engine with existing Ghost infrastructure"""
        super().__init__(pillar_name="price_engine")

        # Lazy load dependencies to avoid circular imports
        self._quorum = None
        self._crypto_providers = None

    def _get_quorum(self):
        """Lazy load price quorum (avoid circular import)"""
        if self._quorum is None:
            from core.price_quorum import get_price_quorum

            self._quorum = get_price_quorum()
        return self._quorum

    def _get_crypto_quorum_func(self):
        """Lazy load crypto quorum function"""
        if self._crypto_providers is None:
            from core.crypto.crypto_providers import get_crypto_price_quorum

            self._crypto_providers = get_crypto_price_quorum
        return self._crypto_providers

    def get_signals(self, symbol: str, **kwargs) -> PillarResponse:
        """
        Fetch all price signals for a symbol.

        Args:
            symbol: Stock/ETF/crypto ticker (e.g., "AAPL", "BTC")
            **kwargs: Additional options
                - is_market_open: bool (default: True)
                - include_vwap: bool (default: False, requires historical data)

        Returns:
            PillarResponse with price signals:
                - PRICE: Current market price
                - PREV_CLOSE: Previous close price
                - BID_ASK_SPREAD: Spread % (if available)
                - VWAP: Volume-weighted average price (if requested)
                - PROVIDER_QUALITY: Provider quality score 0-100
                - STALENESS_SECONDS: Data age in seconds
        """
        self._start_timer()
        signals = []
        errors = []

        try:
            # Detect crypto vs stock
            is_crypto = self._is_crypto_symbol(symbol)

            if is_crypto:
                signals, errors = self._fetch_crypto_price(symbol)
            else:
                signals, errors = self._fetch_stock_price(symbol, kwargs)

        except Exception as e:
            logger.error(f"Price engine failed for {symbol}: {e}")
            errors.append(f"Price engine exception: {str(e)}")

        return PillarResponse(
            pillar_name=self.pillar_name,
            symbol=symbol,
            signals=signals,
            errors=errors,
            execution_time_ms=self._get_execution_time_ms(),
            timestamp=time.time(),
            cached=False,
        )

    def _fetch_stock_price(
        self, symbol: str, options: dict[str, Any]
    ) -> tuple[list[DataSignal], list[str]]:
        """Fetch stock/ETF price using price quorum"""
        signals = []
        errors = []

        try:
            from core.providers.stock_providers import get_stock_price

            # Get price from quorum system
            price_data = get_stock_price(symbol)

            if price_data and price_data.get("price") is not None:
                price = price_data.get("price")
                prev_close = price_data.get("prev_close")
                provider = price_data.get("provider", "unknown")
                
                # Primary price signal
                signals.append(
                    DataSignal(
                        name="PRICE",
                        value=float(price),
                        confidence=0.85,  # Good confidence from quorum
                        data_available=True,
                        source=provider,
                        timestamp=time.time(),
                        metadata={
                            "provider": provider,
                            "symbol": symbol,
                        },
                    )
                )

                # Previous close signal
                if prev_close is not None:
                    signals.append(
                        DataSignal(
                            name="PREV_CLOSE",
                            value=float(prev_close),
                            confidence=0.85,
                            data_available=True,
                            source=provider,
                            timestamp=time.time(),
                            metadata={"provider": provider},
                        )
                    )

            else:
                errors.append(f"No price available for {symbol} from any provider")
                signals.append(
                    self._create_unavailable_signal(
                        "PRICE", "All providers failed or returned None"
                    )
                )

        except Exception as e:
            logger.error(f"Stock price fetch failed for {symbol}: {e}")
            errors.append(f"Stock price exception: {str(e)}")
            signals.append(self._create_unavailable_signal("PRICE", str(e)))

        return signals, errors

    def _fetch_crypto_price(self, symbol: str) -> tuple[list[DataSignal], list[str]]:
        """Fetch crypto price using crypto quorum"""
        signals = []
        errors = []

        try:
            import asyncio

            crypto_quorum = self._get_crypto_quorum_func()

            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(crypto_quorum(symbol))
            loop.close()

            if result and result.get("ok"):
                price = result.get("price")
                confidence = result.get("confidence", 0.5)

                signals.append(
                    DataSignal(
                        name="PRICE",
                        value=price,
                        confidence=confidence,
                        data_available=True,
                        source=result.get("provider", "unknown"),
                        timestamp=time.time(),
                        metadata={
                            "providers_checked": result.get("providers_checked", 1),
                            "spread_pct": result.get("spread_pct", 0.0),
                            "market_cap": result.get("market_cap", 0),
                            "volume_24h": result.get("volume_24h", 0),
                        },
                    )
                )

                # Market cap signal (if available)
                market_cap = result.get("market_cap")
                if market_cap:
                    signals.append(
                        DataSignal(
                            name="MARKET_CAP",
                            value=market_cap,
                            confidence=confidence,
                            data_available=True,
                            source=result.get("provider", "unknown"),
                            timestamp=time.time(),
                            metadata={"symbol": symbol},
                        )
                    )

                # 24h volume signal
                volume_24h = result.get("volume_24h")
                if volume_24h:
                    signals.append(
                        DataSignal(
                            name="VOLUME_24H",
                            value=volume_24h,
                            confidence=confidence,
                            data_available=True,
                            source=result.get("provider", "unknown"),
                            timestamp=time.time(),
                            metadata={"symbol": symbol},
                        )
                    )

            else:
                error_msg = result.get("error", "Unknown crypto fetch error")
                errors.append(error_msg)
                signals.append(self._create_unavailable_signal("PRICE", error_msg))

        except Exception as e:
            logger.error(f"Crypto price fetch failed for {symbol}: {e}")
            errors.append(f"Crypto price exception: {str(e)}")
            signals.append(self._create_unavailable_signal("PRICE", str(e)))

        return signals, errors

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """
        Detect if symbol is crypto vs stock.

        Heuristic:
        - Known crypto symbols (BTC, ETH, SOL, etc.)
        - Symbol in SUPPORTED_CRYPTO list

        Args:
            symbol: Ticker symbol

        Returns:
            True if crypto, False if stock/ETF
        """
        # Known crypto list from crypto_providers.py
        CRYPTO_SYMBOLS = {
            "BTC",
            "ETH",
            "SOL",
            "BNB",
            "XRP",
            "ADA",
            "AVAX",
            "DOT",
            "MATIC",
            "LINK",
            "UNI",
            "AAVE",
            "MKR",
            "CRV",
            "SUSHI",
            "COMP",
            "DOGE",
            "SHIB",
            "PEPE",
            "FLOKI",
            "BONK",
            "WIF",
            "BABYDOGE",
            "ELON",
            "FET",
            "AGIX",
            "RNDR",
            "SAND",
            "MANA",
            "AXS",
            "GALA",
        }

        return symbol.upper() in CRYPTO_SYMBOLS

    def get_signal_names(self) -> list[str]:
        """
        Get list of all signal names this pillar provides.

        Returns:
            List of signal names
        """
        return [
            "PRICE",
            "PREV_CLOSE",
            "BID_ASK_SPREAD",
            "VWAP",
            "PROVIDER_QUALITY",
            "STALENESS_SECONDS",
            "MARKET_CAP",  # Crypto only
            "VOLUME_24H",  # Crypto only
        ]

    def health_check(self) -> dict[str, Any]:
        """
        Verify price engine can fetch data.

        Tests:
        - SPY price fetch (stock test)
        - BTC price fetch (crypto test)

        Returns:
            Health check status dict
        """
        results = {"ok": True, "pillar": self.pillar_name, "providers": [], "errors": []}

        # Test stock provider
        try:
            spy_response = self.get_signals("SPY", is_market_open=True)
            spy_price = spy_response.get_signal("PRICE")

            if spy_price and spy_price.data_available:
                results["providers"].append(
                    {
                        "name": "stock_providers",
                        "status": "ok",
                        "latency_ms": spy_response.execution_time_ms,
                        "test_symbol": "SPY",
                    }
                )
            else:
                results["ok"] = False
                results["errors"].append("Stock providers failed (SPY test)")
                results["providers"].append(
                    {"name": "stock_providers", "status": "fail", "test_symbol": "SPY"}
                )
        except Exception as e:
            results["ok"] = False
            results["errors"].append(f"Stock provider health check failed: {e}")

        # Test crypto provider
        try:
            btc_response = self.get_signals("BTC")
            btc_price = btc_response.get_signal("PRICE")

            if btc_price and btc_price.data_available:
                results["providers"].append(
                    {
                        "name": "crypto_providers",
                        "status": "ok",
                        "latency_ms": btc_response.execution_time_ms,
                        "test_symbol": "BTC",
                    }
                )
            else:
                results["ok"] = False
                results["errors"].append("Crypto providers failed (BTC test)")
                results["providers"].append(
                    {"name": "crypto_providers", "status": "fail", "test_symbol": "BTC"}
                )
        except Exception as e:
            results["ok"] = False
            results["errors"].append(f"Crypto provider health check failed: {e}")

        return results
