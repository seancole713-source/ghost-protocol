"""Price quorum orchestration for multi-provider consensus."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import median

from .concurrency import AsyncRateLimiter, ConcurrencyMetrics, ExecutionTimer

LOGGER = logging.getLogger(__name__)
PriceFetcher = Callable[[], tuple[float | None, float | None, str]]


@dataclass
class PriceProvider:
    name: str
    fetcher: PriceFetcher
    enabled: bool = True
    rate_limiter: AsyncRateLimiter | None = None


@dataclass
class PriceQuote:
    provider: str
    price: float | None
    prev_close: float | None
    latency_ms: float
    raw_provider: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.price is not None


@dataclass
class PriceDecision:
    price: float | None
    prev_close: float | None
    provider_label: str
    reason: str
    quorum_size: int
    quotes: list[PriceQuote]
    latency_ms: float


class PriceQuorum:
    def __init__(
        self,
        *,
        min_quorum_open: int = 3,
        min_quorum_closed: int = 1,
        tolerance_open: float = 0.03,
        tolerance_closed: float = 0.06,
        logger: logging.Logger | None = None,
        name: str = "price-quorum",
    ) -> None:
        self.min_quorum_open = max(1, min_quorum_open)
        self.min_quorum_closed = max(1, min_quorum_closed)
        self.tolerance_open = max(0.0, tolerance_open)
        self.tolerance_closed = max(0.0, tolerance_closed)
        self.logger = logger or LOGGER
        self.name = name
        self._metrics = ConcurrencyMetrics()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=name, daemon=True)
        self._thread.start()

    @property
    def metrics(self) -> ConcurrencyMetrics:
        return self._metrics

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def snapshot(self) -> dict[str, float | int]:
        data = self._metrics.snapshot()
        data["name"] = self.name
        return data

    def get_price(
        self,
        symbol: str,
        providers: Iterable[PriceProvider],
        *,
        prev_close: float | None = None,
        is_market_open: bool,
        timeout: float = 6.0,
    ) -> PriceDecision:
        coro = self._get_price_async(
            symbol,
            list(providers),
            prev_close=prev_close,
            is_market_open=is_market_open,
            timeout=timeout,
        )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout + 1.0)

    async def _get_price_async(
        self,
        symbol: str,
        providers: list[PriceProvider],
        *,
        prev_close: float | None,
        is_market_open: bool,
        timeout: float,
    ) -> PriceDecision:
        quorum_requirement = self.min_quorum_open if is_market_open else self.min_quorum_closed
        tolerance = self.tolerance_open if is_market_open else self.tolerance_closed

        async with asyncio.timeout(timeout):
            with ExecutionTimer(
                f"price-quorum:{symbol}", logger=self.logger, metrics=self._metrics
            ):
                tasks = [
                    asyncio.create_task(self._fetch(provider))
                    for provider in providers
                    if provider.enabled
                ]
                quotes: list[PriceQuote] = []
                if tasks:
                    done, _ = await asyncio.wait(tasks)
                    for task in done:
                        try:
                            quotes.append(task.result())
                        except Exception as exc:  # pragma: no cover - defensive
                            self.logger.warning(
                                "provider task failed", extra={"symbol": symbol, "error": str(exc)}
                            )
                decision = self._decide(symbol, quotes, quorum_requirement, tolerance, prev_close)
                return decision

    async def _fetch(self, provider: PriceProvider) -> PriceQuote:
        start = self._loop.time()
        try:
            if provider.rate_limiter is not None:
                await provider.rate_limiter.acquire()
            price, prev, raw_provider = await asyncio.to_thread(provider.fetcher)
            latency_ms = (self._loop.time() - start) * 1000.0
            return PriceQuote(
                provider=provider.name,
                price=price,
                prev_close=prev,
                latency_ms=latency_ms,
                raw_provider=raw_provider or provider.name,
            )
        except Exception as exc:
            latency_ms = (self._loop.time() - start) * 1000.0
            return PriceQuote(
                provider=provider.name,
                price=None,
                prev_close=None,
                latency_ms=latency_ms,
                raw_provider=provider.name,
                error=str(exc),
            )

    def _decide(
        self,
        symbol: str,
        quotes: list[PriceQuote],
        quorum_requirement: int,
        tolerance: float,
        prev_close: float | None,
    ) -> PriceDecision:
        latency_ms = max((q.latency_ms for q in quotes), default=0.0)
        valid = [q for q in quotes if q.ok and q.price is not None]
        if not valid:
            return PriceDecision(
                price=None,
                prev_close=prev_close,
                provider_label="unavailable",
                reason="no_quotes",
                quorum_size=0,
                quotes=quotes,
                latency_ms=latency_ms,
            )
        prices = [q.price for q in valid if q.price is not None]
        if not prices:
            return PriceDecision(
                price=None,
                prev_close=prev_close,
                provider_label="unavailable",
                reason="no_prices",
                quorum_size=0,
                quotes=quotes,
                latency_ms=latency_ms,
            )
        m = median(prices)
        agreeing = [
            q for q in valid if q.price is not None and m > 0 and abs(q.price - m) / m <= tolerance
        ]
        if len(agreeing) >= quorum_requirement:
            consensus_price = median([q.price for q in agreeing if q.price is not None])
            label = (
                sorted(agreeing, key=lambda q: q.latency_ms)[0].raw_provider
                if agreeing
                else "consensus"
            )
            prev = next((q.prev_close for q in agreeing if q.prev_close is not None), prev_close)
            return PriceDecision(
                price=consensus_price,
                prev_close=prev,
                provider_label=label,
                reason="consensus",
                quorum_size=len(agreeing),
                quotes=quotes,
                latency_ms=latency_ms,
            )
        return PriceDecision(
            price=None,
            prev_close=prev_close,
            provider_label="unavailable",
            reason="quorum_failed",
            quorum_size=len(agreeing),
            quotes=quotes,
            latency_ms=latency_ms,
        )


_PRICE_QUORUM: PriceQuorum | None = None


def get_price_quorum() -> PriceQuorum:
    global _PRICE_QUORUM
    if _PRICE_QUORUM is None:
        _PRICE_QUORUM = PriceQuorum()
    return _PRICE_QUORUM
