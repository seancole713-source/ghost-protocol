"""Concurrency utilities for Ghost system.

Provides lightweight timing instrumentation, shared metrics tracking, and
asynchronous rate limiting primitives used across the application.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)


@dataclass
class ConcurrencyMetrics:
    """Thread-safe aggregate statistics for concurrent workloads."""

    active_tasks: int = 0
    max_concurrency: int = 0
    total_executions: int = 0
    total_failures: int = 0
    total_latency_s: float = 0.0
    last_latency_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_start(self) -> None:
        with self._lock:
            self.active_tasks += 1
            if self.active_tasks > self.max_concurrency:
                self.max_concurrency = self.active_tasks

    def record_end(self, latency_s: float, failed: bool = False) -> None:
        with self._lock:
            self.active_tasks = max(0, self.active_tasks - 1)
            self.total_executions += 1
            if failed:
                self.total_failures += 1
            self.total_latency_s += max(0.0, latency_s)
            self.last_latency_s = max(0.0, latency_s)

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if self.total_executions == 0:
                return 0.0
            return (self.total_latency_s / self.total_executions) * 1000.0

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "active_tasks": self.active_tasks,
                "max_concurrency": self.max_concurrency,
                "total_executions": self.total_executions,
                "total_failures": self.total_failures,
                "avg_latency_ms": self.avg_latency_ms,
                "last_latency_ms": self.last_latency_s * 1000.0,
            }


class ExecutionTimer:
    """Context manager capturing execution duration and emitting metrics/logs."""

    def __init__(
        self,
        label: str,
        *,
        logger: logging.Logger | None = None,
        metrics: ConcurrencyMetrics | None = None,
        log_level: int = logging.DEBUG,
    ) -> None:
        self.label = label
        self.logger = logger or LOGGER
        self.metrics = metrics
        self.log_level = log_level
        self._start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> ExecutionTimer:
        self._start = time.perf_counter()
        if self.metrics is not None:
            self.metrics.record_start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        end = time.perf_counter()
        self.elapsed = end - (self._start or end)
        failed = exc_type is not None
        if self.metrics is not None:
            self.metrics.record_end(self.elapsed, failed=failed)
        if self.logger is not None and self.logger.isEnabledFor(self.log_level):
            self.logger.log(
                self.log_level,
                "%s completed in %.2f ms%s",
                self.label,
                self.elapsed * 1000.0,
                " [failed]" if failed else "",
            )
        return False  # never suppress exceptions


class AsyncRateLimiter:
    """Token bucket rate limiter supporting async and sync acquisition.

    The limiter enforces ``rate`` executions every ``per`` seconds. Jitter is
    applied when the bucket is empty to avoid thundering herds.
    """

    def __init__(self, rate: int, per: float, *, jitter: float = 0.25) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if per <= 0:
            raise ValueError("per must be positive")
        self.rate = rate
        self.per = float(per)
        self.jitter = max(0.0, float(jitter))
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def _reserve(self) -> float:
        """Reserve a slot and return wait time in seconds."""
        now = time.monotonic()
        with self._lock:
            # Drop timestamps outside the window
            while self._timestamps and (now - self._timestamps[0]) >= self.per:
                self._timestamps.popleft()
            if len(self._timestamps) < self.rate:
                self._timestamps.append(now)
                return 0.0
            wait = (self._timestamps[0] + self.per) - now
            wait = max(0.0, wait)
            if self.jitter:
                wait += random.uniform(0.0, self.jitter * self.per)
            # Pretend this token is scheduled after the wait completes
            self._timestamps.append(now + wait)
            self._timestamps.popleft()
            return wait

    async def acquire(self) -> None:
        wait = self._reserve()
        if wait > 0:
            await asyncio.sleep(wait)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def blocking_acquire(self) -> None:
        wait = self._reserve()
        if wait > 0:
            time.sleep(wait)
