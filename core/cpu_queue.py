"""CPU intensive task queue with metrics instrumentation."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import TypeVar

from .concurrency import ConcurrencyMetrics, ExecutionTimer

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class CPUTaskQueue:
    """Bounded concurrency thread pool for CPU heavy workloads."""

    def __init__(self, max_workers: int = 2, *, name: str = "cpu-queue") -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._name = name
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=name)
        self._metrics = ConcurrencyMetrics()
        self._max_workers = max_workers
        LOGGER.info("CPUTaskQueue initialized", extra={"queue_name": name, "max_workers": max_workers})

    @property
    def metrics(self) -> ConcurrencyMetrics:
        return self._metrics

    def run(
        self,
        fn: Callable[..., T],
        *args,
        label: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> T:
        """Execute ``fn`` on the queue and wait for completion."""

        task_label = label or f"{self._name}:task"
        with ExecutionTimer(task_label, logger=LOGGER, metrics=self._metrics):
            future: Future[T] = self._executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FutureTimeout as exc:
                future.cancel()
                raise TimeoutError(f"Task '{task_label}' exceeded timeout {timeout}s") from exc

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def snapshot(self) -> dict[str, float | int]:
        data = self._metrics.snapshot()
        data["name"] = self._name
        data["max_workers"] = self._max_workers
        return data


_CPU_QUEUE: CPUTaskQueue | None = None


def get_cpu_queue() -> CPUTaskQueue:
    global _CPU_QUEUE
    if _CPU_QUEUE is None:
        max_workers = int(os.getenv("CPU_QUEUE_MAX_WORKERS", "2"))
        _CPU_QUEUE = CPUTaskQueue(max_workers=max_workers)
    return _CPU_QUEUE
