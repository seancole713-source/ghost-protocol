import asyncio
import time

from core.concurrency import AsyncRateLimiter, ConcurrencyMetrics, ExecutionTimer
from core.cpu_queue import CPUTaskQueue


def test_execution_timer_records_metrics():
    metrics = ConcurrencyMetrics()
    with ExecutionTimer("unit-test", metrics=metrics, log_level=0):
        time.sleep(0.01)
    snapshot = metrics.snapshot()
    assert snapshot["total_executions"] == 1
    assert snapshot["active_tasks"] == 0


def test_async_rate_limiter_blocks_when_exhausted():
    limiter = AsyncRateLimiter(rate=1, per=0.2, jitter=0.0)

    async def _consume() -> float:
        await limiter.acquire()
        start = time.perf_counter()
        await limiter.acquire()
        return time.perf_counter() - start

    elapsed = asyncio.run(_consume())
    assert elapsed >= 0.19


def test_cpu_queue_executes_and_reports_metrics():
    queue = CPUTaskQueue(max_workers=1, name="test-queue")

    def _work() -> int:
        time.sleep(0.01)
        return 42

    result = queue.run(_work, label="unit-work")
    snapshot = queue.snapshot()
    queue.shutdown()

    assert result == 42
    assert snapshot["total_executions"] >= 1
    assert snapshot["max_workers"] == 1
