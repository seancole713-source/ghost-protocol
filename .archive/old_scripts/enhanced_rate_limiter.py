#!/usr/bin/env python3
"""
Enhanced Rate Limiting System for Price Providers
Features:
- Token bucket algorithm per provider
- Exponential backoff
- Provider health monitoring
- Automatic failover
- Request queuing
"""

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any


class ProviderStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    FAILED = "failed"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float  # Current tokens
    last_refill: float  # Last refill timestamp

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def time_until_available(self, tokens: int = 1) -> float:
        """Get seconds until N tokens are available."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


@dataclass
class ProviderHealth:
    """Track provider health metrics."""

    status: ProviderStatus
    total_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    last_success: float | None = None
    last_failure: float | None = None
    consecutive_failures: int = 0
    avg_response_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)

    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy enough to use."""
        return (
            self.status != ProviderStatus.FAILED
            and self.consecutive_failures < 5
            and self.success_rate > 0.5
        )


class EnhancedRateLimiter:
    """
    Advanced rate limiting system for price providers.

    Features:
    - Per-provider token buckets
    - Exponential backoff on rate limits
    - Provider health monitoring
    - Automatic failover to backup providers
    - Request queuing with priority
    """

    def __init__(self):
        self.lock = Lock()

        # Provider configurations (requests per second)
        self.provider_limits = {
            "yahoo": 5,  # 5 req/sec = 300/min (conservative for Yahoo Finance)
            "yfinance": 5,
            "polygon": 10,  # Polygon is more generous
            "alphavantage": 1,  # 5 calls/min for free tier = ~0.08/sec, use 1 for burst
            "chatgpt": 2,  # Conservative for API costs
        }

        # Initialize token buckets
        self.buckets: dict[str, TokenBucket] = {}
        for provider, limit in self.provider_limits.items():
            self.buckets[provider] = TokenBucket(
                capacity=limit * 10,  # 10 second burst capacity
                refill_rate=limit,
                tokens=limit * 10,  # Start full
                last_refill=time.time(),
            )

        # Provider health tracking
        self.health: dict[str, ProviderHealth] = defaultdict(
            lambda: ProviderHealth(status=ProviderStatus.HEALTHY)
        )

        # Backoff tracking (provider -> next_available_time)
        self.backoff: dict[str, float] = {}

        # Response time tracking for latency monitoring
        self.response_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def can_request(self, provider: str) -> bool:
        """Check if a request can be made to this provider."""
        with self.lock:
            # Check if provider is in backoff
            if provider in self.backoff:
                if time.time() < self.backoff[provider]:
                    return False
                else:
                    # Backoff expired, remove it
                    del self.backoff[provider]

            # Check if provider is healthy
            if not self.health[provider].is_healthy:
                return False

            # Check token bucket
            bucket = self.buckets.get(provider)
            if bucket:
                return bucket.tokens >= 1

            return True

    def wait_time(self, provider: str) -> float:
        """Get seconds until provider is available."""
        with self.lock:
            # Check backoff first
            if provider in self.backoff:
                backoff_wait = max(0, self.backoff[provider] - time.time())
                if backoff_wait > 0:
                    return backoff_wait

            # Check token bucket
            bucket = self.buckets.get(provider)
            if bucket:
                return bucket.time_until_available(1)

            return 0.0

    def request(self, provider: str, func: Callable, *args, **kwargs) -> tuple[Any, bool]:
        """
        Execute a request with rate limiting.

        Returns: (result, success)
        """
        with self.lock:
            # Try to consume token
            bucket = self.buckets.get(provider)
            if bucket and not bucket.consume(1):
                # No tokens available
                bucket.time_until_available(1)
                return (None, False)

            # Update metrics
            self.health[provider].total_requests += 1

        # Execute request (outside lock for concurrency)
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Update success metrics
            with self.lock:
                self._record_success(provider, elapsed)

            return (result, True)

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e).lower()

            # Detect rate limiting
            is_rate_limited = any(
                phrase in error_msg
                for phrase in ["429", "rate limit", "too many requests", "quota exceeded"]
            )

            with self.lock:
                if is_rate_limited:
                    self._record_rate_limit(provider)
                else:
                    self._record_failure(provider, elapsed)

            return (None, False)

    def _record_success(self, provider: str, response_time: float):
        """Record successful request."""
        health = self.health[provider]
        health.last_success = time.time()
        health.consecutive_failures = 0
        health.status = ProviderStatus.HEALTHY

        # Update average response time
        self.response_times[provider].append(response_time)
        health.avg_response_time = sum(self.response_times[provider]) / len(
            self.response_times[provider]
        )

    def _record_failure(self, provider: str, response_time: float):
        """Record failed request."""
        health = self.health[provider]
        health.failed_requests += 1
        health.last_failure = time.time()
        health.consecutive_failures += 1

        # Apply exponential backoff
        backoff_seconds = min(2**health.consecutive_failures, 300)  # Max 5 minutes
        self.backoff[provider] = time.time() + backoff_seconds

        # Update status
        if health.consecutive_failures >= 5:
            health.status = ProviderStatus.FAILED
        elif health.consecutive_failures >= 3:
            health.status = ProviderStatus.DEGRADED

    def _record_rate_limit(self, provider: str):
        """Record rate limit hit."""
        health = self.health[provider]
        health.rate_limited_requests += 1
        health.status = ProviderStatus.RATE_LIMITED

        # Apply aggressive backoff for rate limits
        # Start with 60 seconds, increase with repeated rate limits
        rate_limit_count = health.rate_limited_requests % 10  # Reset every 10
        backoff_seconds = min(60 * (2**rate_limit_count), 3600)  # Max 1 hour
        self.backoff[provider] = time.time() + backoff_seconds

    def get_best_provider(self, providers: list[str]) -> str | None:
        """
        Select the best available provider based on health and availability.

        Returns: provider name or None if none available
        """
        with self.lock:
            candidates = []

            for provider in providers:
                if not self.can_request(provider):
                    continue

                health = self.health[provider]
                if not health.is_healthy:
                    continue

                # Score: higher is better
                # Factors: success rate, response time, status
                score = health.success_rate * 100
                score -= health.avg_response_time  # Penalize slow providers

                if health.status == ProviderStatus.HEALTHY:
                    score += 50
                elif health.status == ProviderStatus.DEGRADED:
                    score += 10

                candidates.append((provider, score))

            if not candidates:
                return None

            # Return provider with highest score
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

    def get_health_report(self) -> dict:
        """Get comprehensive health report for all providers."""
        with self.lock:
            report = {}
            for provider, health in self.health.items():
                backoff_time = 0.0
                if provider in self.backoff:
                    backoff_time = max(0, self.backoff[provider] - time.time())

                bucket = self.buckets.get(provider)
                available_tokens = bucket.tokens if bucket else 0

                report[provider] = {
                    "status": health.status.value,
                    "success_rate": f"{health.success_rate * 100:.1f}%",
                    "total_requests": health.total_requests,
                    "failed_requests": health.failed_requests,
                    "rate_limited": health.rate_limited_requests,
                    "consecutive_failures": health.consecutive_failures,
                    "avg_response_time_ms": f"{health.avg_response_time * 1000:.0f}",
                    "available_tokens": f"{available_tokens:.1f}",
                    "backoff_seconds": f"{backoff_time:.0f}",
                    "is_healthy": health.is_healthy,
                }

            return report


# Global instance
_rate_limiter = EnhancedRateLimiter()


def get_rate_limiter() -> EnhancedRateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


# Convenience functions
def rate_limited_request(provider: str, func: Callable, *args, **kwargs):
    """Make a rate-limited request."""
    return _rate_limiter.request(provider, func, *args, **kwargs)


def get_best_provider(providers: list[str]) -> str | None:
    """Get the best available provider."""
    return _rate_limiter.get_best_provider(providers)


def get_rate_limit_health() -> dict:
    """Get rate limiting health report."""
    return _rate_limiter.get_health_report()
