"""
Typed tool call adapters for Ghost ChatGPT Analyst.

Provides retry decorators, provider attribution, error handling, and
structured responses for all analyst tool endpoints.
"""

import functools
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

# Type variable for generic decorator
T = TypeVar("T", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def with_retry(
    max_attempts: int = 3, backoff_base: float = 2.0, exceptions: tuple = (Exception,)
) -> Callable[[T], T]:
    """
    Retry decorator with exponential backoff for tool calls.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (seconds)
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)

                    # Log successful retry if not first attempt
                    if attempt > 0:
                        logger.info(
                            "tool_retry_success",
                            extra={
                                "tool": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                            },
                        )

                    return result

                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        wait_time = backoff_base**attempt
                        logger.warning(
                            "tool_retry",
                            extra={
                                "tool": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "error": str(e),
                                "wait_seconds": wait_time,
                            },
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            "tool_max_retries_exceeded",
                            extra={
                                "tool": func.__name__,
                                "attempts": max_attempts,
                                "error": str(e),
                            },
                        )

            # All attempts failed
            raise last_exception or RuntimeError(
                f"{func.__name__} failed after {max_attempts} attempts"
            )

        return cast(T, wrapper)

    return decorator


def with_provider_attribution(provider: str) -> Callable[[T], T]:
    """
    Add provider attribution to tool response metadata.

    Args:
        provider: Name of data provider (e.g., "yfinance", "polygon", "sec_edgar")

    Returns:
        Decorated function that adds provider metadata
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            result = func(*args, **kwargs)

            # Add provider metadata if result is a dict
            if isinstance(result, dict):
                result.setdefault("_meta", {})
                result["_meta"]["provider"] = provider
                result["_meta"]["fetched_at"] = datetime.now(UTC).isoformat()
                result["_meta"]["tool"] = func.__name__

            return result

        return cast(T, wrapper)

    return decorator


def sanitize_response(response: dict[str, Any], max_text_length: int = 5000) -> dict[str, Any]:
    """
    Sanitize tool response to prevent token overflow and remove sensitive data.

    Args:
        response: Raw tool response
        max_text_length: Maximum length for text fields

    Returns:
        Sanitized response
    """
    if not isinstance(response, dict):
        return response

    sanitized = {}

    for key, value in response.items():
        # Truncate long strings
        if isinstance(value, str) and len(value) > max_text_length:
            sanitized[key] = value[:max_text_length] + f"... (truncated from {len(value)} chars)"

        # Recursively sanitize nested dicts
        elif isinstance(value, dict):
            sanitized[key] = sanitize_response(value, max_text_length)

        # Truncate large lists
        elif isinstance(value, list):
            if len(value) > 100:
                sanitized[key] = value[:100]
                logger.warning(
                    "tool_response_truncated",
                    extra={"key": key, "original_length": len(value), "truncated_to": 100},
                )
            else:
                sanitized[key] = value

        else:
            sanitized[key] = value

    return sanitized


class ToolCallError(Exception):
    """Base exception for tool call failures"""

    pass


class ProviderError(ToolCallError):
    """External provider returned an error"""

    pass


class ValidationError(ToolCallError):
    """Input validation failed"""

    pass


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize stock symbol.

    Args:
        symbol: Raw symbol input

    Returns:
        Normalized symbol (uppercase, trimmed)

    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError("Symbol must be a non-empty string")

    normalized = symbol.strip().upper()

    if len(normalized) > 10:
        raise ValidationError(f"Symbol too long: {normalized}")

    if not normalized.replace("-", "").replace(".", "").isalnum():
        raise ValidationError(f"Symbol contains invalid characters: {normalized}")

    return normalized


def validate_lookback(hours: int, min_hours: int = 1, max_hours: int = 8760) -> int:
    """
    Validate lookback period in hours.

    Args:
        hours: Requested lookback period
        min_hours: Minimum allowed (default: 1 hour)
        max_hours: Maximum allowed (default: 1 year)

    Returns:
        Validated hours

    Raises:
        ValidationError: If hours out of range
    """
    if not isinstance(hours, int):
        raise ValidationError(f"Hours must be an integer, got {type(hours)}")

    if hours < min_hours or hours > max_hours:
        raise ValidationError(f"Hours must be between {min_hours} and {max_hours}, got {hours}")

    return hours


class ToolResponse:
    """Structured tool response with consistent format"""

    def __init__(
        self, ok: bool, data: Any = None, error: str | None = None, provider: str | None = None
    ):
        self.ok = ok
        self.data = data
        self.error = error
        self.provider = provider
        self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization"""
        result = {"ok": self.ok, "timestamp": self.timestamp}

        if self.data is not None:
            result["data"] = self.data

        if self.error:
            result["error"] = self.error

        if self.provider:
            result["provider"] = self.provider

        return result

    @classmethod
    def success(cls, data: Any, provider: str | None = None) -> "ToolResponse":
        """Create successful response"""
        return cls(ok=True, data=data, provider=provider)

    @classmethod
    def failure(cls, error: str, provider: str | None = None) -> "ToolResponse":
        """Create error response"""
        return cls(ok=False, error=error, provider=provider)


# Example usage and pre-built tool wrappers


@with_retry(max_attempts=3, backoff_base=1.5)
@with_provider_attribution("internal")
def get_portfolio_snapshot() -> dict[str, Any]:
    """
    Get current portfolio snapshot with retry and attribution.

    Returns:
        Portfolio data with positions, NAV, PnL
    """
    import requests

    try:
        resp = requests.get("http://localhost:5000/api/position", timeout=5)
        resp.raise_for_status()
        position = resp.json()

        return ToolResponse.success(
            {
                "symbol": position.get("symbol"),
                "qty": position.get("qty"),
                "avg_cost": position.get("avg_cost"),
                "has_position": position.get("qty", 0) > 0,
            }
        ).to_dict()

    except Exception as e:
        logger.error(f"portfolio_snapshot_error: {e}")
        return ToolResponse.failure(str(e)).to_dict()


@with_retry(max_attempts=2)
@with_provider_attribution("internal")
def get_regime_current() -> dict[str, Any]:
    """
    Get current market regime with retry.

    Returns:
        Regime state (BULL, BEAR, SIDEWAYS, HIGH_VOL)
    """
    import requests

    try:
        resp = requests.get("http://localhost:5000/api/regime/current", timeout=5)

        if resp.status_code == 403:
            # Auth required but not critical
            return ToolResponse.success({"regime": "UNKNOWN", "note": "auth_required"}).to_dict()

        resp.raise_for_status()
        data = resp.json()

        return ToolResponse.success(
            {
                "regime": data.get("regime", {}).get("regime", "UNKNOWN"),
                "confidence": data.get("regime", {}).get("confidence"),
                "ok": data.get("ok", False),
            }
        ).to_dict()

    except Exception as e:
        logger.warning(f"regime_fetch_error: {e}")
        return ToolResponse.success({"regime": "UNKNOWN", "error": str(e)}).to_dict()


def format_tool_error(error: Exception, tool_name: str) -> dict[str, Any]:
    """
    Format tool error for consistent logging and response.

    Args:
        error: Exception that occurred
        tool_name: Name of tool that failed

    Returns:
        Formatted error dict
    """
    return {
        "ok": False,
        "error": str(error),
        "error_type": type(error).__name__,
        "tool": tool_name,
        "timestamp": datetime.now(UTC).isoformat(),
    }
