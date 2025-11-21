"""
Base Pillar - Abstract Interface for All Data Pillars
=====================================================

Defines unified interface that all 6 data pillars must implement.

Design Principles:
- Explicit data_available flags (never silently fake data)
- Graceful degradation (fallback to estimates when APIs unavailable)
- Performance tracking (execution_time_ms for all operations)
- Error transparency (capture and return all errors)
- Cache-friendly (support TTL-based caching)

Author: Ghost AI
Date: 2025-01-XX
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataSignal:
    """
    Unified signal structure across all data pillars.
    
    Attributes:
        name: Signal identifier (e.g., "RSI_14", "SPY_PRICE", "NEWS_SENTIMENT")
        value: Signal value (float, str, or None)
        confidence: Confidence score 0.0-1.0 (1.0 = high confidence)
        data_available: True if real data, False if fallback/estimate/unavailable
        source: Data provider (e.g., "polygon", "alpha_vantage", "textblob")
        timestamp: Unix timestamp when data was fetched
        metadata: Additional context (errors, provider details, etc.)
    
    Examples:
        # Real data signal
        DataSignal(
            name="RSI_14",
            value=67.5,
            confidence=1.0,
            data_available=True,
            source="calculated",
            timestamp=1704067200.0,
            metadata={"period": 14, "overbought_threshold": 70}
        )
        
        # Unavailable data signal (honest failure)
        DataSignal(
            name="TWITTER_SENTIMENT",
            value=0.0,
            confidence=0.0,
            data_available=False,
            source="placeholder",
            timestamp=1704067200.0,
            metadata={"error": "TWITTER_BEARER_TOKEN not configured"}
        )
    """

    name: str
    value: float | str | None
    confidence: float  # 0.0-1.0
    data_available: bool
    source: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if self.timestamp <= 0:
            raise ValueError(f"Timestamp must be positive, got {self.timestamp}")


@dataclass
class PillarResponse:
    """
    Unified response from all pillar engines.
    
    Attributes:
        pillar_name: Pillar identifier (e.g., "price_engine", "sentiment_engine")
        symbol: Target symbol for analysis
        signals: List of all signals from this pillar
        errors: List of errors encountered during execution
        execution_time_ms: Performance metric (milliseconds)
        timestamp: Response generation timestamp
        cached: Whether response came from cache
    
    Example:
        PillarResponse(
            pillar_name="price_engine",
            symbol="AAPL",
            signals=[
                DataSignal(name="PRICE", value=150.25, ...),
                DataSignal(name="BID_ASK_SPREAD", value=0.02, ...),
            ],
            errors=[],
            execution_time_ms=45.2,
            timestamp=1704067200.0,
            cached=False
        )
    """

    pillar_name: str
    symbol: str
    signals: list[DataSignal]
    errors: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=lambda: time.time())
    cached: bool = False

    def get_signal(self, name: str) -> DataSignal | None:
        """Get specific signal by name"""
        for signal in self.signals:
            if signal.name == name:
                return signal
        return None

    def has_errors(self) -> bool:
        """Check if pillar encountered any errors"""
        return len(self.errors) > 0

    def signal_count(self) -> int:
        """Count of signals returned"""
        return len(self.signals)

    def available_signal_count(self) -> int:
        """Count of signals with data_available=True"""
        return sum(1 for s in self.signals if s.data_available)


class BasePillar(ABC):
    """
    Abstract base class for all data pillars.
    
    All 6 pillars must inherit from this class and implement:
    - get_signals(symbol) - Main data fetching method
    - get_signal_names() - List of all signals this pillar provides
    - health_check() - Verify pillar can fetch data
    
    Design:
    - Use dependency injection for external services (cache, metrics, etc.)
    - Support graceful degradation (fallback to estimates)
    - Never silently fake data (always set data_available=False for fallbacks)
    - Track performance (execution time, cache hits, errors)
    """

    def __init__(self, pillar_name: str):
        """
        Initialize base pillar.
        
        Args:
            pillar_name: Unique pillar identifier (e.g., "price_engine")
        """
        self.pillar_name = pillar_name
        self._start_time = 0.0

    @abstractmethod
    def get_signals(self, symbol: str, **kwargs) -> PillarResponse:
        """
        Fetch all signals for a given symbol.
        
        Args:
            symbol: Stock/crypto ticker (e.g., "AAPL", "BTC")
            **kwargs: Additional parameters (timeframe, limit, etc.)
        
        Returns:
            PillarResponse with all signals from this pillar
        
        Raises:
            ValueError: If symbol is invalid
        """
        pass

    @abstractmethod
    def get_signal_names(self) -> list[str]:
        """
        Get list of all signal names this pillar provides.
        
        Returns:
            List of signal names (e.g., ["PRICE", "BID_ASK_SPREAD", "VWAP"])
        
        Example:
            price_engine.get_signal_names()
            # ["PRICE", "PREV_CLOSE", "BID", "ASK", "BID_ASK_SPREAD", "VWAP"]
        """
        pass

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """
        Verify pillar can fetch data from providers.
        
        Returns:
            Dict with status, providers, and errors
        
        Example:
            {
                "ok": True,
                "pillar": "price_engine",
                "providers": [
                    {"name": "polygon", "status": "ok", "latency_ms": 45.2},
                    {"name": "yahoo", "status": "ok", "latency_ms": 123.5}
                ],
                "errors": []
            }
        """
        pass

    def _start_timer(self):
        """Start execution timer"""
        self._start_time = time.time()

    def _get_execution_time_ms(self) -> float:
        """Get execution time in milliseconds"""
        if self._start_time == 0.0:
            return 0.0
        elapsed = time.time() - self._start_time
        return round(elapsed * 1000, 2)

    def _create_unavailable_signal(
        self, name: str, error: str, metadata: dict[str, Any] | None = None
    ) -> DataSignal:
        """
        Create signal for unavailable data (honest failure).
        
        Args:
            name: Signal name
            error: Error message explaining why data unavailable
            metadata: Additional context
        
        Returns:
            DataSignal with data_available=False
        """
        meta = metadata or {}
        meta["error"] = error

        return DataSignal(
            name=name,
            value=None,
            confidence=0.0,
            data_available=False,
            source="unavailable",
            timestamp=time.time(),
            metadata=meta,
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} pillar={self.pillar_name}>"
