"""
Shared data models used across the application.
Using dataclasses for immutability and type safety.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class Direction(str, Enum):
    """Trade direction."""
    UP = "UP"
    DOWN = "DOWN"
    
    def opposite(self) -> "Direction":
        """Get the opposite direction."""
        return Direction.DOWN if self == Direction.UP else Direction.UP


class TradeOutcome(str, Enum):
    """Possible outcomes for a trade."""
    PENDING = "PENDING"
    WIN = "WIN"
    LOSS = "LOSS"
    STOPPED = "STOPPED"
    EXPIRED = "EXPIRED"
    BREAK_EVEN = "BREAK_EVEN"


@dataclass(frozen=True)
class Prediction:
    """
    Raw prediction from the prediction engine.
    
    This represents the output of the Ghost prediction model
    before any V3 filtering or scoring is applied.
    """
    symbol: str
    direction: Direction
    confidence: float
    current_price: float
    target_price: float
    stop_loss: float
    timestamp: datetime
    news_influenced: bool = False
    asset_type: str = "crypto"
    
    def __post_init__(self):
        """Validate prediction data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if self.current_price <= 0:
            raise ValueError(f"Current price must be positive, got {self.current_price}")


@dataclass(frozen=True)
class ScoredPrediction:
    """
    Prediction after V3 filtering and scoring.
    
    This is what gets sent to users - it includes all V3 metadata
    like strategy type, backtest win rate, and hold period.
    """
    symbol: str
    direction: Direction
    confidence: float
    current_price: float
    target_price: float
    stop_loss: float
    hold_hours: int
    timestamp: datetime
    
    # V3 metadata
    strategy: str
    original_direction: Direction
    is_inverse: bool
    backtest_win_rate: float
    score: float
    news_influenced: bool = False
    asset_type: str = "crypto"
    
    @property
    def is_crypto(self) -> bool:
        """Check if this is a crypto prediction."""
        from config.symbols import is_crypto
        return is_crypto(self.symbol)
    
    @property
    def hold_days(self) -> int:
        """Get hold period in days."""
        return self.hold_hours // 24
    
    @property
    def expected_return_pct(self) -> float:
        """Calculate expected return percentage."""
        return ((self.target_price - self.current_price) / self.current_price) * 100
    
    @property
    def risk_pct(self) -> float:
        """Calculate risk percentage (distance to stop loss)."""
        return abs((self.stop_loss - self.current_price) / self.current_price) * 100


@dataclass
class PaperTrade:
    """
    Logged trade for tracking and validation.
    
    This tracks a prediction from signal to outcome,
    storing all V3 metadata for validation analysis.
    """
    id: Optional[str] = None
    cascade_id: Optional[str] = None
    symbol: str = ""
    direction: Direction = Direction.UP
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0
    entry_time: Optional[datetime] = None
    check_time: Optional[datetime] = None
    hold_hours: int = 72
    confidence: float = 0.5
    
    # V3 fields
    v3_validated: bool = False
    v3_strategy: Optional[str] = None
    v3_is_inverse: bool = False
    v3_original_direction: Optional[Direction] = None
    v3_backtest_win_rate: Optional[float] = None
    
    # Outcome fields (filled when trade resolves)
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    outcome: TradeOutcome = TradeOutcome.PENDING
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    
    @property
    def is_resolved(self) -> bool:
        """Check if trade has been resolved."""
        return self.outcome != TradeOutcome.PENDING
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was a win."""
        return self.outcome == TradeOutcome.WIN


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    reason: Optional[str] = None
    
    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class FilterResult:
    """Result of V3 filter processing."""
    passed: bool
    symbol: str
    reason: str
    prediction: Optional[ScoredPrediction] = None
