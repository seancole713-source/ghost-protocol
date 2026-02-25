"""
Pytest fixtures for testing.

Provides reusable test fixtures for predictions, filters, and mocks.
"""
import pytest
from datetime import datetime
from typing import List

from core.models import Prediction, Direction
from core.v3_filter import V3Filter


# =============================================================================
# V3 FILTER FIXTURES
# =============================================================================

@pytest.fixture
def v3_filter():
    """V3 filter with default settings (78% min confidence)."""
    f = V3Filter(min_confidence=0.78)
    f.reset_stats()
    return f


@pytest.fixture
def v3_filter_low_threshold():
    """V3 filter with low confidence threshold for edge case testing."""
    f = V3Filter(min_confidence=0.50)
    f.reset_stats()
    return f


# =============================================================================
# ETH PREDICTION FIXTURES
# =============================================================================

@pytest.fixture
def eth_down_high_conf() -> Prediction:
    """ETH DOWN prediction at 80% confidence - should pass and flip."""
    return Prediction(
        symbol='ETH',
        direction=Direction.DOWN,
        confidence=0.80,
        current_price=2300.0,
        target_price=2200.0,
        stop_loss=2350.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def eth_down_low_conf() -> Prediction:
    """ETH DOWN prediction at 45% confidence - should be rejected."""
    return Prediction(
        symbol='ETH',
        direction=Direction.DOWN,
        confidence=0.45,
        current_price=2300.0,
        target_price=2200.0,
        stop_loss=2350.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def eth_down_exactly_70() -> Prediction:
    """ETH DOWN prediction at exactly 78% confidence - should pass."""
    return Prediction(
        symbol='ETH',
        direction=Direction.DOWN,
        confidence=0.78,
        current_price=2300.0,
        target_price=2200.0,
        stop_loss=2350.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def eth_down_just_below_70() -> Prediction:
    """ETH DOWN prediction at 77.9% confidence - should be rejected."""
    return Prediction(
        symbol='ETH',
        direction=Direction.DOWN,
        confidence=0.779,
        current_price=2300.0,
        target_price=2200.0,
        stop_loss=2350.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def eth_up_high_conf() -> Prediction:
    """ETH UP prediction at 80% confidence - should be rejected (inverse only on DOWN)."""
    return Prediction(
        symbol='ETH',
        direction=Direction.UP,
        confidence=0.80,
        current_price=2300.0,
        target_price=2400.0,
        stop_loss=2250.0,
        timestamp=datetime.now(),
    )


# =============================================================================
# XRP PREDICTION FIXTURES
# =============================================================================

@pytest.fixture
def xrp_up_high_conf() -> Prediction:
    """XRP UP prediction at 80% confidence - should pass."""
    return Prediction(
        symbol='XRP',
        direction=Direction.UP,
        confidence=0.80,
        current_price=1.65,
        target_price=1.75,
        stop_loss=1.58,
        timestamp=datetime.now(),
    )


@pytest.fixture
def xrp_down_high_conf() -> Prediction:
    """XRP DOWN prediction at 80% confidence - should pass (mean_reversion)."""
    return Prediction(
        symbol='XRP',
        direction=Direction.DOWN,
        confidence=0.80,
        current_price=1.65,
        target_price=1.55,
        stop_loss=1.72,
        timestamp=datetime.now(),
    )


@pytest.fixture
def xrp_low_conf() -> Prediction:
    """XRP prediction at 50% confidence - should be rejected."""
    return Prediction(
        symbol='XRP',
        direction=Direction.UP,
        confidence=0.50,
        current_price=1.65,
        target_price=1.75,
        stop_loss=1.58,
        timestamp=datetime.now(),
    )


# =============================================================================
# LINK PREDICTION FIXTURES
# =============================================================================

@pytest.fixture
def link_up_high_conf() -> Prediction:
    """LINK UP prediction at 80% confidence - should pass."""
    return Prediction(
        symbol='LINK',
        direction=Direction.UP,
        confidence=0.80,
        current_price=12.50,
        target_price=13.25,
        stop_loss=12.00,
        timestamp=datetime.now(),
    )


# =============================================================================
# INVALID/REJECTED PREDICTION FIXTURES
# =============================================================================

@pytest.fixture
def btc_down_high_conf() -> Prediction:
    """BTC DOWN prediction (not V3 validated) - should be rejected."""
    return Prediction(
        symbol='BTC',
        direction=Direction.DOWN,
        confidence=0.85,
        current_price=75000.0,
        target_price=72000.0,
        stop_loss=76500.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def sol_down_high_conf() -> Prediction:
    """SOL DOWN prediction (removed from V3) - should be rejected."""
    return Prediction(
        symbol='SOL',
        direction=Direction.DOWN,
        confidence=0.90,
        current_price=120.0,
        target_price=110.0,
        stop_loss=125.0,
        timestamp=datetime.now(),
    )


@pytest.fixture
def blacklisted_prediction() -> Prediction:
    """SHIB prediction (blacklisted) - should be rejected."""
    return Prediction(
        symbol='SHIB',
        direction=Direction.UP,
        confidence=0.95,
        current_price=0.00001,
        target_price=0.000012,
        stop_loss=0.000009,
        timestamp=datetime.now(),
    )


# =============================================================================
# MIXED PREDICTION FIXTURES
# =============================================================================

@pytest.fixture
def sample_predictions(
    eth_down_high_conf,
    xrp_up_high_conf,
    btc_down_high_conf,
) -> List[Prediction]:
    """Mix of valid and invalid predictions."""
    return [eth_down_high_conf, xrp_up_high_conf, btc_down_high_conf]


@pytest.fixture
def all_valid_predictions(
    eth_down_high_conf,
    xrp_up_high_conf,
    link_up_high_conf,
) -> List[Prediction]:
    """All V3 validated predictions that should pass."""
    return [eth_down_high_conf, xrp_up_high_conf, link_up_high_conf]


@pytest.fixture
def all_invalid_predictions(
    btc_down_high_conf,
    sol_down_high_conf,
    blacklisted_prediction,
) -> List[Prediction]:
    """All predictions that should be rejected."""
    return [btc_down_high_conf, sol_down_high_conf, blacklisted_prediction]
