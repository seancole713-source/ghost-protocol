#!/usr/bin/env python3
"""
Ghost Protocol Personal Watchlist - Test Suite
==============================================

Comprehensive tests for personal watchlist module.

Test Categories:
1. Unit Tests: Core manager CRUD operations
2. Integration Tests: API endpoints + prediction integration
3. Scheduler Tests: Daily/intraday prediction scheduling
4. Alert Tests: Telegram alert formatting and cooldowns

Usage:
    python3 -m pytest tests/test_personal_watchlist.py -v
    python3 -m pytest tests/test_personal_watchlist.py::test_add_remove_symbols -v
"""

import os
import sys
import time
from unittest.mock import Mock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def watchlist_manager():
    """Create PersonalWatchlistManager instance for testing."""
    from core.personal_watchlist import PersonalWatchlistManager

    # Use test database or mock
    manager = PersonalWatchlistManager()
    yield manager

    # Cleanup: remove all test symbols
    try:
        with manager.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ghost_watchlist_items WHERE symbol LIKE 'TEST%'")
            conn.commit()
    except Exception:
        pass


@pytest.fixture
def mock_prediction_store():
    """Mock prediction_store for testing enrichment."""
    with patch("core.personal_watchlist.get_prediction_store") as mock:
        mock_store = Mock()
        mock_store.get_latest_prediction.return_value = {
            "id": 999,
            "direction": "UP",
            "confidence": 0.75,
            "expected_move_pct": 5.2,
            "horizon_h": 48,
            "run_at": time.time(),
            "current_price": 150.00,
        }
        mock.return_value = mock_store
        yield mock_store


# ============================================================================
# UNIT TESTS: CRUD OPERATIONS
# ============================================================================


def test_add_symbol_stock(watchlist_manager):
    """Test adding a stock symbol to watchlist."""
    result = watchlist_manager.add_symbol(symbol="TESTAAPL", asset_type="stock", owns_position=False, notes="Test Apple")

    assert result["ok"] is True
    assert result["action"] in ("added", "updated")
    assert result["symbol"] == "TESTAAPL"
    assert result["asset_type"] == "stock"
    assert "id" in result


def test_add_symbol_crypto(watchlist_manager):
    """Test adding a crypto symbol to watchlist."""
    result = watchlist_manager.add_symbol(symbol="TESTBTC", asset_type="crypto", owns_position=True, alert_threshold_pct=7.0, priority=3)

    assert result["ok"] is True
    assert result["symbol"] == "TESTBTC"
    assert result["asset_type"] == "crypto"


def test_add_duplicate_symbol(watchlist_manager):
    """Test adding duplicate symbol (should update existing)."""
    # Add first time
    result1 = watchlist_manager.add_symbol(symbol="TESTDUP", asset_type="stock")
    assert result1["ok"] is True
    assert result1["action"] == "added"

    # Add again (should update)
    result2 = watchlist_manager.add_symbol(symbol="TESTDUP", asset_type="stock", owns_position=True)
    assert result2["ok"] is True
    assert result2["action"] == "updated"


def test_add_invalid_asset_type(watchlist_manager):
    """Test adding symbol with invalid asset_type."""
    result = watchlist_manager.add_symbol(symbol="TESTINV", asset_type="invalid_type")
    assert result["ok"] is False
    assert "Invalid asset_type" in result["error"]


def test_remove_symbol(watchlist_manager):
    """Test removing symbol from watchlist (soft delete)."""
    # Add symbol first
    watchlist_manager.add_symbol(symbol="TESTREM", asset_type="stock")

    # Remove it
    result = watchlist_manager.remove_symbol(symbol="TESTREM", asset_type="stock")
    assert result["ok"] is True
    assert result["symbol"] == "TESTREM"

    # Verify it's not in active watchlist
    watchlist = watchlist_manager.get_watchlist(active_only=True)
    assert not any(item["symbol"] == "TESTREM" for item in watchlist)


def test_remove_nonexistent_symbol(watchlist_manager):
    """Test removing symbol that doesn't exist."""
    result = watchlist_manager.remove_symbol(symbol="NOTEXIST", asset_type="stock")
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_get_watchlist(watchlist_manager):
    """Test retrieving all watchlist symbols."""
    # Add multiple symbols
    watchlist_manager.add_symbol(symbol="TESTGET1", asset_type="stock")
    watchlist_manager.add_symbol(symbol="TESTGET2", asset_type="crypto")

    # Get watchlist
    watchlist = watchlist_manager.get_watchlist(active_only=True)
    symbols = [item["symbol"] for item in watchlist]

    assert "TESTGET1" in symbols
    assert "TESTGET2" in symbols


def test_get_symbols_by_type(watchlist_manager):
    """Test filtering watchlist by asset type."""
    # Add symbols of different types
    watchlist_manager.add_symbol(symbol="TESTTYPE1", asset_type="stock")
    watchlist_manager.add_symbol(symbol="TESTTYPE2", asset_type="crypto")

    # Get stocks only
    stocks = watchlist_manager.get_symbols_by_type("stock")
    assert "TESTTYPE1" in stocks
    assert "TESTTYPE2" not in stocks

    # Get crypto only
    crypto = watchlist_manager.get_symbols_by_type("crypto")
    assert "TESTTYPE2" in crypto
    assert "TESTTYPE1" not in crypto


def test_update_position_flag(watchlist_manager):
    """Test updating owns_position flag."""
    # Add symbol
    watchlist_manager.add_symbol(symbol="TESTPOS", asset_type="stock", owns_position=False)

    # Update to owned
    result = watchlist_manager.update_position_flag(symbol="TESTPOS", asset_type="stock", owns_position=True)
    assert result["ok"] is True
    assert result["owns_position"] is True

    # Verify in database
    watchlist = watchlist_manager.get_watchlist(active_only=True)
    item = next((x for x in watchlist if x["symbol"] == "TESTPOS"), None)
    assert item is not None
    assert item["owns_position"] is True


# ============================================================================
# UNIT TESTS: ENRICHMENT
# ============================================================================


def test_get_enriched_watchlist(watchlist_manager, mock_prediction_store):
    """Test getting enriched watchlist with predictions."""
    # Add test symbol
    watchlist_manager.add_symbol(symbol="TESTENR", asset_type="stock")

    # Get enriched watchlist
    enriched = watchlist_manager.get_enriched_watchlist()

    # Find test symbol
    item = next((x for x in enriched if x["symbol"] == "TESTENR"), None)
    assert item is not None
    assert "prediction" in item
    assert item["prediction"]["direction"] == "UP"
    assert item["prediction"]["confidence"] == 0.75


# ============================================================================
# UNIT TESTS: PREDICTION TRACKING
# ============================================================================


def test_track_prediction(watchlist_manager):
    """Test tracking prediction generation event."""
    # Add symbol first
    result = watchlist_manager.add_symbol(symbol="TESTTRACK", asset_type="stock")
    watchlist_item_id = result["id"]

    # Track prediction
    tracking_id = watchlist_manager.track_prediction(
        watchlist_item_id=watchlist_item_id,
        symbol="TESTTRACK",
        prediction_id=12345,
        direction="UP",
        confidence=0.82,
        expected_move_pct=3.5,
        horizon_h=48,
        price_at_prediction=100.50,
        reason="market_open",
    )

    assert tracking_id > 0


def test_get_prediction_history(watchlist_manager):
    """Test retrieving prediction history for symbol."""
    # Add symbol and track prediction
    result = watchlist_manager.add_symbol(symbol="TESTHIST", asset_type="stock")
    watchlist_item_id = result["id"]

    watchlist_manager.track_prediction(
        watchlist_item_id=watchlist_item_id,
        symbol="TESTHIST",
        prediction_id=999,
        direction="DOWN",
        confidence=0.65,
        expected_move_pct=-2.1,
        horizon_h=48,
        price_at_prediction=50.00,
        reason="market_close",
    )

    # Get history
    history = watchlist_manager.get_prediction_history(symbol="TESTHIST", limit=10)

    assert len(history) > 0
    assert history[0]["symbol"] == "TESTHIST"
    assert history[0]["direction"] == "DOWN"
    assert history[0]["reason"] == "market_close"


# ============================================================================
# UNIT TESTS: PRICE SNAPSHOTS & BIG MOVES
# ============================================================================


def test_record_price_snapshot(watchlist_manager):
    """Test recording price snapshot for big-move detection."""
    # Add symbol
    result = watchlist_manager.add_symbol(symbol="TESTSNAP", asset_type="crypto")
    watchlist_item_id = result["id"]

    # Record snapshot
    snapshot_id = watchlist_manager.record_price_snapshot(
        watchlist_item_id=watchlist_item_id, symbol="TESTSNAP", price=25000.00, change_pct_24h=3.2, volume_24h=1000000000
    )

    assert snapshot_id > 0


def test_detect_big_moves(watchlist_manager):
    """Test detecting symbols with significant price moves."""
    # Add symbol
    result = watchlist_manager.add_symbol(symbol="TESTBIG", asset_type="crypto", alert_threshold_pct=5.0)
    watchlist_item_id = result["id"]

    # Record snapshots simulating big move
    watchlist_manager.record_price_snapshot(watchlist_item_id=watchlist_item_id, symbol="TESTBIG", price=100.00, change_pct_24h=0, volume_24h=None)

    # Wait a bit (simulate time passing)
    time.sleep(1)

    watchlist_manager.record_price_snapshot(watchlist_item_id=watchlist_item_id, symbol="TESTBIG", price=107.00, change_pct_24h=7.0, volume_24h=None)  # +7% move

    # Detect big moves
    big_movers = watchlist_manager.detect_big_moves(lookback_minutes=1)

    # Find our test symbol
    mover = next((x for x in big_movers if x["symbol"] == "TESTBIG"), None)
    assert mover is not None
    assert abs(mover["move_pct"]) >= 5.0  # Should exceed threshold


# ============================================================================
# UNIT TESTS: ALERT LOGGING
# ============================================================================


def test_log_alert(watchlist_manager):
    """Test logging Telegram alert."""
    # Add symbol
    result = watchlist_manager.add_symbol(symbol="TESTALERT", asset_type="stock")
    watchlist_item_id = result["id"]

    # Log alert
    alert_id = watchlist_manager.log_alert(
        watchlist_item_id=watchlist_item_id,
        symbol="TESTALERT",
        alert_type="big_move",
        direction="UP",
        confidence=0.78,
        expected_move_pct=4.5,
        current_price=150.00,
        change_pct=6.2,
        message="Test alert message",
        telegram_sent=True,
        telegram_chat_id=12345,
    )

    assert alert_id > 0


def test_check_alert_cooldown(watchlist_manager):
    """Test alert cooldown enforcement."""
    # Add symbol and log alert
    result = watchlist_manager.add_symbol(symbol="TESTCOOL", asset_type="stock")
    watchlist_item_id = result["id"]

    watchlist_manager.log_alert(
        watchlist_item_id=watchlist_item_id,
        symbol="TESTCOOL",
        alert_type="open",
        direction="UP",
        confidence=0.5,
        expected_move_pct=2.0,
        current_price=100.0,
        change_pct=0.0,
        message="Test",
        telegram_sent=True,
    )

    # Check cooldown immediately (should be blocked)
    can_send = watchlist_manager.check_alert_cooldown(symbol="TESTCOOL", alert_type="open", cooldown_hours=4)
    assert can_send is False  # Should be blocked (alert just sent)

    # Check different alert type (should be allowed)
    can_send_other = watchlist_manager.check_alert_cooldown(symbol="TESTCOOL", alert_type="close", cooldown_hours=4)
    assert can_send_other is True  # Different type, no cooldown


def test_get_alert_stats(watchlist_manager):
    """Test retrieving alert statistics."""
    stats = watchlist_manager.get_alert_stats(days=7)

    assert "total" in stats
    assert "by_type" in stats
    assert isinstance(stats["total"], int)
    assert isinstance(stats["by_type"], dict)


# ============================================================================
# INTEGRATION TESTS: API ENDPOINTS
# ============================================================================


@pytest.mark.integration
def test_api_add_symbol():
    """Test POST /api/v3/watchlist/add endpoint."""
    from fastapi.testclient import TestClient
    from wolf_app import APP

    client = TestClient(APP)

    response = client.post(
        "/api/v3/watchlist/add",
        json={"symbol": "APIAPL", "asset_type": "stock", "owns_position": False, "notes": "API test", "alert_threshold_pct": 5.0, "priority": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["symbol"] == "APIAPL"


@pytest.mark.integration
def test_api_get_user_watchlist():
    """Test GET /api/v3/watchlist/user endpoint."""
    from fastapi.testclient import TestClient
    from wolf_app import APP

    client = TestClient(APP)

    response = client.get("/api/v3/watchlist/user")

    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "count" in data
    assert isinstance(data["items"], list)


@pytest.mark.integration
def test_api_remove_symbol():
    """Test POST /api/v3/watchlist/remove endpoint."""
    from fastapi.testclient import TestClient
    from wolf_app import APP

    client = TestClient(APP)

    # Add symbol first
    client.post("/api/v3/watchlist/add", json={"symbol": "APIREM", "asset_type": "stock"})

    # Remove it
    response = client.post("/api/v3/watchlist/remove", json={"symbol": "APIREM", "asset_type": "stock"})

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
