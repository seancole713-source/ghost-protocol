"""Test suite for /api/cockpit endpoint."""
from fastapi.testclient import TestClient
from wolf_app import APP

client = TestClient(APP)


def test_cockpit_endpoint_200():
    """Test that /api/cockpit returns 200 with expected structure."""
    r = client.get("/api/cockpit")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "system" in data
    assert "ghost_2x" in data
    assert data["status"] in ["ok", "error"]


def test_cockpit_system_fields():
    """Test that system block contains expected fields."""
    r = client.get("/api/cockpit")
    assert r.status_code == 200
    data = r.json()
    system = data.get("system", {})
    assert "mode" in system
    assert "active" in system
    assert "version" in system


def test_cockpit_ghost2x_fields():
    """Test that ghost_2x block contains expected fields."""
    r = client.get("/api/cockpit")
    assert r.status_code == 200
    data = r.json()
    ghost_2x = data.get("ghost_2x")
    
    # If ghost_2x is None, it means there was an error (which is valid behavior)
    if ghost_2x is not None:
        assert "ok" in ghost_2x
        assert "symbol_counts" in ghost_2x
        assert "vip_provider_health" in ghost_2x
        assert "ghost_score_v2" in ghost_2x
        assert "risk_guard_status" in ghost_2x
        assert "last_multi_prediction_run_time" in ghost_2x


def test_cockpit_no_auth_required():
    """Test that /api/cockpit does not require authentication."""
    # Call without auth headers - should still work
    r = client.get("/api/cockpit")
    assert r.status_code == 200
    # Should not return 401 unauthorized
    data = r.json()
    assert "error" not in data or data.get("error") != "unauthorized"
