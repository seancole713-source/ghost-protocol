"""
API Endpoint Integration Tests (Phase 6.2)

Tests all major API endpoints for proper response structure and data integrity.

Ghost Protocol v5 — Session 6
"""

import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app
from wolf_app import APP

client = TestClient(APP)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_endpoint(self):
        """Test /health endpoint returns valid response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "health" in data
    
    def test_integrity_audit(self):
        """Test /integrity/audit/readonly endpoint."""
        response = client.get("/integrity/audit/readonly")
        assert response.status_code == 200
        data = response.json()
        assert "health_score" in data or "score" in data
        assert "errors" in data or "error_count" in data


class TestPredictionEndpoints:
    """Test prediction-related endpoints."""
    
    def test_picks_latest(self):
        """Test /api/picks/latest endpoint."""
        response = client.get("/api/picks/latest")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict) or isinstance(data, list)
    
    def test_hunter_feed(self):
        """Test /api/v3/hunter/feed endpoint."""
        response = client.get("/api/v3/hunter/feed")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data or "feed" in data


class TestQualityEndpoints:
    """Test quality monitoring endpoints (Phase 4.2, 4.3, 5.6)."""
    
    def test_quality_diversity(self):
        """Test /api/quality/diversity endpoint."""
        response = client.get("/api/quality/diversity")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert "diversity_score" in data
        assert "up_pct" in data
        assert "down_pct" in data
    
    def test_quality_duplicates(self):
        """Test /api/quality/duplicates endpoint."""
        response = client.get("/api/quality/duplicates")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert "duplicate_count" in data
    
    def test_quality_scheduling(self):
        """Test /api/quality/scheduling endpoint."""
        response = client.get("/api/quality/scheduling")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert "consistency_score" in data
    
    def test_quality_summary(self):
        """Test /api/quality/summary endpoint."""
        response = client.get("/api/quality/summary")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data
        assert "overall_quality_score" in data
        assert "diversity" in data
        assert "duplicates" in data
        assert "scheduling" in data


class TestHistoryEndpoints:
    """Test history and tracking endpoints."""
    
    def test_history_api(self):
        """Test /api/history endpoint."""
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
    
    def test_accuracy_api(self):
        """Test /api/v3/accuracy endpoint."""
        response = client.get("/api/v3/accuracy")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data or "accuracy" in data


class TestNewsEndpoints:
    """Test news-related endpoints."""
    
    def test_news_feed(self):
        """Test /api/v3/news/feed endpoint."""
        response = client.get("/api/v3/news/feed?limit=5")
        # May return 429 (rate limited) or 200 (success)
        assert response.status_code in [200, 429]
        if response.status_code == 200:
            data = response.json()
            assert "ok" in data or "items" in data


class TestSubsystemEndpoints:
    """Test AI subsystem endpoints."""
    
    def test_subsystems_status(self):
        """Test /api/subsystems/status endpoint."""
        response = client.get("/api/subsystems/status")
        assert response.status_code == 200
        data = response.json()
        assert "subsystems" in data or "intelligence" in data
    
    def test_heartbeat(self):
        """Test /debug/heartbeat endpoint."""
        response = client.get("/debug/heartbeat")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data or "heartbeat" in data


class TestWatchlistEndpoints:
    """Test watchlist endpoints."""
    
    def test_watchlist_stocks(self):
        """Test /api/watchlist/stocks endpoint."""
        response = client.get("/api/watchlist/stocks")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))
    
    def test_watchlist_crypto(self):
        """Test /api/watchlist/crypto endpoint."""
        response = client.get("/api/watchlist/crypto")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
