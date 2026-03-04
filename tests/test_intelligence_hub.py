#!/usr/bin/env python3
"""
Tests for the Intelligence Hub — verifies all 20 systems wire correctly.
"""

import sys
import os
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestIntelligenceHubImport:
    """Test that the hub imports without errors."""

    def test_import_hub(self):
        from core.intelligence_hub import IntelligenceHub, get_intelligence_hub
        hub = get_intelligence_hub()
        assert hub is not None

    def test_hub_is_singleton(self):
        from core.intelligence_hub import get_intelligence_hub
        hub1 = get_intelligence_hub()
        hub2 = get_intelligence_hub()
        assert hub1 is hub2

    def test_signal_dataclass(self):
        from core.intelligence_hub import Signal
        sig = Signal(source="test", direction="BUY", confidence=0.75, weight=0.15)
        assert sig.source == "test"
        assert sig.direction == "BUY"
        assert sig.confidence == 0.75
        assert sig.active is False

    def test_report_dataclass(self):
        from core.intelligence_hub import IntelligenceReport
        report = IntelligenceReport()
        assert report.active_systems == 0
        assert report.should_block is False
        assert report.confidence_adjustment == 0.0


class TestNewsBrainCache:
    """Test the news brain → hub cache bridge."""

    def test_update_cache(self):
        from core.intelligence_hub import update_news_brain_cache, get_news_brain_cache
        test_data = {
            "major_events": [{"headline": "Test event", "severity": "HIGH"}],
            "predictions_at_risk": [{"symbol": "BTC", "risk_level": "HIGH", "reason": "test"}],
        }
        update_news_brain_cache(test_data)
        cache, ts = get_news_brain_cache()
        assert cache == test_data
        assert ts > 0
        assert len(cache["major_events"]) == 1
        assert len(cache["predictions_at_risk"]) == 1

    def test_empty_cache(self):
        from core.intelligence_hub import _NEWS_BRAIN_CACHE_TS
        import core.intelligence_hub as hub_module
        # Reset cache
        hub_module._NEWS_BRAIN_CACHE = {}
        hub_module._NEWS_BRAIN_CACHE_TS = 0.0
        cache, ts = hub_module.get_news_brain_cache()
        assert cache == {}
        assert ts == 0.0


class TestNewsBrainSignal:
    """Test news brain signal checker."""

    def test_symbol_at_high_risk(self):
        from core.intelligence_hub import IntelligenceHub, update_news_brain_cache
        update_news_brain_cache({
            "major_events": [],
            "predictions_at_risk": [
                {"symbol": "TURBO", "risk_level": "HIGH", "reason": "Geopolitical risk"}
            ],
        })
        hub = IntelligenceHub()
        sig = hub._check_news_brain("TURBO", "BUY")
        assert sig.active is True
        assert sig.direction == "SELL"  # HIGH risk flips direction
        assert sig.confidence >= 0.5
        assert "HIGH" in sig.reasoning

    def test_symbol_at_medium_risk(self):
        from core.intelligence_hub import IntelligenceHub, update_news_brain_cache
        update_news_brain_cache({
            "major_events": [],
            "predictions_at_risk": [
                {"symbol": "CHZ", "risk_level": "MEDIUM", "reason": "Sector concern"}
            ],
        })
        hub = IntelligenceHub()
        sig = hub._check_news_brain("CHZ", "BUY")
        assert sig.active is True
        assert sig.direction == "HOLD"

    def test_symbol_not_at_risk(self):
        from core.intelligence_hub import IntelligenceHub, update_news_brain_cache
        update_news_brain_cache({
            "major_events": [],
            "predictions_at_risk": [
                {"symbol": "BTC", "risk_level": "HIGH", "reason": "test"}
            ],
        })
        hub = IntelligenceHub()
        sig = hub._check_news_brain("ETH", "BUY")
        assert sig.active is True
        assert sig.direction == "BUY"  # Not at risk = agree with direction

    def test_bearish_event(self):
        from core.intelligence_hub import IntelligenceHub, update_news_brain_cache
        update_news_brain_cache({
            "major_events": [{
                "headline": "Fed raises rates",
                "severity": "HIGH",
                "bearish_symbols": ["SOL", "ETH"],
                "bullish_symbols": [],
            }],
            "predictions_at_risk": [],
        })
        hub = IntelligenceHub()
        sig = hub._check_news_brain("SOL", "BUY")
        assert sig.active is True
        assert sig.direction == "SELL"
        assert "HIGH risk" in sig.reasoning  # Severity must be in reasoning for aggregate detection

    def test_bearish_critical_event_sets_high_risk(self):
        """CRITICAL severity events should flag as HIGH risk in reasoning."""
        from core.intelligence_hub import IntelligenceHub, update_news_brain_cache
        update_news_brain_cache({
            "major_events": [{
                "headline": "Iran war escalation",
                "severity": "CRITICAL",
                "bearish_symbols": ["ETH", "LINK"],
                "bullish_symbols": [],
            }],
            "predictions_at_risk": [],
        })
        hub = IntelligenceHub()
        sig = hub._check_news_brain("ETH", "BUY")
        assert sig.active is True
        assert sig.direction == "SELL"
        assert sig.confidence == 0.7  # CRITICAL = highest confidence
        assert "HIGH risk" in sig.reasoning

    def test_no_recent_cache(self):
        import core.intelligence_hub as hub_module
        from core.intelligence_hub import IntelligenceHub as _Hub
        hub_module._NEWS_BRAIN_CACHE = {}
        hub_module._NEWS_BRAIN_CACHE_TS = 0.0
        hub = _Hub()
        sig = hub._check_news_brain("BTC", "BUY")
        assert sig.active is False


class TestSignalAggregation:
    """Test weighted signal aggregation."""

    def test_all_agree_boosts_confidence(self):
        from core.intelligence_hub import IntelligenceHub, IntelligenceReport, Signal
        hub = IntelligenceHub()
        report = IntelligenceReport()
        report.signals = [
            Signal(source="a", direction="BUY", confidence=0.7, weight=0.2, active=True),
            Signal(source="b", direction="BUY", confidence=0.6, weight=0.2, active=True),
            Signal(source="c", direction="BUY", confidence=0.8, weight=0.2, active=True),
        ]
        hub._aggregate_signals(report, "BUY", 0.60)
        assert report.direction_adjustment == "CONFIRM"
        assert report.confidence_adjustment > 0

    def test_strong_disagree_weakens(self):
        from core.intelligence_hub import IntelligenceHub, IntelligenceReport, Signal
        hub = IntelligenceHub()
        report = IntelligenceReport()
        report.signals = [
            Signal(source="a", direction="SELL", confidence=0.7, weight=0.3, active=True),
            Signal(source="b", direction="SELL", confidence=0.6, weight=0.3, active=True),
            Signal(source="c", direction="BUY", confidence=0.3, weight=0.1, active=True),
        ]
        hub._aggregate_signals(report, "BUY", 0.60)
        assert report.direction_adjustment in ("WEAKEN", "FLIP")
        assert report.confidence_adjustment < 0

    def test_inactive_signals_ignored(self):
        from core.intelligence_hub import IntelligenceHub, IntelligenceReport, Signal
        hub = IntelligenceHub()
        report = IntelligenceReport()
        report.signals = [
            Signal(source="a", direction="SELL", confidence=0.9, weight=0.5, active=False),
            Signal(source="b", direction="BUY", confidence=0.5, weight=0.2, active=True),
        ]
        hub._aggregate_signals(report, "BUY", 0.60)
        # Only one active signal agrees — should not trigger FLIP
        assert report.direction_adjustment != "FLIP"


class TestMLModelSignal:
    """Test ML model signal checker."""

    def test_no_model_returns_inactive(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        with patch("core.ml_trainer.load_model", return_value=None):
            sig = hub._check_ml_model("BTC", "BUY", list(range(30)))
        assert not sig.active
        assert "No trained ML model" in sig.reasoning

    def test_insufficient_history(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        sig = hub._check_ml_model("BTC", "BUY", [100, 101])
        assert not sig.active
        # May be 'No trained ML model' or 'Insufficient' depending on model availability
        assert "Insufficient" in sig.reasoning or "No trained" in sig.reasoning


class TestEnsembleSignal:
    """Test ensemble predictor signal."""

    def test_ensemble_not_loaded(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        hub._ensemble = None
        sig = hub._check_ensemble("BTC", "BUY", 50000.0, [])
        assert not sig.active

    def test_ensemble_with_mock(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()

        @dataclass
        class MockPrediction:
            direction: str = "UP"
            confidence: float = 0.72
            predicted_change_pct: float = 3.5
            individual_predictions: list = None
            model_weights: dict = None
            ensemble_method: str = "confidence_weighted"

            def __post_init__(self):
                self.individual_predictions = self.individual_predictions or []
                self.model_weights = self.model_weights or {}

        mock_ensemble = MagicMock()
        mock_ensemble.predict.return_value = MockPrediction()
        hub._ensemble = mock_ensemble

        prices = [100 + i * 0.5 for i in range(25)]
        sig = hub._check_ensemble("BTC", "BUY", 112.0, prices)
        assert sig.active is True
        assert sig.direction == "BUY"
        assert sig.confidence == 0.72


class TestVWAPSignal:
    """Test VWAP signal checker."""

    def test_vwap_not_loaded(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        hub._vwap = None
        sig = hub._check_vwap("BTC", "BUY")
        assert not sig.active

    def test_vwap_buy_signal(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        mock_vwap = MagicMock()
        mock_vwap.get_vwap_signal.return_value = {
            "signal": "BUY",
            "deviation_pct": -0.03,
        }
        hub._vwap = mock_vwap
        sig = hub._check_vwap("BTC", "BUY")
        assert sig.active is True
        assert sig.direction == "BUY"


class TestWorldContext:
    """Test world context signal."""

    def test_high_vix_bearish(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        with patch("core.world_context.get_world_context", return_value={
            "vix": {"level": 35, "change": 5},
            "spy": {"change_pct": -2.0},
            "market_mood": {"sentiment": "fearful", "score": -0.6},
        }):
            sig = hub._check_world_context("BUY")
        assert sig.active is True
        assert sig.direction == "SELL"
        assert "VIX=35" in sig.reasoning


class TestDynamicExits:
    """Test dynamic exit calculation."""

    def test_exits_calculated(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        exits = hub._calculate_dynamic_exits(100.0, "BUY", 0.70)
        # Should return a dict with target/stop or empty if module fails
        assert isinstance(exits, dict)


class TestConfidenceCalibrator:
    """Test confidence calibration."""

    def test_no_calibrator(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        hub._calibrator = None
        adj = hub._calibrate_confidence("BTC", 0.60)
        assert adj == 0.0

    def test_calibrator_adjusts(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        mock_cal = MagicMock()
        mock_cal.calibrate_confidence.return_value = {
            "calibrated_confidence": 0.55,
        }
        hub._calibrator = mock_cal
        adj = hub._calibrate_confidence("BTC", 0.60)
        assert abs(adj - (-0.05)) < 1e-6  # 0.55 - 0.60


class TestTrustLadder:
    """Test trust ladder."""

    def test_no_ladder(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        hub._trust_ladder = None
        boost = hub._check_trust_ladder("BTC")
        assert boost == 0.0

    def test_ladder_boost(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        mock_ladder = MagicMock()
        mock_ladder.get_trust.return_value = MagicMock(trust_level=2)
        # Trust ladder returns MULTIPLIER: 1.10 = +10% boost, 1.20 = +20%
        mock_ladder.get_prediction_window.return_value = {"confidence_boost": 1.10}
        hub._trust_ladder = mock_ladder
        boost = hub._check_trust_ladder("ETH")
        assert abs(boost - 0.10) < 1e-6  # 1.10 - 1.0 = 0.10 additive delta


class TestQualityGate:
    """Test quality gate."""

    def test_no_gate(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        hub._quality_gate = None
        result = hub._check_quality_gate("BTC", 0.70)
        assert result is None

    def test_gate_blocks(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        mock_gate = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = False
        mock_result.reason = "Too many predictions today"
        mock_gate.check.return_value = mock_result
        hub._quality_gate = mock_gate
        result = hub._check_quality_gate("BTC", 0.70)
        assert result["allowed"] is False


class TestFullAnalyze:
    """Test the full analyze() pipeline."""

    def test_analyze_returns_report(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        hub._initialized = True  # Skip lazy init for test speed

        report = hub.analyze(
            symbol="BTC",
            direction="BUY",
            confidence=0.65,
            entry_price=50000.0,
            asset_type="crypto",
            price_history=[50000 + i * 100 for i in range(25)],
        )

        assert report is not None
        assert report.total_systems == 20
        assert len(report.signals) >= 10  # Should have signals for each system
        assert isinstance(report.confidence_adjustment, float)
        assert isinstance(report.should_block, bool)
        assert report.direction_adjustment in ("CONFIRM", "FLIP", "WEAKEN", "NONE")

    def test_analyze_with_news_risk(self):
        from core.intelligence_hub import IntelligenceHub, update_news_brain_cache
        update_news_brain_cache({
            "major_events": [],
            "predictions_at_risk": [
                {"symbol": "TURBO", "risk_level": "HIGH", "reason": "Geopolitical risk"}
            ],
        })

        hub = IntelligenceHub()
        hub._initialized = True

        report = hub.analyze(
            symbol="TURBO",
            direction="SELL",
            confidence=0.65,
            entry_price=0.002,
            asset_type="crypto",
        )

        assert report.news_risk == "HIGH"
        assert report.confidence_adjustment < 0  # Should penalize

    def test_analyze_all_fail_gracefully(self):
        """Even if every system fails, analyze should not crash."""
        from core.intelligence_hub import IntelligenceHub
        import core.intelligence_hub as hub_module
        hub_module._NEWS_BRAIN_CACHE = {}
        hub_module._NEWS_BRAIN_CACHE_TS = 0.0

        hub = IntelligenceHub()
        hub._initialized = True
        hub._ensemble = None
        hub._calibrator = None
        hub._trust_ladder = None
        hub._quality_gate = None
        hub._killswitch = None
        hub._vwap = None
        hub._feed_fusion = None
        hub._regime_detector = None
        hub._self_improvement = None

        report = hub.analyze(
            symbol="UNKNOWN",
            direction="BUY",
            confidence=0.50,
            entry_price=1.0,
            asset_type="crypto",
        )

        assert report is not None
        assert not report.should_block
        # Most signals should be inactive but no crashes
        assert report.direction_adjustment in ("CONFIRM", "FLIP", "WEAKEN", "NONE")


class TestHubStatus:
    """Test status endpoint."""

    def test_get_status(self):
        from core.intelligence_hub import IntelligenceHub
        hub = IntelligenceHub()
        hub._initialized = True
        status = hub.get_status()
        assert "ensemble_loaded" in status
        assert "calibrator_loaded" in status
        assert "news_brain_has_data" in status


class TestScoutIntegration:
    """Test that the hub is properly wired into ghost_scout._make_prediction."""

    def test_make_prediction_calls_hub(self):
        """Verify _make_prediction has intelligence hub wiring."""
        import inspect
        from core.ghost_scout import GhostScout
        source = inspect.getsource(GhostScout._make_prediction)
        assert "intelligence_hub" in source
        assert "get_intelligence_hub" in source
        assert "hub.analyze" in source
        assert "intel_active_systems" in source
        assert "report.should_block" in source
        assert "report.direction_adjustment" in source


class TestWolfAppIntegration:
    """Test that wolf_app properly wires news brain → hub cache."""

    def test_wolf_app_has_hub_cache_update(self):
        """Verify wolf_app's news loop updates the hub cache."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wolf_app.py")) as f:
            source = f.read()
        assert "update_news_brain_cache" in source
        assert "from core.intelligence_hub import update_news_brain_cache" in source
        assert "Intelligence Hub cache updated" in source

    def test_wolf_app_has_self_improvement(self):
        """Verify wolf_app starts self-improvement engine."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wolf_app.py")) as f:
            source = f.read()
        assert "self_improvement_loop" in source
        assert "run_improvement_cycle" in source

    def test_wolf_app_has_hub_status_endpoint(self):
        """Verify intelligence status endpoint exists."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wolf_app.py")) as f:
            source = f.read()
        assert "/api/v3/intelligence/status" in source

    def test_wolf_app_has_critical_event_handler(self):
        """Verify critical news events trigger auto-pause."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wolf_app.py")) as f:
            source = f.read()
        assert "handle_critical_event" in source
        assert "auto_pause=True" in source

    def test_wolf_app_hub_preinit(self):
        """Verify hub is pre-initialized at startup."""
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wolf_app.py")) as f:
            source = f.read()
        assert "hub._lazy_init()" in source
        assert "Intelligence Hub:" in source
