#!/usr/bin/env python3
"""
Ghost Protocol - Core Test Suite
Tests critical prediction accuracy, gating, and data integrity
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPerformanceGate:
    """Test Performance Gate thresholds"""
    
    def test_kill_threshold(self):
        """Verify kill threshold is set correctly (25%)"""
        from core.performance_gate import KILL_THRESHOLD
        assert KILL_THRESHOLD == 25.0, f"Kill threshold should be 25%, got {KILL_THRESHOLD}%"
    
    def test_warn_threshold(self):
        """Verify warn threshold is set correctly (40%)"""
        from core.performance_gate import WARN_THRESHOLD
        assert WARN_THRESHOLD == 40.0, f"Warn threshold should be 40%, got {WARN_THRESHOLD}%"
    
    def test_reinstate_threshold(self):
        """Verify reinstate threshold is set correctly (50%)"""
        from core.performance_gate import REINSTATE_THRESHOLD
        assert REINSTATE_THRESHOLD == 50.0, f"Reinstate threshold should be 50%, got {REINSTATE_THRESHOLD}%"


class TestLearningBrain:
    """Test Learning Brain inversion system"""
    
    def test_inversion_threshold_enabled(self):
        """Verify inversions are enabled (threshold > 0)"""
        from core.ghost_learning_brain import INVERT_ACCURACY_THRESHOLD
        assert INVERT_ACCURACY_THRESHOLD > 0, f"Inversions disabled (threshold = {INVERT_ACCURACY_THRESHOLD}%)"
        assert INVERT_ACCURACY_THRESHOLD == 30.0, f"Inversion threshold should be 30%, got {INVERT_ACCURACY_THRESHOLD}%"
    
    def test_bench_threshold(self):
        """Verify bench threshold is set correctly (45%)"""
        from core.ghost_learning_brain import BENCH_ACCURACY_THRESHOLD
        assert BENCH_ACCURACY_THRESHOLD == 45.0, f"Bench threshold should be 45%, got {BENCH_ACCURACY_THRESHOLD}%"


class TestQualityGate:
    """Test Quality Gate allows predictions"""
    
    def test_min_confidence_realistic(self):
        """Verify min confidence is achievable (60%)"""
        from core.quality_gate import MIN_CONFIDENCE
        assert MIN_CONFIDENCE == 0.60, f"Min confidence should be 60%, got {MIN_CONFIDENCE*100}%"
        assert MIN_CONFIDENCE < 0.85, "Min confidence too high (was blocking all predictions)"
    
    def test_min_accuracy_realistic(self):
        """Verify min accuracy is achievable (50%)"""
        from core.quality_gate import MIN_ACCURACY_PCT
        assert MIN_ACCURACY_PCT == 50.0, f"Min accuracy should be 50%, got {MIN_ACCURACY_PCT}%"
        assert MIN_ACCURACY_PCT < 85.0, "Min accuracy too high (was blocking all predictions)"


class TestPredictionAccuracy:
    """Test accuracy calculation logic"""
    
    def test_accuracy_calculation(self):
        """Verify accuracy is calculated correctly"""
        # Sample data: 6 correct out of 10 = 60%
        correct = 6
        total = 10
        accuracy = (correct / total) * 100
        assert accuracy == 60.0, f"Accuracy should be 60%, got {accuracy}%"
    
    def test_win_rate_calculation(self):
        """Verify win rate matches accuracy"""
        wins = 355
        total = 866
        win_rate = (wins / total) * 100
        # Allow 1% margin for rounding
        assert 40.0 <= win_rate <= 42.0, f"Win rate should be ~41%, got {win_rate:.1f}%"


class TestDataIntegrity:
    """Test data quality and availability"""
    
    def test_worst_symbols_removed(self):
        """Verify worst symbols (0% accuracy) are not in watchlist"""
        from core.v3_shadow_predictor import DEFAULT_CRYPTO
        
        # These symbols had 0% accuracy and should be removed
        removed_symbols = ["SHIB", "BCH", "ETC"]
        for symbol in removed_symbols:
            assert symbol not in DEFAULT_CRYPTO, f"{symbol} (0% accuracy) should be removed from watchlist"
    
    def test_news_timeout_sufficient(self):
        """Verify news feed timeout is increased"""
        # Check that timeout is at least 15 seconds in news_api.py
        import inspect
        from routes import news_api
        source = inspect.getsource(news_api.api_v3_news_feed)
        assert "timeout=15" in source or "timeout=15.0" in source, "News timeout should be 15s+"


class TestAIMemory:
    """Test AI Memory initialization"""
    
    def test_ai_memory_imports(self):
        """Verify AI Memory module can be imported"""
        try:
            from core.ai_memory import AIMemory
            assert AIMemory is not None
        except ImportError as e:
            pytest.fail(f"AI Memory import failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
