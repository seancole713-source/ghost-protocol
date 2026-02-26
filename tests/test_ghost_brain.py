#!/usr/bin/env python3
"""
Tests for Ghost Brain v2 — the centralized learning intelligence.

Tests all 7 cognitive abilities:
  1. INVERT   — flip reliably-wrong symbols
  2. SCALE    — adjust confidence to match reality
  3. BIAS     — detect directional bias
  4. TIER     — classify performance tiers
  5. GUARD    — prevent correlated bets
  6. DECAY    — weight recent data heavier (architecture hook)
  7. REPORT   — honest self-assessment
"""

import os
import sys
import pytest
from unittest.mock import patch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ghost_brain import (
    GhostBrain,
    BrainDecision,
    BRAIN_ENABLED,
    INVERT_BELOW,
    EXCLUDE_BELOW,
    BOOST_ABOVE,
    STRONG_BOOST_ABOVE,
    MIN_SAMPLES,
    BOOST_MULT,
    STRONG_BOOST_MULT,
    NOISE_PENALTY_MULT,
    CONFIDENCE_CAP,
    BIAS_THRESHOLD,
    MAX_SAME_DIRECTION,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_accuracy(accuracy_pct: float, total: int = 50) -> dict:
    """Helper to build accuracy data for one symbol."""
    return {
        "accuracy_pct": accuracy_pct,
        "total": total,
        "correct": int(accuracy_pct * total / 100),
    }


from typing import Dict


# =============================================================================
# TEST: ABILITY 1 — INVERT (reliably wrong → flip direction)
# =============================================================================

class TestBrainInvert:
    """Test that symbols with <38% accuracy get inverted."""

    def test_invert_30pct_accuracy(self):
        """30% raw → should flip UP→DOWN, effective 70%."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "BTC", "UP", 0.80,
            {"BTC": make_accuracy(30.0, 50)},
        )
        assert decision.inverted is True
        assert decision.direction == "DOWN"
        assert decision.action == "INVERT"
        assert decision.raw_accuracy == 30.0
        assert decision.effective_accuracy == 70.0
        assert decision.tier == "🔄INVERTED"

    def test_invert_10pct_accuracy(self):
        """10% raw → very reliably wrong, should invert with strong boost."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "ETH", "DOWN", 0.75,
            {"ETH": make_accuracy(10.0, 100)},
        )
        assert decision.inverted is True
        assert decision.direction == "UP"
        assert decision.effective_accuracy == 90.0
        # Should get strong boost (90% effective > 70% threshold)
        assert decision.confidence > 0.75

    def test_invert_36pct_accuracy(self):
        """36.7% (Ghost's actual average) → should invert."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "TURBO", "UP", 0.82,
            {"TURBO": make_accuracy(36.7, 200)},
        )
        assert decision.inverted is True
        assert decision.direction == "DOWN"
        assert decision.effective_accuracy == pytest.approx(63.3, abs=0.1)

    def test_no_invert_above_threshold(self):
        """50% accuracy → NOT inverted (above INVERT_BELOW)."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "SOL", "UP", 0.80,
            {"SOL": make_accuracy(50.0, 50)},
        )
        assert decision.inverted is False
        assert decision.direction == "UP"

    def test_invert_preserves_down_direction(self):
        """DOWN prediction with low accuracy → flips to UP."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "XRP", "DOWN", 0.80,
            {"XRP": make_accuracy(25.0, 60)},
        )
        assert decision.inverted is True
        assert decision.direction == "UP"


# =============================================================================
# TEST: ABILITY 2 — CONFIDENCE SCALING
# =============================================================================

class TestBrainScale:
    """Test that confidence is scaled based on accuracy tier."""

    def test_strong_boost_for_high_accuracy(self):
        """75% accuracy → strong boost (×1.30)."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "PANW", "UP", 0.80,
            {"PANW": make_accuracy(75.0, 50)},
        )
        assert decision.confidence == pytest.approx(
            min(CONFIDENCE_CAP, 0.80 * STRONG_BOOST_MULT), abs=0.01
        )
        assert decision.tier == "🟢HOT"

    def test_boost_for_good_accuracy(self):
        """65% accuracy → boost (×1.15)."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "NET", "DOWN", 0.80,
            {"NET": make_accuracy(65.0, 50)},
        )
        assert decision.confidence == pytest.approx(
            min(CONFIDENCE_CAP, 0.80 * BOOST_MULT), abs=0.01
        )

    def test_penalty_for_cold_accuracy(self):
        """52% accuracy → penalty (×0.85)."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "DOGE", "UP", 0.80,
            {"DOGE": make_accuracy(52.0, 50)},
        )
        assert decision.confidence == pytest.approx(0.80 * NOISE_PENALTY_MULT, abs=0.01)
        assert decision.tier == "🔴COLD"

    def test_no_adjustment_for_warm(self):
        """58% accuracy → no adjustment."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "LINK", "UP", 0.80,
            {"LINK": make_accuracy(58.0, 50)},
        )
        assert decision.confidence == 0.80
        assert decision.tier == "🟡WARM"

    def test_confidence_cap(self):
        """Confidence should never exceed CONFIDENCE_CAP."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "FIRE", "UP", 0.95,
            {"FIRE": make_accuracy(80.0, 100)},
        )
        assert decision.confidence <= CONFIDENCE_CAP

    def test_inverted_symbol_gets_boost(self):
        """Inverted symbol with high effective accuracy gets boosted."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "BTC", "UP", 0.80,
            {"BTC": make_accuracy(15.0, 100)},  # 85% effective
        )
        assert decision.inverted is True
        assert decision.effective_accuracy == 85.0
        # Should be boosted (85% > STRONG_BOOST_ABOVE)
        assert decision.confidence > 0.80


# =============================================================================
# TEST: EXCLUDE (noise zone)
# =============================================================================

class TestBrainExclude:
    """Test that symbols in the noise zone (38-48%) get excluded."""

    def test_exclude_42pct(self):
        """42% accuracy → noise zone, should exclude."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "ADA", "UP", 0.80,
            {"ADA": make_accuracy(42.0, 50)},
        )
        assert decision.action == "EXCLUDE"
        assert decision.tier == "⛔EXCLUDED"

    def test_exclude_45pct(self):
        """45% accuracy → still noise zone (below 48%)."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "DOT", "DOWN", 0.80,
            {"DOT": make_accuracy(45.0, 50)},
        )
        assert decision.action == "EXCLUDE"

    def test_not_excluded_at_49pct(self):
        """49% accuracy → above noise zone, should be sent."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "AVAX", "UP", 0.80,
            {"AVAX": make_accuracy(49.0, 50)},
        )
        assert decision.action == "SEND"
        assert decision.tier == "🔴COLD"  # But penalized

    def test_boundary_38pct_inverts_not_excludes(self):
        """37.9% → just below INVERT_BELOW, should invert not exclude."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "TEST", "UP", 0.80,
            {"TEST": make_accuracy(37.9, 50)},
        )
        assert decision.action == "INVERT"
        assert decision.inverted is True


# =============================================================================
# TEST: INSUFFICIENT DATA
# =============================================================================

class TestBrainInsufficientData:
    """Test that the brain passes through when data is insufficient."""

    def test_no_data(self):
        """No accuracy data for symbol → pass through neutral."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "NEW", "UP", 0.80, {},
        )
        assert decision.action == "SEND"
        assert decision.tier == "⚪NEUTRAL"
        assert decision.inverted is False
        assert decision.direction == "UP"
        assert decision.confidence == 0.80

    def test_too_few_predictions(self):
        """Only 5 predictions → not enough data, pass through."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "FEW", "DOWN", 0.75,
            {"FEW": make_accuracy(20.0, 5)},
        )
        assert decision.action == "SEND"
        assert decision.tier == "⚪NEUTRAL"
        assert decision.inverted is False  # Don't invert without enough data

    def test_exactly_min_samples(self):
        """Exactly MIN_SAMPLES predictions → brain should act."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "EDGE", "UP", 0.80,
            {"EDGE": make_accuracy(30.0, MIN_SAMPLES)},
        )
        # With MIN_SAMPLES and 30% accuracy, brain should invert
        assert decision.action == "INVERT"
        assert decision.inverted is True


# =============================================================================
# TEST: ABILITY 3 — DIRECTION BIAS DETECTION
# =============================================================================

class TestBrainBias:
    """Test direction bias detection."""

    def test_detect_bullish_bias(self):
        """90% UP predictions → should detect bullish bias."""
        brain = GhostBrain()
        predictions = {}
        for i in range(9):
            predictions[f"BULL{i}"] = {"direction": "UP", "confidence": 0.80}
        predictions["BEAR0"] = {"direction": "DOWN", "confidence": 0.80}

        bias = brain._detect_direction_bias(predictions)
        assert bias["biased"] is True
        assert bias["direction"] == "UP"
        assert bias["pct"] == 0.9

    def test_detect_bearish_bias(self):
        """85% DOWN predictions → should detect bearish bias."""
        brain = GhostBrain()
        predictions = {}
        for i in range(17):
            predictions[f"BEAR{i}"] = {"direction": "DOWN", "confidence": 0.80}
        for i in range(3):
            predictions[f"BULL{i}"] = {"direction": "UP", "confidence": 0.80}

        bias = brain._detect_direction_bias(predictions)
        assert bias["biased"] is True
        assert bias["direction"] == "DOWN"

    def test_no_bias_balanced(self):
        """50/50 split → no bias."""
        brain = GhostBrain()
        predictions = {}
        for i in range(5):
            predictions[f"UP{i}"] = {"direction": "UP", "confidence": 0.80}
            predictions[f"DOWN{i}"] = {"direction": "DOWN", "confidence": 0.80}

        bias = brain._detect_direction_bias(predictions)
        assert bias["biased"] is False

    def test_no_bias_empty(self):
        """Empty predictions → no bias."""
        brain = GhostBrain()
        bias = brain._detect_direction_bias({})
        assert bias["biased"] is False


# =============================================================================
# TEST: ABILITY 4 — TIER CLASSIFICATION
# =============================================================================

class TestBrainTiers:
    """Test the full tier spectrum."""

    def test_full_tier_spectrum(self):
        """Each accuracy range maps to the correct tier."""
        brain = GhostBrain()
        test_cases = [
            (10.0, "🔄INVERTED"),   # < 38%
            (30.0, "🔄INVERTED"),   # < 38%
            (40.0, "⛔EXCLUDED"),   # 38-48%
            (45.0, "⛔EXCLUDED"),   # 38-48%
            (50.0, "🔴COLD"),       # 48-55%
            (53.0, "🔴COLD"),       # 48-55%
            (58.0, "🟡WARM"),       # 55-62%
            (60.0, "🟡WARM"),       # 55-62%
            (65.0, "🟢HOT"),        # 62-70%
            (75.0, "🟢HOT"),        # 70%+
            (90.0, "🟢HOT"),        # 70%+ (strong boost)
        ]

        for accuracy, expected_tier in test_cases:
            decision = brain.analyze_symbol(
                f"T{accuracy:.0f}", "UP", 0.80,
                {f"T{accuracy:.0f}": make_accuracy(accuracy, 50)},
            )
            assert decision.tier == expected_tier, (
                f"accuracy={accuracy}% → expected {expected_tier}, got {decision.tier}"
            )


# =============================================================================
# TEST: ABILITY 5 — CORRELATION GUARD
# =============================================================================

class TestBrainCorrelation:
    """Test that correlated bets get penalized."""

    def test_too_many_crypto_up(self):
        """8 crypto UP picks → weakest should get penalized."""
        brain = GhostBrain()
        accuracy_data = {}
        predictions = {}

        # Create 8 crypto symbols all predicting UP with good accuracy
        crypto_symbols = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOT"]
        for sym in crypto_symbols:
            predictions[sym] = {"direction": "UP", "confidence": 0.80}
            accuracy_data[sym] = make_accuracy(60.0, 50)

        decisions = brain.analyze_batch(predictions, accuracy_data)

        # Should have correlation warnings
        assert len(brain._correlation_warnings) > 0

    def test_no_penalty_under_limit(self):
        """5 crypto UP picks (under limit) → no penalty."""
        brain = GhostBrain()
        accuracy_data = {}
        predictions = {}

        crypto_symbols = ["BTC", "ETH", "SOL", "XRP", "ADA"]
        for sym in crypto_symbols:
            predictions[sym] = {"direction": "UP", "confidence": 0.80}
            accuracy_data[sym] = make_accuracy(60.0, 50)

        decisions = brain.analyze_batch(predictions, accuracy_data)
        assert len(brain._correlation_warnings) == 0


# =============================================================================
# TEST: ABILITY 7 — SELF-ASSESSMENT REPORT
# =============================================================================

class TestBrainReport:
    """Test report generation."""

    def test_report_has_key_sections(self):
        """Report should include all key sections."""
        brain = GhostBrain()
        accuracy_data = {
            "BTC": make_accuracy(30.0, 50),   # Will invert
            "ETH": make_accuracy(75.0, 50),   # Will boost
            "SOL": make_accuracy(42.0, 50),   # Will exclude
        }
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.80},
            "ETH": {"direction": "UP", "confidence": 0.80},
            "SOL": {"direction": "DOWN", "confidence": 0.80},
        }

        brain.analyze_batch(predictions, accuracy_data)
        report = brain.generate_report()

        assert "GHOST BRAIN REPORT" in report
        assert "Inverted" in report
        assert "Excluded" in report
        assert "BTC" in report
        assert "ETH" in report
        assert "SOL" in report

    def test_telegram_summary_compact(self):
        """Telegram summary should be compact one-liner."""
        brain = GhostBrain()
        accuracy_data = {
            "BTC": make_accuracy(30.0, 50),
            "ETH": make_accuracy(75.0, 50),
        }
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.80},
            "ETH": {"direction": "UP", "confidence": 0.80},
        }

        brain.analyze_batch(predictions, accuracy_data)
        summary = brain.generate_telegram_summary()

        assert "🧠 Brain:" in summary
        # Should be one line
        assert "\n" not in summary

    def test_health_endpoint(self):
        """Health endpoint should return structured data."""
        brain = GhostBrain()
        accuracy_data = {
            "BTC": make_accuracy(30.0, 50),
            "ETH": make_accuracy(75.0, 50),
        }
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.80},
            "ETH": {"direction": "UP", "confidence": 0.80},
        }

        brain.analyze_batch(predictions, accuracy_data)
        health = brain.get_health()

        assert health["status"] == "active"
        assert health["enabled"] is True
        assert health["decisions"] == 2
        assert "accuracy_lift" in health
        assert "tiers" in health


# =============================================================================
# TEST: BRAIN DISABLED
# =============================================================================

class TestBrainDisabled:
    """Test that brain passes through cleanly when disabled."""

    @patch("core.ghost_brain.BRAIN_ENABLED", False)
    def test_disabled_passes_through(self):
        """When disabled, brain should return raw values unchanged."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "BTC", "UP", 0.80,
            {"BTC": make_accuracy(20.0, 100)},
        )
        assert decision.action == "SEND"
        assert decision.direction == "UP"
        assert decision.confidence == 0.80
        assert decision.inverted is False
        assert decision.tier == "⚪NEUTRAL"


# =============================================================================
# TEST: CYCLE STATS TRACKING
# =============================================================================

class TestBrainStats:
    """Test that the brain tracks its decisions correctly."""

    def test_batch_stats(self):
        """Batch analysis should produce correct cycle stats."""
        brain = GhostBrain()
        accuracy_data = {
            "BTC": make_accuracy(20.0, 50),   # INVERT
            "ETH": make_accuracy(20.0, 50),   # INVERT
            "SOL": make_accuracy(42.0, 50),   # EXCLUDE
            "XRP": make_accuracy(75.0, 50),   # BOOST (HOT)
            "ADA": make_accuracy(58.0, 50),   # WARM
            "NEW": make_accuracy(30.0, 5),    # insufficient data → NEUTRAL
        }
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.80},
            "ETH": {"direction": "DOWN", "confidence": 0.75},
            "SOL": {"direction": "UP", "confidence": 0.80},
            "XRP": {"direction": "UP", "confidence": 0.80},
            "ADA": {"direction": "DOWN", "confidence": 0.70},
            "NEW": {"direction": "UP", "confidence": 0.80},
        }

        brain.analyze_batch(predictions, accuracy_data)

        assert brain._cycle_stats["inverted"] == 2
        assert brain._cycle_stats["excluded"] == 1
        assert brain._cycle_stats["boosted"] >= 1  # XRP + inverted boosts
        assert brain._cycle_stats["sent"] >= 4  # BTC, ETH, XRP, ADA, NEW

    def test_batch_resets_stats(self):
        """Second batch call should reset stats."""
        brain = GhostBrain()
        accuracy_data = {"BTC": make_accuracy(20.0, 50)}
        predictions = {"BTC": {"direction": "UP", "confidence": 0.80}}

        brain.analyze_batch(predictions, accuracy_data)
        assert brain._cycle_stats["inverted"] == 1

        # Second batch should reset
        brain.analyze_batch({}, {})
        assert brain._cycle_stats["inverted"] == 0


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestBrainEdgeCases:
    """Test edge cases and robustness."""

    def test_case_insensitive_lookup(self):
        """Should find accuracy data regardless of case."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "btc", "UP", 0.80,
            {"BTC": make_accuracy(30.0, 50)},
        )
        assert decision.inverted is True

    def test_zero_accuracy(self):
        """0% accuracy → should invert (100% effective)."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "ZERO", "UP", 0.80,
            {"ZERO": make_accuracy(0.0, 50)},
        )
        assert decision.inverted is True
        assert decision.effective_accuracy == 100.0

    def test_100_accuracy(self):
        """100% accuracy → strongest boost."""
        brain = GhostBrain()
        decision = brain.analyze_symbol(
            "PERF", "UP", 0.80,
            {"PERF": make_accuracy(100.0, 50)},
        )
        assert decision.tier == "🟢HOT"
        assert decision.confidence > 0.80

    def test_invalid_direction_in_batch(self):
        """Predictions with invalid direction should be skipped."""
        brain = GhostBrain()
        predictions = {
            "BTC": {"direction": "SIDEWAYS", "confidence": 0.80},
            "ETH": {"direction": "UP", "confidence": 0.80},
        }
        accuracy_data = {
            "BTC": make_accuracy(50.0, 50),
            "ETH": make_accuracy(50.0, 50),
        }

        decisions = brain.analyze_batch(predictions, accuracy_data)
        # BTC should be skipped (invalid direction)
        assert "BTC" not in decisions
        assert "ETH" in decisions

    def test_non_dict_prediction_skipped(self):
        """Non-dict predictions should be skipped gracefully."""
        brain = GhostBrain()
        predictions = {
            "BAD": "not a dict",
            "ETH": {"direction": "UP", "confidence": 0.80},
        }
        accuracy_data = {"ETH": make_accuracy(50.0, 50)}

        decisions = brain.analyze_batch(predictions, accuracy_data)
        assert "BAD" not in decisions
        assert "ETH" in decisions
