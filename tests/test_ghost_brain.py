#!/usr/bin/env python3
"""
🧠 Ghost Brain v3 — Comprehensive Test Suite
=============================================

Tests all 25 cognitive abilities with real-world scenarios.
All tests are in-memory (no database dependency).
"""

import sys
import os
import pytest
from datetime import datetime, timedelta
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ghost_brain import (
    GhostBrain, BrainDecision,
    INVERT_BELOW, EXCLUDE_BELOW, BOOST_ABOVE, STRONG_BOOST_ABOVE,
    MIN_SAMPLES, CONFIDENCE_CAP, NOISE_PENALTY_MULT,
    BOOST_MULT, STRONG_BOOST_MULT, MAX_SAME_DIRECTION,
    BIAS_THRESHOLD, CIRCUIT_BREAKER_THRESHOLD,
)
from core.brain_data import BrainContext, SymbolContext


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def make_accuracy(accuracy_pct: float, total: int = 50) -> Dict:
    """Create old-style accuracy_data entry."""
    correct = int(total * accuracy_pct / 100.0)
    return {"total": total, "correct": correct, "accuracy_pct": accuracy_pct}


def make_sym_ctx(
    accuracy_pct: float = 50.0,
    total: int = 50,
    up_accuracy: float = None,
    down_accuracy: float = None,
    up_total: int = 25,
    down_total: int = 25,
    recent_accuracy: float = None,
    recent_total: int = 30,
    streak: int = 0,
    trust_level: int = 1,
    avg_win_mag: float = 0.0,
    avg_loss_mag: float = 0.0,
    days_tracked: int = 30,
    dow_accuracy: dict = None,
) -> SymbolContext:
    """Create a SymbolContext with sensible defaults."""
    correct = int(total * accuracy_pct / 100.0)
    _up_acc = up_accuracy if up_accuracy is not None else accuracy_pct
    _down_acc = down_accuracy if down_accuracy is not None else accuracy_pct
    _recent_acc = recent_accuracy if recent_accuracy is not None else accuracy_pct
    return SymbolContext(
        total_predictions=total,
        correct_predictions=correct,
        accuracy_pct=accuracy_pct,
        up_total=up_total,
        up_correct=int(up_total * _up_acc / 100.0),
        up_accuracy=_up_acc,
        down_total=down_total,
        down_correct=int(down_total * _down_acc / 100.0),
        down_accuracy=_down_acc,
        recent_total=recent_total,
        recent_correct=int(recent_total * _recent_acc / 100.0),
        recent_accuracy=_recent_acc,
        trust_level=trust_level,
        consecutive_wins=max(0, streak),
        consecutive_losses=max(0, -streak),
        current_streak=streak,
        avg_win_magnitude=avg_win_mag,
        avg_loss_magnitude=avg_loss_mag,
        days_tracked=days_tracked,
        dow_accuracy=dow_accuracy or {},
    )


def make_context(
    symbols: dict = None,
    regime: str = "unknown",
    fear_greed: int = 50,
    btc_24h: float = 0.0,
    spy_24h: float = 0.0,
    is_weekend: bool = False,
    current_day: int = 2,
    rolling_3d_acc: float = 55.0,
    rolling_3d_total: int = 50,
    calibration: dict = None,
) -> BrainContext:
    """Create a BrainContext with sensible defaults."""
    return BrainContext(
        symbols=symbols or {},
        market_regime=regime,
        fear_greed_index=fear_greed,
        btc_24h_change=btc_24h,
        spy_24h_change=spy_24h,
        is_weekend=is_weekend,
        current_day=current_day,
        rolling_3d_accuracy=rolling_3d_acc,
        rolling_3d_total=rolling_3d_total,
        calibration_curve=calibration or {},
    )


# ═══════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY (v2 interface still works)
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompat:
    """Ensure v2-style accuracy_data still works."""

    def test_v2_invert(self):
        brain = GhostBrain()
        accuracy_data = {"BTC": make_accuracy(30.0)}
        d = brain.analyze_symbol("BTC", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.action == "INVERT"
        assert d.direction == "DOWN"
        assert d.inverted is True

    def test_v2_exclude(self):
        brain = GhostBrain()
        accuracy_data = {"ETH": make_accuracy(42.0)}
        d = brain.analyze_symbol("ETH", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.action == "EXCLUDE"

    def test_v2_send_hot(self):
        brain = GhostBrain()
        accuracy_data = {"AAPL": make_accuracy(66.0)}
        d = brain.analyze_symbol("AAPL", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.action == "SEND"
        assert d.tier == "🟢HOT"

    def test_v2_batch(self):
        brain = GhostBrain()
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.72},
            "AAPL": {"direction": "UP", "confidence": 0.72},
        }
        accuracy_data = {
            "BTC": make_accuracy(30.0),
            "AAPL": make_accuracy(66.0),
        }
        results = brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        assert results["BTC"].action == "INVERT"
        assert results["AAPL"].action == "SEND"

    def test_v2_insufficient_data(self):
        brain = GhostBrain()
        accuracy_data = {"NEW": make_accuracy(30.0, total=5)}
        d = brain.analyze_symbol("NEW", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.action == "SEND"
        assert d.tier == "⚪NEUTRAL"
        assert d.confidence == 0.72


# ═══════════════════════════════════════════════════════════════
# #1: PER-DIRECTION ACCURACY
# ═══════════════════════════════════════════════════════════════

class TestDirectionSplit:
    """Test that the brain handles UP vs DOWN accuracy separately."""

    def test_only_down_is_bad(self):
        """UP accuracy is 60% but DOWN is 10% — don't invert UP calls.

        brain_accuracy = 30*0.70 + 60*0.20 + 30*0.10 = 36.0 → INVERT zone
        But UP direction accuracy 60% >= EXCLUDE_BELOW → skip inversion.
        """
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(
                accuracy_pct=30.0,  # overall bad → brain_accuracy < 38%
                up_accuracy=60.0, up_total=25,   # UP is fine
                down_accuracy=10.0, down_total=25,  # DOWN is terrible
            ),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        # Should NOT invert UP because UP accuracy (60%) >= EXCLUDE_BELOW
        assert d.action == "SEND"  # kept, not inverted
        assert d.inverted is False

    def test_both_bad_invert(self):
        """Both UP and DOWN are bad — should invert."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(
                accuracy_pct=25.0,
                up_accuracy=25.0, up_total=25,
                down_accuracy=25.0, down_total=25,
            ),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.action == "INVERT"
        assert d.direction == "DOWN"

    def test_direction_split_in_decision(self):
        """Direction split data should be in the decision."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "ETH": make_sym_ctx(accuracy_pct=55.0, up_accuracy=60.0, down_accuracy=50.0),
        })
        d = brain.analyze_symbol("ETH", "UP", 0.72, context=ctx)
        assert "up" in d.direction_split
        assert "down" in d.direction_split


# ═══════════════════════════════════════════════════════════════
# #2: RECENCY WEIGHTING
# ═══════════════════════════════════════════════════════════════

class TestRecency:
    """Test that recent accuracy is weighted heavier than all-time."""

    def test_recent_good_overrides_bad_alltime(self):
        """All-time is 35% (invert zone) but recent 30d is 65% — should NOT invert.

        brain_accuracy = 65*0.70 + 35*0.20 + 35*0.10 = 45.5+7+3.5 = 56
        """
        brain = GhostBrain()
        ctx = make_context(symbols={
            "SOL": make_sym_ctx(
                accuracy_pct=35.0,          # all-time: terrible
                recent_accuracy=65.0,       # recent: great!
                recent_total=30,
            ),
        })
        d = brain.analyze_symbol("SOL", "UP", 0.72, context=ctx)
        assert d.brain_accuracy > 50.0
        assert d.action != "INVERT"

    def test_recent_bad_overrides_good_alltime(self):
        """All-time is 65% but recent is 30% — brain_accuracy should drop.

        brain_accuracy = 30*0.70 + 65*0.20 + 65*0.10 = 21+13+6.5 = 40.5
        """
        brain = GhostBrain()
        ctx = make_context(symbols={
            "AAPL": make_sym_ctx(
                accuracy_pct=65.0,
                recent_accuracy=30.0,
                recent_total=30,
            ),
        })
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        assert d.brain_accuracy < 50.0

    def test_insufficient_recent_uses_alltime(self):
        """Not enough recent data — should fall back to all-time.

        With recent_total=3 (< MIN=10), brain uses alltime for all components.
        brain_accuracy = 65*0.70 + 65*0.20 + 65*0.10 = 65
        """
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(
                accuracy_pct=65.0,
                recent_accuracy=20.0,  # would be bad...
                recent_total=3,        # ...but too few samples
            ),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.brain_accuracy >= 60.0


# ═══════════════════════════════════════════════════════════════
# #3, #16: CONFIDENCE CALIBRATION
# ═══════════════════════════════════════════════════════════════

class TestCalibration:
    """Test confidence calibration curve."""

    def test_overconfident_deflated(self):
        """Model says 80% but calibration shows only 55% actual → deflate."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"AAPL": make_sym_ctx(accuracy_pct=65.0)},
            calibration={"0.7": 0.55, "0.8": 0.55},
        )
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        # HOT zone: boosted to ~0.828, then calibration deflates to ~0.66
        assert d.confidence < 0.80

    def test_no_calibration_passthrough(self):
        """Without calibration curve, confidence passes through zone logic only."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"AAPL": make_sym_ctx(accuracy_pct=58.0)},
        )
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        # WARM zone: no boost, no calibration → confidence ~0.72
        assert 0.68 <= d.confidence <= 0.76


# ═══════════════════════════════════════════════════════════════
# #4: STREAK AWARENESS
# ═══════════════════════════════════════════════════════════════

class TestStreak:
    """Test win/loss streak modifiers."""

    def test_hot_streak_boosts(self):
        """5-win streak → confidence bonus."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(accuracy_pct=58.0, streak=5),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence > 0.72
        assert "streak" in d.confidence_modifiers

    def test_cold_streak_penalizes(self):
        """5-loss streak → confidence penalty."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(accuracy_pct=58.0, streak=-5),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence < 0.72
        assert d.confidence_modifiers.get("streak", 0) < 0

    def test_no_streak_no_modifier(self):
        """No streak → no modifier added."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(accuracy_pct=58.0, streak=0),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert "streak" not in d.confidence_modifiers


# ═══════════════════════════════════════════════════════════════
# #5: MARKET REGIME
# ═══════════════════════════════════════════════════════════════

class TestRegime:
    """Test market regime confidence modifiers."""

    def test_panic_reduces_confidence(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            regime="panic",
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence < 0.72
        assert d.confidence_modifiers.get("regime", 0) < 0

    def test_calm_boosts_confidence(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            regime="calm",
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence > 0.72
        assert d.confidence_modifiers.get("regime", 0) > 0

    def test_unknown_regime_no_modifier(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            regime="unknown",
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert "regime" not in d.confidence_modifiers


# ═══════════════════════════════════════════════════════════════
# #6: MAGNITUDE WEIGHTING
# ═══════════════════════════════════════════════════════════════

class TestMagnitude:
    """Test that win/loss magnitude affects brain accuracy."""

    def test_big_wins_small_losses_bonus(self):
        """Wins are 3x bigger than losses → accuracy bonus."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(
                accuracy_pct=55.0,
                avg_win_mag=6.0,   # big wins
                avg_loss_mag=2.0,  # small losses
            ),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.brain_accuracy > 55.0

    def test_small_wins_big_losses_penalty(self):
        """Wins are tiny, losses are huge → accuracy penalty."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "ETH": make_sym_ctx(
                accuracy_pct=55.0,
                avg_win_mag=1.0,   # tiny wins
                avg_loss_mag=5.0,  # big losses (ratio < 0.5)
            ),
        })
        d = brain.analyze_symbol("ETH", "UP", 0.72, context=ctx)
        assert d.brain_accuracy < 55.0


# ═══════════════════════════════════════════════════════════════
# #7: DAY-OF-WEEK
# ═══════════════════════════════════════════════════════════════

class TestDOW:
    """Test day-of-week accuracy patterns."""

    def test_bad_day_penalizes(self):
        """Friday accuracy is 30% vs avg 60% → penalty."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={
                "BTC": make_sym_ctx(
                    accuracy_pct=60.0,
                    dow_accuracy={5: 30.0},  # Friday (5=Friday in SQL DOW)
                ),
            },
            current_day=5,  # Friday
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("dow", 0) < 0

    def test_good_day_boosts(self):
        """Tuesday accuracy is 75% vs avg 55% → boost."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={
                "BTC": make_sym_ctx(
                    accuracy_pct=55.0,
                    dow_accuracy={2: 75.0},  # Tuesday
                ),
            },
            current_day=2,  # Tuesday
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("dow", 0) > 0


# ═══════════════════════════════════════════════════════════════
# #10: FEAR & GREED
# ═══════════════════════════════════════════════════════════════

class TestFearGreed:
    """Test Fear & Greed index integration."""

    def test_extreme_fear_penalizes_crypto(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            fear_greed=15,  # extreme fear
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("fear_greed", 0) < 0

    def test_extreme_greed_penalizes_crypto(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"DOGE": make_sym_ctx(accuracy_pct=58.0)},
            fear_greed=85,  # extreme greed
        )
        d = brain.analyze_symbol("DOGE", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("fear_greed", 0) < 0

    def test_neutral_fg_no_modifier(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            fear_greed=50,  # neutral
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert "fear_greed" not in d.confidence_modifiers

    def test_stocks_less_affected(self):
        """Stocks should be less affected by crypto F&G."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"AAPL": make_sym_ctx(accuracy_pct=58.0)},
            fear_greed=15,
        )
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        fg_mod = d.confidence_modifiers.get("fear_greed", 0)
        # Stock penalty is -0.05 (smaller than crypto -0.12)
        assert fg_mod >= -0.06


# ═══════════════════════════════════════════════════════════════
# #11: SECTOR CORRELATION
# ═══════════════════════════════════════════════════════════════

class TestSectorCorrelation:
    """Test sector-wide accuracy influence."""

    def test_cold_meme_sector(self):
        """All meme coins are bad → penalty for DOGE."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "DOGE": make_sym_ctx(accuracy_pct=58.0),
            "SHIB": make_sym_ctx(accuracy_pct=25.0),
            "PEPE": make_sym_ctx(accuracy_pct=28.0),
            "BONK": make_sym_ctx(accuracy_pct=22.0),
        })
        d = brain.analyze_symbol("DOGE", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("sector", 0) < 0

    def test_hot_tech_sector(self):
        """Tech stocks are all strong → boost for AAPL."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "AAPL": make_sym_ctx(accuracy_pct=65.0),
            "MSFT": make_sym_ctx(accuracy_pct=73.0),
            "GOOGL": make_sym_ctx(accuracy_pct=72.0),
            "AMZN": make_sym_ctx(accuracy_pct=71.0),
        })
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("sector", 0) > 0

    def test_no_sector_no_modifier(self):
        """Symbol not in any sector → no modifier."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "RANDOM": make_sym_ctx(accuracy_pct=58.0),
        })
        d = brain.analyze_symbol("RANDOM", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("sector", 0) == 0


# ═══════════════════════════════════════════════════════════════
# #18: CROSS-ASSET LEADING INDICATORS
# ═══════════════════════════════════════════════════════════════

class TestCrossAsset:
    """Test BTC/SPY leading indicator influence."""

    def test_btc_crash_penalizes_crypto_up(self):
        """BTC down 6% + predicting altcoin UP → penalty."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"DOGE": make_sym_ctx(accuracy_pct=58.0)},
            btc_24h=-6.0,
        )
        d = brain.analyze_symbol("DOGE", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("cross_asset", 0) < 0

    def test_btc_pump_no_penalty_for_up(self):
        """BTC up 5% + predicting altcoin UP → no penalty."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"DOGE": make_sym_ctx(accuracy_pct=58.0)},
            btc_24h=5.0,
        )
        d = brain.analyze_symbol("DOGE", "UP", 0.72, context=ctx)
        cross_mod = d.confidence_modifiers.get("cross_asset", 0)
        assert cross_mod >= 0

    def test_spy_crash_penalizes_stock_up(self):
        """SPY down 4% + predicting AAPL UP → penalty."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"AAPL": make_sym_ctx(accuracy_pct=58.0)},
            spy_24h=-4.0,
        )
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("cross_asset", 0) < 0


# ═══════════════════════════════════════════════════════════════
# #20: WEEKEND DETECTOR
# ═══════════════════════════════════════════════════════════════

class TestWeekend:
    """Test weekend crypto penalty."""

    def test_weekend_crypto_penalty(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            is_weekend=True,
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.confidence_modifiers.get("weekend", 0) < 0

    def test_weekday_no_penalty(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            is_weekend=False,
        )
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert "weekend" not in d.confidence_modifiers

    def test_weekend_stock_no_penalty(self):
        """Stocks don't trade weekends — no penalty needed."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"AAPL": make_sym_ctx(accuracy_pct=58.0)},
            is_weekend=True,
        )
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        assert "weekend" not in d.confidence_modifiers


# ═══════════════════════════════════════════════════════════════
# #23: CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Test emergency brake on bad accuracy streaks."""

    def test_circuit_breaker_activates(self):
        """3-day accuracy 35% with 50 predictions → brake."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            rolling_3d_acc=35.0,
            rolling_3d_total=50,
        )
        results = brain.analyze_batch(
            {"BTC": {"direction": "UP", "confidence": 0.72}},
            context=ctx,
        )
        assert brain._circuit_breaker_active is True
        # Confidence should be reduced
        assert results["BTC"].confidence < 0.72

    def test_circuit_breaker_inactive_when_ok(self):
        """3-day accuracy 60% → no brake."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            rolling_3d_acc=60.0,
            rolling_3d_total=50,
        )
        brain.analyze_batch(
            {"BTC": {"direction": "UP", "confidence": 0.72}},
            context=ctx,
        )
        assert brain._circuit_breaker_active is False

    def test_circuit_breaker_needs_min_predictions(self):
        """Not enough 3-day predictions → no brake even if accuracy is bad."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)},
            rolling_3d_acc=20.0,
            rolling_3d_total=5,  # too few (need 30)
        )
        brain.analyze_batch(
            {"BTC": {"direction": "UP", "confidence": 0.72}},
            context=ctx,
        )
        assert brain._circuit_breaker_active is False


# ═══════════════════════════════════════════════════════════════
# #14: AUTO-PRUNE
# ═══════════════════════════════════════════════════════════════

class TestAutoPrune:
    """Test chronic noise symbol detection."""

    def test_chronic_noise_flagged(self):
        """Symbol at 44% for 120 days, 150 predictions → prune.

        brain_accuracy = 44 (EXCLUDE zone). Prune check fires.
        """
        brain = GhostBrain()
        ctx = make_context(symbols={
            "NOISE": make_sym_ctx(
                accuracy_pct=44.0, total=150, days_tracked=120
            ),
        })
        d = brain.analyze_symbol("NOISE", "UP", 0.72, context=ctx)
        assert d.prune_candidate is True
        assert "NOISE" in brain._prune_candidates

    def test_new_symbol_not_pruned(self):
        """Symbol at 44% but only 30 predictions → too few to prune."""
        brain = GhostBrain()
        ctx = make_context(symbols={
            "NEW": make_sym_ctx(
                accuracy_pct=44.0, total=30, days_tracked=20
            ),
        })
        d = brain.analyze_symbol("NEW", "UP", 0.72, context=ctx)
        assert d.prune_candidate is False


# ═══════════════════════════════════════════════════════════════
# #9, #25: SELF-EVOLVING THRESHOLDS
# ═══════════════════════════════════════════════════════════════

class TestThresholdOptimization:
    """Test threshold self-optimization."""

    def test_finds_better_thresholds(self):
        """Optimizer should find thresholds with non-negative lift."""
        brain = GhostBrain()
        data = {}
        for i, acc in enumerate([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]):
            data[f"SYM{i}"] = make_accuracy(float(acc), total=50)

        result = brain.optimize_thresholds(data)
        assert "optimal_invert_below" in result
        assert "optimal_exclude_below" in result
        assert result["optimal_effective"] >= result["current_effective"]

    def test_optimal_lift_non_negative(self):
        """Optimal thresholds should show non-negative lift."""
        brain = GhostBrain()
        data = {
            "BAD1": make_accuracy(15.0),
            "BAD2": make_accuracy(22.0),
            "MID": make_accuracy(45.0),
            "GOOD": make_accuracy(65.0),
        }
        result = brain.optimize_thresholds(data)
        assert result["lift"] >= 0


# ═══════════════════════════════════════════════════════════════
# #21: BACKTEST REPLAY
# ═══════════════════════════════════════════════════════════════

class TestBacktest:
    """Test backtest replay engine."""

    def test_backtest_shows_improvement(self):
        """Brain should improve over raw when inversions help."""
        brain = GhostBrain()
        predictions = {
            "BAD_CRYPTO": {"direction": "UP", "confidence": 0.72},
            "GOOD_STOCK": {"direction": "UP", "confidence": 0.72},
        }
        accuracy = {
            "BAD_CRYPTO": make_accuracy(25.0, total=50),
            "GOOD_STOCK": make_accuracy(68.0, total=50),
        }
        result = brain.backtest_replay(predictions, accuracy)
        assert result["brain_accuracy"] > result["raw_accuracy"]
        assert result["recommendation"] == "SHIP"


# ═══════════════════════════════════════════════════════════════
# #19: EXPECTED VALUE
# ═══════════════════════════════════════════════════════════════

class TestExpectedValue:
    """Test EV computation."""

    def test_positive_ev(self):
        """55% accuracy with big wins = positive EV.
        EV = 5.0*0.55 - 2.0*0.45 = 2.75 - 0.90 = 1.85
        """
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BTC": make_sym_ctx(
                accuracy_pct=55.0,
                avg_win_mag=5.0,
                avg_loss_mag=2.0,
            ),
        })
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.expected_value > 0

    def test_negative_ev(self):
        """52% accuracy with big losses = negative EV.
        EV = 1.0*0.52 - 5.0*0.48 = 0.52 - 2.40 = -1.88
        """
        brain = GhostBrain()
        ctx = make_context(symbols={
            "BAD": make_sym_ctx(
                accuracy_pct=52.0,
                avg_win_mag=1.0,
                avg_loss_mag=5.0,
            ),
        })
        d = brain.analyze_symbol("BAD", "UP", 0.72, context=ctx)
        assert d.expected_value < 0


# ═══════════════════════════════════════════════════════════════
# ASSET CLASS SEPARATION
# ═══════════════════════════════════════════════════════════════

class TestAssetClass:
    """Test stock vs crypto classification and separation."""

    def test_crypto_classified(self):
        brain = GhostBrain()
        ctx = make_context(symbols={"BTC": make_sym_ctx(accuracy_pct=58.0)})
        d = brain.analyze_symbol("BTC", "UP", 0.72, context=ctx)
        assert d.asset_class == "crypto"

    def test_stock_classified(self):
        brain = GhostBrain()
        ctx = make_context(symbols={"AAPL": make_sym_ctx(accuracy_pct=58.0)})
        d = brain.analyze_symbol("AAPL", "UP", 0.72, context=ctx)
        assert d.asset_class == "stock"

    def test_hood_is_stock(self):
        """HOOD (Robinhood) should be classified as stock, not crypto."""
        brain = GhostBrain()
        ctx = make_context(symbols={"HOOD": make_sym_ctx(accuracy_pct=58.0)})
        d = brain.analyze_symbol("HOOD", "UP", 0.72, context=ctx)
        assert d.asset_class == "stock"

    def test_t_is_stock(self):
        """T (AT&T) should be classified as stock, not crypto."""
        brain = GhostBrain()
        ctx = make_context(symbols={"T": make_sym_ctx(accuracy_pct=58.0)})
        d = brain.analyze_symbol("T", "UP", 0.72, context=ctx)
        assert d.asset_class == "stock"

    def test_coin_is_stock(self):
        """COIN (Coinbase) should be classified as stock."""
        brain = GhostBrain()
        ctx = make_context(symbols={"COIN": make_sym_ctx(accuracy_pct=58.0)})
        d = brain.analyze_symbol("COIN", "UP", 0.72, context=ctx)
        assert d.asset_class == "stock"


# ═══════════════════════════════════════════════════════════════
# FULL TIER SPECTRUM
# ═══════════════════════════════════════════════════════════════

class TestTierSpectrum:
    """Test that all zones produce correct tiers."""

    def test_full_spectrum(self):
        brain = GhostBrain()
        cases = [
            ("INV", 20.0, "🔄INVERTED", "INVERT"),
            ("EXC", 42.0, "⛔EXCLUDED", "EXCLUDE"),
            ("COLD", 50.0, "🔴COLD", "SEND"),
            ("WARM", 58.0, "🟡WARM", "SEND"),
            ("HOT", 66.0, "🟢HOT", "SEND"),
            ("FIRE", 75.0, "🔥FIRE", "SEND"),
        ]
        for sym, acc, expected_tier, expected_action in cases:
            accuracy_data = {sym: make_accuracy(acc)}
            d = brain.analyze_symbol(sym, "UP", 0.72, accuracy_data=accuracy_data)
            assert d.tier == expected_tier, f"{sym}: expected tier {expected_tier}, got {d.tier}"
            assert d.action == expected_action, f"{sym}: expected action {expected_action}, got {d.action}"


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

class TestReporting:
    """Test report generation."""

    def test_report_has_key_sections(self):
        brain = GhostBrain()
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.72},
            "AAPL": {"direction": "UP", "confidence": 0.72},
        }
        accuracy_data = {
            "BTC": make_accuracy(30.0),
            "AAPL": make_accuracy(66.0),
        }
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        report = brain.generate_report()
        assert "GHOST BRAIN v3" in report
        assert "INVERTED" in report or "HOT" in report
        assert "STOCKS" in report or "CRYPTO" in report

    def test_telegram_summary(self):
        brain = GhostBrain()
        predictions = {"BTC": {"direction": "UP", "confidence": 0.72}}
        accuracy_data = {"BTC": make_accuracy(30.0)}
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        summary = brain.generate_telegram_summary()
        assert "Brain v3" in summary
        assert "flipped" in summary

    def test_health_endpoint(self):
        brain = GhostBrain()
        predictions = {"BTC": {"direction": "UP", "confidence": 0.72}}
        accuracy_data = {"BTC": make_accuracy(66.0)}
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        health = brain.get_health()
        assert health["version"] == "v3"
        assert health["enabled"] is True
        assert len(health["decisions"]) == 1
        assert "by_asset_class" in health
        assert "feature_importance" in health


# ═══════════════════════════════════════════════════════════════
# BRAIN DISABLED
# ═══════════════════════════════════════════════════════════════

class TestBrainDisabled:
    """Test master switch."""

    def test_disabled_passthrough(self):
        import core.ghost_brain as gb
        original = gb.BRAIN_ENABLED
        gb.BRAIN_ENABLED = False
        try:
            brain = GhostBrain()
            d = brain.analyze_symbol("BTC", "UP", 0.72, accuracy_data={"BTC": make_accuracy(20.0)})
            assert d.action == "SEND"
            assert d.direction == "UP"
            assert d.confidence == 0.72
        finally:
            gb.BRAIN_ENABLED = original


# ═══════════════════════════════════════════════════════════════
# DIRECTION BIAS
# ═══════════════════════════════════════════════════════════════

class TestDirectionBias:
    """Test direction bias detection."""

    def test_all_up_detected(self):
        brain = GhostBrain()
        predictions = {f"S{i}": {"direction": "UP", "confidence": 0.7} for i in range(10)}
        accuracy_data = {f"S{i}": make_accuracy(58.0) for i in range(10)}
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        assert brain._direction_bias["biased"] is True
        assert brain._direction_bias["direction"] == "UP"

    def test_mixed_not_biased(self):
        brain = GhostBrain()
        predictions = {}
        for i in range(5):
            predictions[f"U{i}"] = {"direction": "UP", "confidence": 0.7}
            predictions[f"D{i}"] = {"direction": "DOWN", "confidence": 0.7}
        accuracy_data = {k: make_accuracy(58.0) for k in predictions}
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        assert brain._direction_bias["biased"] is False


# ═══════════════════════════════════════════════════════════════
# CORRELATION GUARD
# ═══════════════════════════════════════════════════════════════

class TestCorrelationGuard:
    """Test overflow penalty when too many same-direction picks."""

    def test_overflow_penalized(self):
        """More than MAX_SAME_DIRECTION crypto UP picks → penalty on weakest."""
        brain = GhostBrain()
        # Create 8 crypto UP picks (MAX_SAME_DIRECTION defaults to 6)
        predictions = {}
        accuracy_data = {}
        crypto_syms = ["BTC", "ETH", "SOL", "DOGE", "ADA", "AVAX", "LINK", "DOT"]
        for sym in crypto_syms:
            predictions[sym] = {"direction": "UP", "confidence": 0.72}
            accuracy_data[sym] = make_accuracy(58.0)

        results = brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        assert len(brain._correlation_warnings) > 0

    def test_within_limit_no_penalty(self):
        """3 crypto UP picks (under limit) → no penalty."""
        brain = GhostBrain()
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.72},
            "ETH": {"direction": "UP", "confidence": 0.72},
            "SOL": {"direction": "UP", "confidence": 0.72},
        }
        accuracy_data = {k: make_accuracy(58.0) for k in predictions}
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        assert len(brain._correlation_warnings) == 0


# ═══════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_zero_accuracy(self):
        brain = GhostBrain()
        accuracy_data = {"BTC": make_accuracy(0.0)}
        d = brain.analyze_symbol("BTC", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.action == "INVERT"
        assert d.effective_accuracy == 100.0

    def test_100_accuracy(self):
        brain = GhostBrain()
        accuracy_data = {"BTC": make_accuracy(100.0)}
        d = brain.analyze_symbol("BTC", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.action == "SEND"
        assert d.tier == "🔥FIRE"

    def test_invalid_direction_skipped(self):
        brain = GhostBrain()
        predictions = {
            "BTC": {"direction": "SIDEWAYS", "confidence": 0.72},
            "ETH": {"direction": "UP", "confidence": 0.72},
        }
        results = brain.analyze_batch(predictions, accuracy_data={
            "BTC": make_accuracy(50.0),
            "ETH": make_accuracy(50.0),
        })
        assert "BTC" not in results
        assert "ETH" in results

    def test_non_dict_skipped(self):
        brain = GhostBrain()
        predictions = {
            "BTC": "not a dict",
            "ETH": {"direction": "UP", "confidence": 0.72},
        }
        results = brain.analyze_batch(predictions, accuracy_data={
            "ETH": make_accuracy(50.0),
        })
        assert "BTC" not in results

    def test_confidence_never_exceeds_cap(self):
        """Even with all boosts, confidence should be capped."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(
                accuracy_pct=85.0, streak=10,
            )},
            regime="calm",
        )
        d = brain.analyze_symbol("BTC", "UP", 0.95, context=ctx)
        assert d.confidence <= CONFIDENCE_CAP

    def test_confidence_never_below_floor(self):
        """Even with all penalties, confidence should have a floor."""
        brain = GhostBrain()
        ctx = make_context(
            symbols={"BTC": make_sym_ctx(
                accuracy_pct=50.0, streak=-10,
            )},
            regime="panic",
            fear_greed=10,
            btc_24h=-8.0,
            is_weekend=True,
        )
        d = brain.analyze_symbol("BTC", "UP", 0.50, context=ctx)
        assert d.confidence >= 0.01

    def test_no_data_returns_neutral(self):
        """No accuracy data at all → neutral passthrough."""
        brain = GhostBrain()
        d = brain.analyze_symbol("UNKNOWN", "UP", 0.72)
        assert d.action == "SEND"
        assert d.tier == "⚪NEUTRAL"


# ═══════════════════════════════════════════════════════════════
# BATCH STATS
# ═══════════════════════════════════════════════════════════════

class TestBatchStats:

    def test_stats_tracked(self):
        brain = GhostBrain()
        predictions = {
            "BAD": {"direction": "UP", "confidence": 0.72},
            "MID": {"direction": "UP", "confidence": 0.72},
            "GOOD": {"direction": "UP", "confidence": 0.72},
        }
        accuracy_data = {
            "BAD": make_accuracy(25.0),
            "MID": make_accuracy(44.0),
            "GOOD": make_accuracy(66.0),
        }
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        s = brain._cycle_stats
        assert s["analyzed"] == 3
        assert s["inverted"] >= 1
        assert s["excluded"] >= 1
        assert s["boosted"] >= 1

    def test_batch_resets(self):
        """Second batch should reset stats."""
        brain = GhostBrain()
        p1 = {"A": {"direction": "UP", "confidence": 0.72}}
        brain.analyze_batch(p1, accuracy_data={"A": make_accuracy(30.0)})
        assert brain._cycle_stats["inverted"] >= 1

        p2 = {"B": {"direction": "UP", "confidence": 0.72}}
        brain.analyze_batch(p2, accuracy_data={"B": make_accuracy(66.0)})
        assert brain._cycle_stats["inverted"] == 0
        assert brain._cycle_stats["boosted"] >= 1


# ═══════════════════════════════════════════════════════════════
# #17: INVERSE DECAY
# ═══════════════════════════════════════════════════════════════

class TestInverseDecay:
    """Test that inversions don't last forever."""

    def test_fresh_inversion_no_decay(self):
        brain = GhostBrain()
        accuracy_data = {"BTC": make_accuracy(25.0)}
        d = brain.analyze_symbol("BTC", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.inverted is True
        assert not any("INVERSE_DECAY" in r for r in d.reasons)

    def test_old_inversion_flagged(self):
        brain = GhostBrain()
        brain._inverse_tracker["BTC"] = datetime.now() - timedelta(days=40)
        accuracy_data = {"BTC": make_accuracy(25.0)}
        d = brain.analyze_symbol("BTC", "UP", 0.72, accuracy_data=accuracy_data)
        assert d.inverted is True
        assert any("INVERSE_DECAY" in r for r in d.reasons)


# ═══════════════════════════════════════════════════════════════
# #22: A/B TESTING
# ═══════════════════════════════════════════════════════════════

class TestABTesting:
    """Test A/B group assignment."""

    def test_groups_assigned(self):
        brain = GhostBrain()
        predictions = {
            "BTC": {"direction": "UP", "confidence": 0.72},
            "ETH": {"direction": "UP", "confidence": 0.72},
            "SOL": {"direction": "UP", "confidence": 0.72},
        }
        accuracy_data = {k: make_accuracy(55.0) for k in predictions}
        brain.analyze_batch(predictions, accuracy_data=accuracy_data)
        assert len(brain._ab_groups) == 3
        assert all(g in ("A", "B") for g in brain._ab_groups.values())

    def test_groups_deterministic_within_process(self):
        """Same symbol always gets same group within a single process."""
        brain1 = GhostBrain()
        brain2 = GhostBrain()
        p = {"BTC": {"direction": "UP", "confidence": 0.72}}
        a = {"BTC": make_accuracy(55.0)}
        brain1.analyze_batch(p, accuracy_data=a)
        brain2.analyze_batch(p, accuracy_data=a)
        assert brain1._ab_groups["BTC"] == brain2._ab_groups["BTC"]


# ═══════════════════════════════════════════════════════════════
# #24: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

class TestFeatureImportance:
    """Test ability contribution tracking."""

    def test_contributions_tracked(self):
        brain = GhostBrain()
        ctx = make_context(
            symbols={
                "BTC": make_sym_ctx(accuracy_pct=58.0, streak=3),
            },
            regime="fear",
            is_weekend=True,
        )
        brain.analyze_batch(
            {"BTC": {"direction": "UP", "confidence": 0.72}},
            context=ctx,
        )
        assert len(brain._feature_contributions) > 0
        health = brain.get_health()
        assert len(health["feature_importance"]) > 0


# ═══════════════════════════════════════════════════════════════
# COMPREHENSIVE IQ TEST (35 symbols, stock vs crypto)
# ═══════════════════════════════════════════════════════════════

class TestIQTest:
    """The ultimate brain intelligence test."""

    def test_stock_vs_crypto_separation(self):
        """
        Stocks at ~63% accuracy should be KEPT.
        Crypto at ~28% accuracy should be FLIPPED.
        Combined effective should be > 55%.
        """
        import random
        random.seed(42)
        brain = GhostBrain()

        symbols_ctx = {}
        predictions = {}

        # 15 stocks at 55-72% accuracy
        stock_syms = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
                       "NVDA", "META", "HOOD", "COIN", "T",
                       "AMD", "INTC", "NFLX", "DIS", "BA"]
        for sym in stock_syms:
            acc = random.uniform(55, 72)
            symbols_ctx[sym] = make_sym_ctx(accuracy_pct=acc)
            predictions[sym] = {"direction": "UP", "confidence": 0.72}

        # 20 crypto at 18-37% accuracy (mostly below INVERT threshold)
        crypto_syms = ["BTC", "ETH", "SOL", "DOGE", "SHIB",
                        "PEPE", "TURBO", "JUP", "IOTX", "CHZ",
                        "ALICE", "YFI", "ICP", "BRETT", "GIGA",
                        "BCH", "BONK", "WIF", "FLOKI", "ADA"]
        for sym in crypto_syms:
            acc = random.uniform(18, 37)
            symbols_ctx[sym] = make_sym_ctx(accuracy_pct=acc)
            predictions[sym] = {"direction": "UP", "confidence": 0.72}

        ctx = make_context(symbols=symbols_ctx)
        results = brain.analyze_batch(predictions, context=ctx)

        # Verify stocks were NOT inverted
        stock_inverted = sum(1 for s in stock_syms if results[s].inverted)
        assert stock_inverted == 0, f"Stocks should not be inverted, got {stock_inverted}"

        # Verify most crypto WAS inverted
        crypto_inverted = sum(1 for s in crypto_syms if results[s].inverted)
        assert crypto_inverted >= 15, f"Most crypto should be inverted, got {crypto_inverted}"

        # Verify effective accuracy is good
        sent = [d for d in results.values() if d.action != "EXCLUDE"]
        avg_effective = sum(d.effective_accuracy for d in sent) / len(sent)
        assert avg_effective > 55.0, f"Combined effective should be >55%, got {avg_effective:.1f}%"

        # Verify report works
        report = brain.generate_report()
        assert "STOCKS" in report
        assert "CRYPTO" in report
        assert "v3" in report
