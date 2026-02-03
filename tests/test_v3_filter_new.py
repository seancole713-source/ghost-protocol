"""
Comprehensive V3 filter tests.

Tests the core V3 filtering logic that determines which predictions
should be acted upon based on backtest validation.
"""
import pytest
from datetime import datetime
from core.v3_filter import V3Filter
from core.models import Prediction, Direction
from config.symbols import V3_VALIDATED_STRATEGIES, V3_REMOVED_SYMBOLS, V3_BLACKLIST


class TestV3FilterBasics:
    """Basic V3 filter functionality."""
    
    def test_empty_predictions_returns_empty(self, v3_filter):
        """Empty input should return empty output."""
        result = v3_filter.filter_and_score([])
        assert result == []
    
    def test_max_results_limits_output(self, v3_filter, all_valid_predictions):
        """max_results should limit output count."""
        result = v3_filter.filter_and_score(all_valid_predictions, max_results=1)
        assert len(result) <= 1
    
    def test_stats_are_tracked(self, v3_filter, sample_predictions):
        """Filter should track processing statistics."""
        v3_filter.filter_and_score(sample_predictions)
        stats = v3_filter.stats
        
        assert stats['total_processed'] == 3
        assert stats['passed'] == 2  # ETH and XRP
        assert stats['rejected_not_validated'] == 1  # BTC
    
    def test_reset_stats(self, v3_filter, sample_predictions):
        """reset_stats should clear all statistics."""
        v3_filter.filter_and_score(sample_predictions)
        v3_filter.reset_stats()
        
        assert v3_filter.stats['total_processed'] == 0
        assert v3_filter.stats['passed'] == 0


class TestETHInverse:
    """ETH ghost_inverse strategy tests."""
    
    def test_eth_down_high_conf_passes(self, v3_filter, eth_down_high_conf):
        """ETH DOWN at 75% should pass and flip to UP."""
        result = v3_filter.filter_and_score([eth_down_high_conf])
        
        assert len(result) == 1
        assert result[0].symbol == 'ETH'
        assert result[0].direction == Direction.UP  # Flipped!
        assert result[0].is_inverse == True
        assert result[0].original_direction == Direction.DOWN
        assert result[0].hold_hours == 72
        assert result[0].backtest_win_rate == 0.615
    
    def test_eth_down_low_conf_rejected(self, v3_filter, eth_down_low_conf):
        """ETH DOWN at 45% should be rejected (below 70% threshold)."""
        result = v3_filter.filter_and_score([eth_down_low_conf])
        assert len(result) == 0
    
    def test_eth_up_rejected(self, v3_filter, eth_up_high_conf):
        """ETH UP should be rejected (inverse only triggers on DOWN)."""
        result = v3_filter.filter_and_score([eth_up_high_conf])
        assert len(result) == 0
    
    def test_eth_inverse_target_above_entry(self, v3_filter, eth_down_high_conf):
        """Inversed ETH should have target above entry (BUY signal)."""
        result = v3_filter.filter_and_score([eth_down_high_conf])
        
        assert result[0].target_price > result[0].current_price
        assert result[0].stop_loss < result[0].current_price
    
    def test_eth_inverse_stats_tracked(self, v3_filter, eth_down_high_conf):
        """Inverse count should be tracked in stats."""
        v3_filter.filter_and_score([eth_down_high_conf])
        assert v3_filter.stats['inversed'] == 1


class TestXRPMeanReversion:
    """XRP mean_reversion strategy tests."""
    
    def test_xrp_up_high_conf_passes(self, v3_filter, xrp_up_high_conf):
        """XRP UP at 75% should pass."""
        result = v3_filter.filter_and_score([xrp_up_high_conf])
        
        assert len(result) == 1
        assert result[0].symbol == 'XRP'
        assert result[0].is_inverse == False
        assert result[0].hold_hours == 168
        assert result[0].backtest_win_rate == 0.565
    
    def test_xrp_down_high_conf_passes(self, v3_filter, xrp_down_high_conf):
        """XRP DOWN at 75% should also pass (mean_reversion works both ways)."""
        result = v3_filter.filter_and_score([xrp_down_high_conf])
        
        assert len(result) == 1
        assert result[0].symbol == 'XRP'
        assert result[0].direction == Direction.DOWN
    
    def test_xrp_low_conf_rejected(self, v3_filter, xrp_low_conf):
        """XRP at 50% should be rejected."""
        result = v3_filter.filter_and_score([xrp_low_conf])
        assert len(result) == 0


class TestLINKMeanReversion:
    """LINK mean_reversion strategy tests."""
    
    def test_link_high_conf_passes(self, v3_filter, link_up_high_conf):
        """LINK at 72% should pass."""
        result = v3_filter.filter_and_score([link_up_high_conf])
        
        assert len(result) == 1
        assert result[0].symbol == 'LINK'
        assert result[0].hold_hours == 72
        assert result[0].backtest_win_rate == 0.552


class TestUnvalidatedSymbols:
    """Test that non-V3 symbols are rejected."""
    
    def test_btc_rejected(self, v3_filter, btc_down_high_conf):
        """BTC should be rejected (not in V3 validated)."""
        result = v3_filter.filter_and_score([btc_down_high_conf])
        assert len(result) == 0
    
    def test_sol_rejected_with_reason(self, v3_filter, sol_down_high_conf):
        """SOL should be rejected (removed from V3)."""
        result = v3_filter.filter_single(sol_down_high_conf)
        
        assert result.passed == False
        assert 'REMOVED' in result.reason
    
    def test_blacklisted_rejected(self, v3_filter, blacklisted_prediction):
        """Blacklisted symbols should be rejected."""
        result = v3_filter.filter_single(blacklisted_prediction)
        
        assert result.passed == False
        assert 'BLACKLISTED' in result.reason
    
    def test_random_symbol_rejected(self, v3_filter):
        """Random unknown symbol should be rejected."""
        pred = Prediction(
            symbol='RANDOM',
            direction=Direction.UP,
            confidence=0.99,
            current_price=100.0,
            target_price=110.0,
            stop_loss=95.0,
            timestamp=datetime.now(),
        )
        result = v3_filter.filter_and_score([pred])
        assert len(result) == 0


class TestConfidenceThreshold:
    """Test confidence threshold edge cases."""
    
    def test_exactly_70_percent_passes(self, v3_filter, eth_down_exactly_70):
        """Exactly 70% confidence should pass."""
        result = v3_filter.filter_and_score([eth_down_exactly_70])
        assert len(result) == 1
    
    def test_just_below_70_percent_rejected(self, v3_filter, eth_down_just_below_70):
        """69.9% confidence should be rejected."""
        result = v3_filter.filter_and_score([eth_down_just_below_70])
        assert len(result) == 0
    
    def test_custom_threshold_respected(self, v3_filter_low_threshold, eth_down_low_conf):
        """Custom confidence threshold should be respected."""
        # 50% threshold should let 45% ETH through... but wait, inverse still needs DOWN
        # and 45% < 50% so still rejected
        result = v3_filter_low_threshold.filter_and_score([eth_down_low_conf])
        assert len(result) == 0  # Still rejected (45% < 50%)
        
        # Test with exactly 50%
        pred = Prediction(
            symbol='ETH',
            direction=Direction.DOWN,
            confidence=0.50,
            current_price=2300.0,
            target_price=2200.0,
            stop_loss=2350.0,
            timestamp=datetime.now(),
        )
        result = v3_filter_low_threshold.filter_and_score([pred])
        assert len(result) == 1  # Should pass at 50%


class TestScoring:
    """Test scoring and sorting."""
    
    def test_higher_scores_first(self, v3_filter, all_valid_predictions):
        """Higher scores should rank first."""
        result = v3_filter.filter_and_score(all_valid_predictions)
        
        # Verify sorted by score descending
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score
    
    def test_score_calculation_eth(self, v3_filter, eth_down_high_conf):
        """ETH score should be win_rate * confidence."""
        result = v3_filter.filter_and_score([eth_down_high_conf])
        
        expected_score = 0.615 * 0.75  # ETH win rate * confidence
        assert abs(result[0].score - expected_score) < 0.001
    
    def test_score_calculation_xrp(self, v3_filter, xrp_up_high_conf):
        """XRP score should be win_rate * confidence."""
        result = v3_filter.filter_and_score([xrp_up_high_conf])
        
        expected_score = 0.565 * 0.75  # XRP win rate * confidence
        assert abs(result[0].score - expected_score) < 0.001


class TestFilterSingle:
    """Test single prediction filtering with detailed results."""
    
    def test_filter_single_pass(self, v3_filter, eth_down_high_conf):
        """filter_single should return detailed pass result."""
        result = v3_filter.filter_single(eth_down_high_conf)
        
        assert result.passed == True
        assert result.symbol == 'ETH'
        assert 'PASSED' in result.reason
        assert result.prediction is not None
    
    def test_filter_single_fail(self, v3_filter, btc_down_high_conf):
        """filter_single should return detailed fail result."""
        result = v3_filter.filter_single(btc_down_high_conf)
        
        assert result.passed == False
        assert result.symbol == 'BTC'
        assert result.prediction is None


class TestConfig:
    """Test V3 configuration."""
    
    def test_validated_symbols_have_required_fields(self):
        """All validated strategies should have required fields."""
        for symbol, strategy in V3_VALIDATED_STRATEGIES.items():
            assert strategy.symbol == symbol
            assert strategy.strategy in ('ghost_inverse', 'mean_reversion')
            assert strategy.hold_hours > 0
            assert 0 < strategy.backtest_win_rate < 1
            assert strategy.backtest_trades > 0
            assert 0 < strategy.p_value < 0.05
    
    def test_no_overlap_between_validated_and_removed(self):
        """Validated and removed symbols should not overlap."""
        validated = set(V3_VALIDATED_STRATEGIES.keys())
        removed = set(V3_REMOVED_SYMBOLS.keys())
        
        overlap = validated & removed
        assert len(overlap) == 0, f"Overlap found: {overlap}"
    
    def test_no_overlap_between_validated_and_blacklist(self):
        """Validated and blacklisted symbols should not overlap."""
        validated = set(V3_VALIDATED_STRATEGIES.keys())
        
        overlap = validated & V3_BLACKLIST
        assert len(overlap) == 0, f"Overlap found: {overlap}"
