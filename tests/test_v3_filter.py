#!/usr/bin/env python3
"""
🧪 V3 FILTER TESTS - Critical Path Tests

These tests verify the V3 filter logic that determines which signals
get sent to Telegram. Run before any deployment.

Usage:
    pytest tests/test_v3_filter.py -v

Created: 2026-02-03
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestV3FilterLogic:
    """Test the V3 filter function directly."""
    
    def test_eth_inverse_requires_down_direction(self):
        """ETH inverse should ONLY pass when Ghost predicts DOWN."""
        from core.ghost_notifications import v3_filter_and_score
        
        # ETH with UP direction should be SKIPPED
        # (inverse edge is DOWN->UP, not UP->UP)
        pred_up = [{
            'symbol': 'ETH',
            'direction': 'UP',
            'confidence': 0.85,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_up)
        assert len(result) == 0, "ETH UP should be skipped (only DOWN->inverse is validated)"
        
        # ETH with DOWN direction and high confidence should PASS
        pred_down = [{
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': 0.85,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_down)
        assert len(result) == 1, "ETH DOWN with 85% conf should pass"
        assert result[0]['direction'] == 'UP', "ETH should be flipped to UP (inverse)"
        assert result[0]['v3_is_inverse'] == True, "Should be marked as inverse"
    
    def test_eth_inverse_requires_minimum_confidence(self):
        """ETH inverse should require 78% confidence minimum."""
        from core.ghost_notifications import v3_filter_and_score, V3_MIN_CONFIDENCE
        
        # Low confidence should be SKIPPED
        pred_low = [{
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': 0.45,  # Below 70%
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_low)
        assert len(result) == 0, f"ETH at 45% conf should be skipped (need {V3_MIN_CONFIDENCE:.0%})"
        
        # Edge case: exactly at threshold should PASS
        pred_threshold = [{
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': V3_MIN_CONFIDENCE,  # Exactly 70%
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_threshold)
        assert len(result) == 1, f"ETH at exactly {V3_MIN_CONFIDENCE:.0%} should pass"
    
    def test_xrp_link_require_minimum_confidence(self):
        """XRP and LINK (mean_reversion) should require 78% confidence."""
        from core.ghost_notifications import v3_filter_and_score, V3_MIN_CONFIDENCE
        
        # XRP at low confidence should be SKIPPED
        pred_xrp_low = [{
            'symbol': 'XRP',
            'direction': 'UP',
            'confidence': 0.52,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_xrp_low)
        assert len(result) == 0, "XRP at 52% conf should be skipped"
        
        # LINK at low confidence should be SKIPPED
        pred_link_low = [{
            'symbol': 'LINK',
            'direction': 'DOWN',
            'confidence': 0.45,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_link_low)
        assert len(result) == 0, "LINK at 45% conf should be skipped"
        
        # XRP at high confidence should PASS
        pred_xrp_high = [{
            'symbol': 'XRP',
            'direction': 'UP',
            'confidence': 0.80,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_xrp_high)
        assert len(result) == 1, "XRP at 80% conf should pass"
        assert result[0]['v3_is_inverse'] == False, "XRP is not inverse strategy"
    
    def test_removed_symbols_are_filtered(self):
        """Symbols in V3_REMOVED_SYMBOLS should never pass."""
        from core.ghost_notifications import v3_filter_and_score, V3_REMOVED_SYMBOLS
        
        # SOL is explicitly removed
        pred_sol = [{
            'symbol': 'SOL',
            'direction': 'DOWN',
            'confidence': 0.95,  # Even high confidence
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_sol)
        assert len(result) == 0, "SOL should be filtered (in V3_REMOVED_SYMBOLS)"
        
        # BTC is explicitly removed
        pred_btc = [{
            'symbol': 'BTC',
            'direction': 'DOWN',
            'confidence': 0.99,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_btc)
        assert len(result) == 0, "BTC should be filtered (in V3_REMOVED_SYMBOLS)"
    
    def test_blacklisted_symbols_are_filtered(self):
        """Symbols in V3_BLACKLIST should never pass."""
        from core.ghost_notifications import v3_filter_and_score, V3_BLACKLIST
        
        # MANA is blacklisted
        pred_mana = [{
            'symbol': 'MANA',
            'direction': 'UP',
            'confidence': 0.90,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred_mana)
        assert len(result) == 0, "MANA should be filtered (blacklisted)"


class TestV3Config:
    """Test configuration values are consistent."""
    
    def test_validated_symbols_have_required_fields(self):
        """Each validated strategy should have all required fields."""
        from core.ghost_notifications import V3_VALIDATED_STRATEGIES
        
        required_fields = ['strategy', 'hold_hours', 'win_rate', 'sample_size', 'p_value']
        
        for symbol, config in V3_VALIDATED_STRATEGIES.items():
            for field in required_fields:
                assert field in config, f"{symbol} missing required field: {field}"
            
            # Validate data quality
            assert config['win_rate'] > 0.50, f"{symbol} win_rate should be > 50%"
            assert config['p_value'] < 0.05, f"{symbol} p_value should be < 0.05 (significant)"
            assert config['sample_size'] >= 50, f"{symbol} sample_size should be >= 50"
    
    def test_min_confidence_is_reasonable(self):
        """V3_MIN_CONFIDENCE should be between 50% and 90%."""
        from core.ghost_notifications import V3_MIN_CONFIDENCE
        
        assert 0.50 <= V3_MIN_CONFIDENCE <= 0.90, \
            f"V3_MIN_CONFIDENCE ({V3_MIN_CONFIDENCE}) should be between 50-90%"
    
    def test_no_overlap_between_validated_and_removed(self):
        """A symbol should not be in both validated AND removed."""
        from core.ghost_notifications import V3_VALIDATED_STRATEGIES, V3_REMOVED_SYMBOLS
        
        validated = set(V3_VALIDATED_STRATEGIES.keys())
        removed = set(V3_REMOVED_SYMBOLS.keys())
        
        overlap = validated & removed
        assert len(overlap) == 0, f"Symbols in both validated AND removed: {overlap}"


class TestV3InverseLogic:
    """Test the inverse signal logic specifically."""
    
    def test_inverse_flips_direction(self):
        """When ETH passes inverse filter, direction should flip to UP."""
        from core.ghost_notifications import v3_filter_and_score
        
        pred = [{
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': 0.85,
            'current': 2500.0,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred)
        
        assert len(result) == 1
        assert result[0]['direction'] == 'UP', "Inverse should flip DOWN to UP"
        assert result[0]['v3_original_direction'] == 'DOWN', "Should preserve original"
        assert result[0]['v3_strategy'] == 'ghost_inverse'
    
    def test_inverse_recalculates_targets(self):
        """Inverse signal should have buy-appropriate targets (above current price)."""
        from core.ghost_notifications import v3_filter_and_score
        
        current_price = 2500.0
        pred = [{
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': 0.85,
            'current': current_price,
            'asset_type': 'crypto'
        }]
        result = v3_filter_and_score(pred)
        
        assert len(result) == 1
        # Target should be ABOVE current (it's a BUY after inverse)
        target = result[0].get('target_price', 0)
        assert target > current_price, f"Target {target} should be > current {current_price}"
        
        # Stop should be BELOW current (protecting a long position)
        stop = result[0].get('stop', result[0].get('stop_loss', 0))
        assert stop < current_price, f"Stop {stop} should be < current {current_price}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
