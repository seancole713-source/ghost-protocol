"""
Integration Test: Prediction Flow
=================================

Tests the COMPLETE chain:
1. Production prediction format → core.models.Prediction (adapter)
2. Predictions → V3Filter → scored results
3. Scored results → formatter → message string

This is NOT a unit test. It proves the components work TOGETHER.
"""

import pytest
from datetime import datetime
from core.adapters import (
    production_to_prediction, 
    batch_convert, 
    from_latest_predictions,
    scored_list_to_formatter,
    format_v3_alert_from_scored,
)
from core.v3_filter import V3Filter
from core.models import Direction
from notifications.formatters import format_top10_message, format_v3_alert


class TestAdapterLayer:
    """Test adapter converts production format correctly."""
    
    def test_convert_turbo_engine_output(self):
        """Test conversion from run_single_prediction output."""
        raw = {
            'ok': True,
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': 0.75,
            'current_price': 2310.0,
            'target_price': 2200.0,
            'stop_loss': 2350.0,
            'run_at': 1706990400,
        }
        
        pred = production_to_prediction(raw)
        
        assert pred is not None
        assert pred.symbol == 'ETH'
        assert pred.direction == Direction.DOWN
        assert pred.confidence == 0.75
        assert pred.current_price == 2310.0
        assert pred.target_price == 2200.0
        assert pred.stop_loss == 2350.0
    
    def test_convert_stock_engine_output(self):
        """Test conversion from stock_engine result."""
        raw = {
            'ok': True,
            'symbol': 'WOLF',
            'direction': 'UP',
            'confidence': 0.68,
            'current_price': 15.50,
            'target_price': 16.25,
            'stop_loss': 15.00,
            'engine': 'stock_v2',
            'horizon_hours': 24,
            'gates_passed': ['volume', 'trend'],
            'gates_failed': ['volatility'],
        }
        
        pred = production_to_prediction(raw)
        
        assert pred is not None
        assert pred.symbol == 'WOLF'
        assert pred.direction == Direction.UP
        assert pred.confidence == 0.68
    
    def test_skip_hold_direction(self):
        """HOLD predictions should return None."""
        raw = {
            'ok': True,
            'symbol': 'AAPL',
            'direction': 'HOLD',
            'confidence': 0.50,
            'current_price': 180.0,
        }
        
        pred = production_to_prediction(raw)
        assert pred is None
    
    def test_skip_failed_prediction(self):
        """Failed predictions (ok=False) should return None."""
        raw = {
            'ok': False,
            'symbol': 'XYZ',
            'direction': 'UP',
            'error': 'Price fetch failed',
        }
        
        pred = production_to_prediction(raw)
        assert pred is None
    
    def test_legacy_buy_sell_directions(self):
        """BUY/SELL should convert to UP/DOWN."""
        buy_raw = {'ok': True, 'symbol': 'TSLA', 'direction': 'BUY', 'confidence': 0.60, 'current_price': 200.0}
        sell_raw = {'ok': True, 'symbol': 'NVDA', 'direction': 'SELL', 'confidence': 0.65, 'current_price': 500.0}
        
        buy_pred = production_to_prediction(buy_raw)
        sell_pred = production_to_prediction(sell_raw)
        
        assert buy_pred.direction == Direction.UP
        assert sell_pred.direction == Direction.DOWN
    
    def test_derive_missing_stop_loss(self):
        """Stop loss should be derived if missing (2% from entry)."""
        raw = {
            'ok': True,
            'symbol': 'XRP',
            'direction': 'UP',
            'confidence': 0.72,
            'current_price': 1.00,
            # No stop_loss provided
        }
        
        pred = production_to_prediction(raw)
        
        assert pred is not None
        assert pred.stop_loss == 0.98  # 2% below for UP
    
    def test_batch_convert_filters_invalid(self):
        """Batch convert should filter out invalid predictions."""
        raw_list = [
            {'ok': True, 'symbol': 'ETH', 'direction': 'DOWN', 'confidence': 0.75, 'current_price': 2300.0},
            {'ok': False, 'symbol': 'BAD', 'direction': 'UP', 'error': 'Failed'},
            {'ok': True, 'symbol': 'HOLD', 'direction': 'HOLD', 'confidence': 0.50, 'current_price': 100.0},
            {'ok': True, 'symbol': 'XRP', 'direction': 'UP', 'confidence': 0.72, 'current_price': 1.65},
        ]
        
        predictions = batch_convert(raw_list)
        
        assert len(predictions) == 2
        symbols = [p.symbol for p in predictions]
        assert 'ETH' in symbols
        assert 'XRP' in symbols
        assert 'BAD' not in symbols
        assert 'HOLD' not in symbols
    
    def test_from_latest_predictions_dict(self):
        """Test conversion from _LATEST_PREDICTIONS cache format."""
        cache = {
            'ETH': {'symbol': 'ETH', 'direction': 'DOWN', 'confidence': 0.75, 'current_price': 2300.0, 'ok': True},
            'BTC': {'symbol': 'BTC', 'direction': 'UP', 'confidence': 0.68, 'current_price': 75000.0, 'ok': True},
        }
        
        predictions = from_latest_predictions(cache)
        
        assert len(predictions) == 2


class TestFullPredictionFlow:
    """Integration test: production predictions → filter → format."""
    
    def test_full_flow_with_realistic_predictions(self):
        """
        Test complete flow with realistic production data.
        
        This simulates what happens when:
        1. Production generates predictions for multiple symbols
        2. Adapter converts to core format
        3. V3 filter applies validation rules
        4. Formatter creates Telegram message
        """
        # Simulate production predictions (realistic mix)
        raw_predictions = [
            # ETH DOWN 82% - should PASS (inverse strategy, will become BUY)
            {
                'ok': True,
                'symbol': 'ETH',
                'direction': 'DOWN',
                'confidence': 0.82,
                'current_price': 2310.0,
                'target_price': 2200.0,
                'stop_loss': 2350.0,
                'run_at': datetime.now().timestamp(),
            },
            # XRP UP 80% - should PASS (mean reversion, 168h hold)
            {
                'ok': True,
                'symbol': 'XRP',
                'direction': 'UP',
                'confidence': 0.80,
                'current_price': 1.65,
                'target_price': 1.75,
                'stop_loss': 1.58,
                'run_at': datetime.now().timestamp(),
            },
            # BTC DOWN 80% - should FAIL (not V3 validated)
            {
                'ok': True,
                'symbol': 'BTC',
                'direction': 'DOWN',
                'confidence': 0.80,
                'current_price': 75000.0,
                'target_price': 72000.0,
                'stop_loss': 76500.0,
                'run_at': datetime.now().timestamp(),
            },
            # SOL UP 85% - should FAIL (removed from V3)
            {
                'ok': True,
                'symbol': 'SOL',
                'direction': 'UP',
                'confidence': 0.85,
                'current_price': 180.0,
                'target_price': 190.0,
                'stop_loss': 175.0,
                'run_at': datetime.now().timestamp(),
            },
            # LINK DOWN 79% - should PASS (mean reversion, 72h hold)
            {
                'ok': True,
                'symbol': 'LINK',
                'direction': 'DOWN',
                'confidence': 0.79,
                'current_price': 22.0,
                'target_price': 21.0,
                'stop_loss': 22.50,
                'run_at': datetime.now().timestamp(),
            },
            # ETH UP 85% - should FAIL (inverse requires DOWN)
            {
                'ok': True,
                'symbol': 'ETH',
                'direction': 'UP',
                'confidence': 0.85,
                'current_price': 2310.0,
                'target_price': 2400.0,
                'stop_loss': 2250.0,
                'run_at': datetime.now().timestamp(),
            },
            # Failed prediction - should be filtered by adapter
            {
                'ok': False,
                'symbol': 'FAIL',
                'direction': 'UP',
                'error': 'Price fetch failed',
            },
            # HOLD prediction - should be filtered by adapter
            {
                'ok': True,
                'symbol': 'NEUTRAL',
                'direction': 'HOLD',
                'confidence': 0.50,
                'current_price': 100.0,
            },
        ]
        
        # Step 1: Convert to core format
        predictions = batch_convert(raw_predictions)
        
        # Should have 6 valid predictions (excluding FAIL and HOLD)
        assert len(predictions) == 6, f"Expected 6 predictions, got {len(predictions)}"
        
        # Step 2: Filter through V3
        v3_filter = V3Filter(min_confidence=0.78)
        scored = v3_filter.filter_and_score(predictions)
        
        # Should pass: ETH DOWN (inverse), XRP UP, LINK DOWN
        # Should fail: BTC (not validated), SOL (removed), ETH UP (inverse requires DOWN)
        assert len(scored) == 3, f"Expected 3 scored, got {len(scored)}: {[s.symbol for s in scored]}"
        
        symbols = [p.symbol for p in scored]
        assert 'ETH' in symbols, "ETH should pass (inverse DOWN→BUY)"
        assert 'XRP' in symbols, "XRP should pass (mean reversion)"
        assert 'LINK' in symbols, "LINK should pass (mean reversion)"
        assert 'BTC' not in symbols, "BTC should be filtered (not V3 validated)"
        assert 'SOL' not in symbols, "SOL should be filtered (removed from V3)"
        
        # Step 3: Convert to formatter format
        stocks, crypto = scored_list_to_formatter(scored)
        
        # All V3 symbols are crypto
        assert len(stocks) == 0, "No V3 stocks currently"
        assert len(crypto) == 3, f"Expected 3 crypto, got {len(crypto)}"
        
        # Step 4: Format message
        messages = format_top10_message(stocks, crypto)
        message = "\n".join(messages) if isinstance(messages, list) else messages
        
        # Verify message contains expected content
        assert '🎯 GHOST TOP 10' in message, "Should have header"
        assert 'ETH' in message, "ETH should be in message"
        assert 'XRP' in message, "XRP should be in message"
        assert 'LINK' in message, "LINK should be in message"
        assert '🔄 INVERSE' in message, "ETH inverse should show badge"
        
        # Verify BTC and SOL NOT in message
        assert 'BTC' not in message, "BTC should NOT be in message"
        assert 'SOL' not in message, "SOL should NOT be in message"
    
    def test_flow_when_all_filtered(self):
        """Test when all predictions fail V3 validation."""
        raw_predictions = [
            # ETH UP - fails inverse (requires DOWN)
            {'ok': True, 'symbol': 'ETH', 'direction': 'UP', 'confidence': 0.80, 'current_price': 2300.0},
            # BTC - not V3 validated
            {'ok': True, 'symbol': 'BTC', 'direction': 'DOWN', 'confidence': 0.85, 'current_price': 75000.0},
            # XRP - below confidence threshold
            {'ok': True, 'symbol': 'XRP', 'direction': 'UP', 'confidence': 0.65, 'current_price': 1.65},
        ]
        
        predictions = batch_convert(raw_predictions)
        assert len(predictions) == 3
        
        v3_filter = V3Filter(min_confidence=0.78)
        scored = v3_filter.filter_and_score(predictions)
        
        assert len(scored) == 0, "All should be filtered"
        
        stocks, crypto = scored_list_to_formatter(scored)
        messages = format_top10_message(stocks, crypto)
        message = "\n".join(messages) if isinstance(messages, list) else messages
        
        # When all filtered, we still get header but no picks section
        # Verify no stocks/crypto symbols appear
        assert '🎯 GHOST TOP 10' in message, "Should have header"
        assert 'ETH' not in message, "ETH should not appear (wrong direction)"
        assert 'BTC' not in message, "BTC should not appear (not V3 validated)"
        assert 'XRP' not in message, "XRP should not appear (below threshold)"
    
    def test_v3_alert_formatting(self):
        """Test V3 alert message formatting for individual signals."""
        raw = {
            'ok': True,
            'symbol': 'ETH',
            'direction': 'DOWN',
            'confidence': 0.82,
            'current_price': 2310.0,
            'target_price': 2200.0,
            'stop_loss': 2350.0,
        }
        
        pred = production_to_prediction(raw)
        assert pred is not None
        
        v3_filter = V3Filter()
        scored_list = v3_filter.filter_and_score([pred])
        assert len(scored_list) == 1
        
        scored = scored_list[0]
        
        # Format as V3 alert using adapter
        alert = format_v3_alert_from_scored(scored)
        
        assert '🎯 V3 SIGNAL' in alert
        assert 'ETH' in alert
        assert '🔄 INVERSE' in alert  # ETH uses inverse strategy
        assert '82%' in alert or '82.0%' in alert  # Confidence shown


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_confidence_at_threshold(self):
        """Test exactly 78% confidence passes."""
        raw = {
            'ok': True,
            'symbol': 'XRP',
            'direction': 'UP',
            'confidence': 0.78,  # Exactly at threshold
            'current_price': 1.65,
        }
        
        predictions = batch_convert([raw])
        v3_filter = V3Filter(min_confidence=0.78)
        scored = v3_filter.filter_and_score(predictions)
        
        assert len(scored) == 1, "78% should pass (>= threshold)"
    
    def test_confidence_just_below_threshold(self):
        """Test 77.9% confidence fails."""
        raw = {
            'ok': True,
            'symbol': 'XRP',
            'direction': 'UP',
            'confidence': 0.779,  # Just below
            'current_price': 1.65,
        }
        
        predictions = batch_convert([raw])
        v3_filter = V3Filter(min_confidence=0.78)
        scored = v3_filter.filter_and_score(predictions)
        
        assert len(scored) == 0, "77.9% should fail (< threshold)"
    
    def test_empty_input(self):
        """Test empty prediction list."""
        predictions = batch_convert([])
        v3_filter = V3Filter()
        scored = v3_filter.filter_and_score(predictions)
        
        assert len(scored) == 0
        
        stocks, crypto = scored_list_to_formatter(scored)
        messages = format_top10_message(stocks, crypto)
        message = "\n".join(messages) if isinstance(messages, list) else messages
        assert message  # Should return some message, not crash
    
    def test_malformed_production_data(self):
        """Test handling of malformed production data."""
        raw_list = [
            {},  # Empty dict
            {'symbol': 'ETH'},  # Missing required fields
            {'ok': True, 'symbol': '', 'direction': 'UP'},  # Empty symbol
            {'ok': True, 'symbol': 'XRP', 'direction': 'UP', 'confidence': 'invalid'},  # Invalid confidence
            None,  # None value (if somehow in list)
        ]
        
        # Filter out None before passing
        raw_list = [r for r in raw_list if r is not None]
        
        # Should not crash, just return empty
        predictions = batch_convert(raw_list)
        assert len(predictions) == 0, "All malformed should be filtered"
