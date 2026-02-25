"""
Tests for config module.
"""
import pytest
from config.settings import Settings, settings
from config.symbols import (
    V3_VALIDATED_STRATEGIES,
    V3_REMOVED_SYMBOLS,
    V3_BLACKLIST,
    CRYPTO_SYMBOLS,
    is_crypto,
    is_v3_validated,
    is_blacklisted,
    is_removed,
    get_strategy,
)


class TestSettings:
    """Settings tests."""
    
    def test_default_values(self):
        """Default values should be set."""
        assert settings.V3_MIN_CONFIDENCE == 0.78
        assert settings.V3_DEFAULT_HOLD_HOURS == 72
        assert settings.DEFAULT_TARGET_PCT == 0.066
        assert settings.DEFAULT_STOP_PCT == 0.033
    
    def test_settings_instance_is_singleton(self):
        """settings should be the same instance."""
        from config.settings import settings as s2
        assert settings is s2


class TestSymbolHelpers:
    """Symbol helper function tests."""
    
    def test_is_crypto_true(self):
        """Known crypto symbols should return True."""
        assert is_crypto('BTC') == True
        assert is_crypto('ETH') == True
        assert is_crypto('XRP') == True
        assert is_crypto('btc') == True  # Case insensitive
    
    def test_is_crypto_false(self):
        """Stock symbols should return False."""
        assert is_crypto('AAPL') == False
        assert is_crypto('TSLA') == False
        assert is_crypto('T') == False
    
    def test_is_v3_validated(self):
        """Only validated symbols should return True."""
        assert is_v3_validated('ETH') == True
        assert is_v3_validated('XRP') == True
        assert is_v3_validated('LINK') == True
        assert is_v3_validated('BTC') == False  # Removed, not validated
        assert is_v3_validated('AAPL') == False
    
    def test_is_blacklisted(self):
        """Blacklisted symbols should return True."""
        assert is_blacklisted('SHIB') == True
        assert is_blacklisted('SAND') == True
        assert is_blacklisted('ETH') == False
    
    def test_is_removed(self):
        """Removed symbols should return True."""
        assert is_removed('SOL') == True
        assert is_removed('BTC') == True
        assert is_removed('ETH') == False
    
    def test_get_strategy_exists(self):
        """get_strategy should return strategy for validated symbols."""
        strategy = get_strategy('ETH')
        assert strategy is not None
        assert strategy.symbol == 'ETH'
        assert strategy.strategy == 'ghost_inverse'
    
    def test_get_strategy_not_found(self):
        """get_strategy should return None for unknown symbols."""
        strategy = get_strategy('UNKNOWN')
        assert strategy is None


class TestV3Strategies:
    """V3 strategy configuration tests."""
    
    def test_eth_strategy(self):
        """ETH strategy should be correctly configured."""
        eth = V3_VALIDATED_STRATEGIES['ETH']
        assert eth.strategy == 'ghost_inverse'
        assert eth.direction_override == 'UP'
        assert eth.hold_hours == 72
        assert eth.backtest_win_rate == 0.615
        assert eth.p_value < 0.05
    
    def test_xrp_strategy(self):
        """XRP strategy should be correctly configured."""
        xrp = V3_VALIDATED_STRATEGIES['XRP']
        assert xrp.strategy == 'mean_reversion'
        assert xrp.direction_override is None
        assert xrp.hold_hours == 168
        assert xrp.backtest_win_rate == 0.565
        assert xrp.p_value < 0.05
    
    def test_link_strategy(self):
        """LINK strategy should be correctly configured."""
        link = V3_VALIDATED_STRATEGIES['LINK']
        assert link.strategy == 'mean_reversion'
        assert link.hold_hours == 72
        assert link.backtest_win_rate == 0.552
        assert link.p_value < 0.05
    
    def test_all_strategies_have_p_below_05(self):
        """All validated strategies should have p < 0.05."""
        for symbol, strategy in V3_VALIDATED_STRATEGIES.items():
            assert strategy.p_value < 0.05, f"{symbol} has p={strategy.p_value}"
    
    def test_all_strategies_positive_win_rate(self):
        """All strategies should have positive win rate above 50%."""
        for symbol, strategy in V3_VALIDATED_STRATEGIES.items():
            assert strategy.backtest_win_rate > 0.50, f"{symbol} win rate too low"


class TestV3Exclusions:
    """V3 exclusion list tests."""
    
    def test_removed_symbols_have_reasons(self):
        """All removed symbols should have a reason string."""
        for symbol, reason in V3_REMOVED_SYMBOLS.items():
            assert isinstance(reason, str)
            assert len(reason) > 0
    
    def test_major_crypto_in_removed(self):
        """Major crypto symbols should be in removed (they were analyzed)."""
        assert 'BTC' in V3_REMOVED_SYMBOLS
        assert 'SOL' in V3_REMOVED_SYMBOLS
        assert 'AVAX' in V3_REMOVED_SYMBOLS
    
    def test_blacklist_is_frozenset(self):
        """Blacklist should be immutable."""
        assert isinstance(V3_BLACKLIST, frozenset)
