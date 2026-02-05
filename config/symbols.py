"""
V3 validated strategies and symbol lists.
All symbol-related configuration in one place.

Based on 52,433 trade backtest analysis.
Only strategies with p < 0.05 are included.
"""
from dataclasses import dataclass
from typing import Optional, Dict, FrozenSet


@dataclass(frozen=True)
class ValidatedStrategy:
    """Configuration for a V3 validated trading strategy."""
    symbol: str
    strategy: str  # 'ghost_inverse' | 'mean_reversion'
    direction_override: Optional[str]  # 'UP' | 'DOWN' | None
    hold_hours: int
    backtest_win_rate: float
    backtest_trades: int
    p_value: float


# =============================================================================
# V3 VALIDATED STRATEGIES
# These are the ONLY strategies with p < 0.05 in 52K trade backtest
# =============================================================================
V3_VALIDATED_STRATEGIES: Dict[str, ValidatedStrategy] = {
    'ETH': ValidatedStrategy(
        symbol='ETH',
        strategy='ghost_inverse',
        direction_override='UP',
        hold_hours=72,
        backtest_win_rate=0.615,
        backtest_trades=78,
        p_value=0.027,
    ),
    'XRP': ValidatedStrategy(
        symbol='XRP',
        strategy='mean_reversion',
        direction_override=None,
        hold_hours=168,
        backtest_win_rate=0.565,
        backtest_trades=239,
        p_value=0.043,
    ),
    'LINK': ValidatedStrategy(
        symbol='LINK',
        strategy='mean_reversion',
        direction_override=None,
        hold_hours=72,
        backtest_win_rate=0.552,
        backtest_trades=268,
        p_value=0.048,
    ),
    # =========================================================================
    # STOCKS - Added 2026-02-05 based on 5063+ bar backtest
    # =========================================================================
    'PANW': ValidatedStrategy(
        symbol='PANW',
        strategy='ghost_inverse',
        direction_override='flip',
        hold_hours=168,
        backtest_win_rate=0.646,
        backtest_trades=65,
        p_value=0.0124,
    ),
    'NET': ValidatedStrategy(
        symbol='NET',
        strategy='ghost_inverse',
        direction_override='flip',
        hold_hours=168,
        backtest_win_rate=0.625,
        backtest_trades=72,
        p_value=0.0222,
    ),
    'FTNT': ValidatedStrategy(
        symbol='FTNT',
        strategy='ghost_inverse',
        direction_override='flip',
        hold_hours=168,
        backtest_win_rate=0.623,
        backtest_trades=69,
        p_value=0.0266,
    ),
}


# =============================================================================
# REMOVED SYMBOLS
# These symbols were analyzed but did NOT show statistical significance
# =============================================================================
V3_REMOVED_SYMBOLS: Dict[str, str] = {
    'SOL': 'Inverse 50.2% over 4962 trades - not significant',
    'BTC': 'Inverse 52% over large sample - not significant',
    'AVAX': 'Inverse 50.2% over 4988 trades - not significant',
    'TURBO': 'Inverse 46.5% over 6064 trades - consistently loses',
    'RNDR': 'Inverse 49.3% over 2600 trades - not significant',
    'IQ': 'Inverse 47.8% over 2102 trades - consistently loses',
    'ILV': 'RSI 45.9% over 1853 trades - consistently loses',
    'CHZ': 'Inverse 50.8% over 3201 trades - not significant',
    'ZEC': 'Inverse 50.1% over 3006 trades - not significant',
    'AAVE': 'Inverse 49.8% over 2844 trades - not significant',
    'BNB': 'Inverse 49.5% over 4200 trades - not significant',
    'ADA': 'Inverse 50.3% over 3800 trades - not significant',
    'LTC': 'Inverse 49.7% over 3500 trades - not significant',
}


# =============================================================================
# BLACKLISTED SYMBOLS
# Never trade these regardless of predictions
# =============================================================================
V3_BLACKLIST: FrozenSet[str] = frozenset([
    'TGTX', 'SOUN', 'ABCL', '1INCH', 'SAND', 'MANA', 'DOT', 'SHIB', 'FIL',
    'VET', 'ALGO', 'ARB', 'NEAR', 'SUSHI', 'YFI', 'LDO', 'ETC', 'IMX',
    'APT', 'SUI', 'HBAR', 'RLC',
])


# =============================================================================
# WHITELIST STOCKS
# Stocks from sweetspot analysis with potential edge
# =============================================================================
V3_WHITELIST_STOCKS: FrozenSet[str] = frozenset(['T', 'BMBL', 'XPO'])


# =============================================================================
# CRYPTO SYMBOLS
# All known crypto symbols for asset type detection
# =============================================================================
CRYPTO_SYMBOLS: FrozenSet[str] = frozenset([
    'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'AVAX', 'LINK', 'LTC', 'DOT',
    'MATIC', 'DOGE', 'SHIB', 'ATOM', 'UNI', 'AAVE', 'MKR', 'CRV', 'RNDR',
    'TURBO', 'CHZ', 'ILV', 'ZEC', 'INJ', 'SUI', 'APT', 'ARB', 'OP', 'TIA',
    'PEPE', 'WIF', 'BONK', 'FLOKI', 'MEME', 'ORDI', 'SATS', 'SEI', 'FET',
    'RENDER', 'GRT', 'SNX', 'COMP', '1INCH', 'SAND', 'MANA', 'AXS', 'ENJ',
    'VET', 'ALGO', 'NEAR', 'SUSHI', 'YFI', 'LDO', 'ETC', 'IMX', 'HBAR',
    'RLC', 'FIL', 'ICP', 'EGLD', 'XLM', 'XMR', 'IOTA', 'NEO', 'WAVES',
])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_crypto(symbol: str) -> bool:
    """Check if a symbol is a cryptocurrency."""
    return symbol.upper() in CRYPTO_SYMBOLS


def is_v3_validated(symbol: str) -> bool:
    """Check if a symbol has a V3 validated strategy."""
    return symbol.upper() in V3_VALIDATED_STRATEGIES


def is_blacklisted(symbol: str) -> bool:
    """Check if a symbol is blacklisted."""
    return symbol.upper() in V3_BLACKLIST


def is_removed(symbol: str) -> bool:
    """Check if a symbol was analyzed but removed from V3."""
    return symbol.upper() in V3_REMOVED_SYMBOLS


def get_strategy(symbol: str) -> Optional[ValidatedStrategy]:
    """Get the validated strategy for a symbol, if any."""
    return V3_VALIDATED_STRATEGIES.get(symbol.upper())
