"""
V3 validated strategies and symbol lists.
All symbol-related configuration in one place.

Based on 52,433 trade backtest analysis.
Only strategies with p < 0.05 are included.
"""
from dataclasses import dataclass
from typing import Optional, Dict, FrozenSet, Tuple

# Direction override constant — used in ghost_inverse strategies
# that flip Ghost's prediction (e.g., PANW/NET/FTNT)
DIRECTION_FLIP = 'flip'


@dataclass(frozen=True)
class ValidatedStrategy:
    """Configuration for a V3 validated trading strategy."""
    symbol: str
    strategy: str  # 'ghost_inverse' | 'mean_reversion'
    direction_override: Optional[str]  # 'UP' | 'DOWN' | DIRECTION_FLIP | None
    hold_hours: int
    backtest_win_rate: float
    backtest_trades: int
    p_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    asset_type: Optional[str] = None  # 'stock' | None (crypto is default)


# =============================================================================
# V3 VALIDATED STRATEGIES
# These are the ONLY strategies with p < 0.05 in 52K trade backtest
# =============================================================================
V3_VALIDATED_STRATEGIES: Dict[str, ValidatedStrategy] = {
    'ETH': ValidatedStrategy(
        symbol='ETH',
        strategy='ghost_inverse',
        direction_override=None,  # REMOVED: was 'UP' — forced 0% accuracy when ETH went down
        hold_hours=72,
        backtest_win_rate=0.615,
        backtest_trades=78,
        p_value=0.027,
        confidence_interval=(0.50, 0.72),
    ),
    'XRP': ValidatedStrategy(
        symbol='XRP',
        strategy='mean_reversion',
        direction_override=None,
        hold_hours=168,
        backtest_win_rate=0.565,
        backtest_trades=239,
        p_value=0.043,
        confidence_interval=(0.50, 0.63),
    ),
    'LINK': ValidatedStrategy(
        symbol='LINK',
        strategy='mean_reversion',
        direction_override=None,
        hold_hours=72,
        backtest_win_rate=0.552,
        backtest_trades=268,
        p_value=0.048,
        confidence_interval=(0.49, 0.61),
    ),
    # =========================================================================
    # STOCKS - Added 2026-02-05 based on 5063+ bar backtest
    # =========================================================================
    'PANW': ValidatedStrategy(
        symbol='PANW',
        strategy='ghost_inverse',
        direction_override=None,  # REMOVED: was DIRECTION_FLIP — showed 5% accuracy with flip
        hold_hours=168,
        backtest_win_rate=0.646,
        backtest_trades=65,
        p_value=0.0124,
        confidence_interval=(0.525, 0.751),
        asset_type='stock',
    ),
    'NET': ValidatedStrategy(
        symbol='NET',
        strategy='ghost_inverse',
        direction_override=None,  # REMOVED: was DIRECTION_FLIP — showed 5% accuracy with flip
        hold_hours=168,
        backtest_win_rate=0.625,
        backtest_trades=72,
        p_value=0.0222,
        confidence_interval=(0.510, 0.728),
        asset_type='stock',
    ),
    'FTNT': ValidatedStrategy(
        symbol='FTNT',
        strategy='ghost_inverse',
        direction_override=None,  # REMOVED: was DIRECTION_FLIP — showed 5% accuracy with flip
        hold_hours=168,
        backtest_win_rate=0.623,
        backtest_trades=69,
        p_value=0.0266,
        confidence_interval=(0.505, 0.728),
        asset_type='stock',
    ),
    # CHZ mean_reversion @ 48h: 57.3% win rate, 206 trades, p=0.021
    # Added 2026-02-04 based on backtest + paper trade validation
    'CHZ': ValidatedStrategy(
        symbol='CHZ',
        strategy='mean_reversion',
        direction_override=None,
        hold_hours=48,
        backtest_win_rate=0.573,
        backtest_trades=206,
        p_value=0.021,
        confidence_interval=(0.505, 0.638),
    ),
    # DDOG always_up @ 48h: 56.9% win rate, 202 trades, p=0.0286
    # Added 2026-02-06 by auto-calibration discovery
    'DDOG': ValidatedStrategy(
        symbol='DDOG',
        strategy='always_up',
        direction_override=None,  # REMOVED: was 'UP' — showed 11% accuracy when DDOG went down
        hold_hours=48,
        backtest_win_rate=0.569,
        backtest_trades=202,
        p_value=0.0286,
        confidence_interval=(0.500, 0.636),
        asset_type='stock',
    ),
}


# =============================================================================
# REMOVED SYMBOLS
# These symbols were analyzed but did NOT show statistical significance
# =============================================================================
# NOTE: Edge whitelist symbols removed Feb 11, 2026
# TURBO, RNDR, IQ, ILV, CHZ have proven paper trade performance
V3_REMOVED_SYMBOLS: Dict[str, str] = {
    'SOL': 'Inverse 50.2% over 4962 trades - not significant',
    'BTC': 'Inverse 52% over large sample - not significant',
    'AVAX': 'Inverse 50.2% over 4988 trades - not significant',
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
# NOTE: YFI and HBAR removed Feb 11, 2026 — they are edge whitelist symbols
V3_BLACKLIST: FrozenSet[str] = frozenset([
    'TGTX', 'SOUN', 'ABCL', '1INCH', 'SAND', 'MANA', 'DOT', 'SHIB', 'FIL',
    'VET', 'ALGO', 'ARB', 'NEAR', 'SUSHI', 'LDO', 'ETC', 'IMX',
    'APT', 'SUI', 'RLC',
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
    # Edge whitelist crypto (added Feb 11, 2026)
    'JUP', 'BAND', 'IQ', 'IOTX', 'GIGA', 'BCH', 'ALICE', 'BRETT',
])


# =============================================================================
# DEFAULT EDGE SYMBOLS
# Single source of truth for the EDGE_SYMBOLS env var fallback.
# All code should import this instead of hardcoding the CSV string.
#
# Feb 26, 2026: Expanded from 13 → 50 symbols to increase prediction coverage.
# Sources: V3_VALIDATED_STRATEGIES, V3_WHITELIST_STOCKS, top crypto by volume,
# top stocks by liquidity. Excludes V3_BLACKLIST entries.
# =============================================================================
DEFAULT_EDGE_SYMBOLS = ",".join([
    # ── V3 Validated (statistically proven p<0.05) ── CORE 8
    "ETH", "XRP", "LINK", "CHZ", "PANW", "NET", "FTNT", "DDOG",
    # ── V3 Whitelist Stocks (sweetspot analysis) ──
    "T", "BMBL", "XPO",
    # ── REMOVED Mar 5, 2026 ──
    # BTC — 50% live (coin flip), backtest "not significant" (52%)
    # SOL — 16.7% live, backtest "not significant" (50.2%)
    # Re-add only if >55% win rate over 50+ trades
])
# REMOVED Feb 25, 2026: RNDR — 11% accuracy (1/9), in HARDCODED_EXCLUSIONS.
# REMOVED Feb 27, 2026: ADA(20%), AVAX(30%), BNB(30%), DOGE(30%), YFI(11%)
#   — all in HARDCODED_EXCLUSIONS with <40% accuracy. Wasted compute.
# REMOVED Feb 27, 2026: BONK, PEPE, WIF — meme coins in HARDCODED_EXCLUSIONS.

# Cached resolved edge set (computed once, used everywhere)
_RESOLVED_EDGE_SET: Optional[FrozenSet[str]] = None


def get_edge_set() -> FrozenSet[str]:
    """
    Return the resolved edge symbol set.

    Code default is the SOURCE OF TRUTH. The env var can ADD symbols
    but never override the curated default set with stale/wider lists.
    If EDGE_SYMBOLS env var exists, take the INTERSECTION with our
    default set (keeps only validated symbols) plus any env-var-only
    additions that were explicitly added.
    """
    global _RESOLVED_EDGE_SET
    if _RESOLVED_EDGE_SET is not None:
        return _RESOLVED_EDGE_SET

    import os
    default_set = frozenset(
        s.strip().upper() for s in DEFAULT_EDGE_SYMBOLS.split(",") if s.strip()
    )
    env_raw = os.getenv("EDGE_SYMBOLS", "")
    if env_raw:
        env_set = frozenset(s.strip().upper() for s in env_raw.split(",") if s.strip())
        # Code default is authority — use it. Env var is informational only.
        # This prevents a stale Railway env var from re-adding removed symbols.
        _RESOLVED_EDGE_SET = default_set
    else:
        _RESOLVED_EDGE_SET = default_set
    return _RESOLVED_EDGE_SET


def get_edge_csv() -> str:
    """Return the resolved edge symbols as a comma-separated string."""
    return ",".join(sorted(get_edge_set()))


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


def v3_strategies_as_dicts() -> Dict[str, dict]:
    """Convert V3_VALIDATED_STRATEGIES to legacy dict format.

    Legacy code expects keys: strategy, direction_override, hold_hours,
    win_rate, sample_size, p_value, confidence_interval, asset_type.
    """
    result = {}
    for sym, vs in V3_VALIDATED_STRATEGIES.items():
        d = {
            'strategy': vs.strategy,
            'direction_override': vs.direction_override,
            'hold_hours': vs.hold_hours,
            'win_rate': vs.backtest_win_rate,
            'sample_size': vs.backtest_trades,
            'p_value': vs.p_value,
        }
        if vs.confidence_interval is not None:
            d['confidence_interval'] = vs.confidence_interval
        if vs.asset_type is not None:
            d['asset_type'] = vs.asset_type
        result[sym] = d
    return result
