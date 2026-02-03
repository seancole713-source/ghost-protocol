#!/usr/bin/env python3
"""
🎯 GHOST PROTOCOL - CENTRALIZED CONFIGURATION

All thresholds, magic numbers, and tunable parameters in ONE place.
DO NOT scatter these values throughout the codebase.

Updated: 2026-02-03
"""

# ============================================================================
# V3 BACKTEST-VALIDATED PARAMETERS
# Based on 52,433 trades analyzed across 272 strategy combinations
# Only statistically significant results (p < 0.05) included
# ============================================================================

# Minimum confidence for V3 signals (applies to ALL strategies including inverse)
V3_MIN_CONFIDENCE = 0.70  # 70%

# Whether V3 filtering is enabled
V3_ENABLED = True

# ============================================================================
# VALIDATED CRYPTO STRATEGIES
# These are the ONLY crypto symbols with proven edge (p < 0.05)
# ============================================================================

V3_VALIDATED_STRATEGIES = {
    # ETH ghost_inverse @ 72h: 61.5% win rate, 78 trades, p=0.027
    # IMPORTANT: Only valid when Ghost predicts DOWN (then flip to UP)
    'ETH': {
        'strategy': 'ghost_inverse',
        'direction_override': 'UP',  # Always flip DOWN to UP
        'hold_hours': 72,
        'win_rate': 0.615,
        'sample_size': 78,
        'p_value': 0.027,
        'confidence_interval': (0.50, 0.72),
    },
    # XRP mean_reversion @ 168h: 56.5% win rate, 239 trades, p=0.026
    'XRP': {
        'strategy': 'mean_reversion',
        'direction_override': None,  # Use Ghost's direction
        'hold_hours': 168,  # 7 days
        'win_rate': 0.565,
        'sample_size': 239,
        'p_value': 0.026,
        'confidence_interval': (0.50, 0.63),
    },
    # LINK mean_reversion @ 72h: 55.2% win rate, 268 trades, p=0.049
    'LINK': {
        'strategy': 'mean_reversion',
        'direction_override': None,
        'hold_hours': 72,
        'win_rate': 0.552,
        'sample_size': 268,
        'p_value': 0.049,
        'confidence_interval': (0.49, 0.61),
    },
}

# Derived lists for convenience
V3_WHITELIST_CRYPTO = list(V3_VALIDATED_STRATEGIES.keys())  # ['ETH', 'XRP', 'LINK']

# ============================================================================
# SYMBOLS EXPLICITLY REMOVED FROM V3
# These showed "edge" on small samples but ~50% on large samples
# ============================================================================

V3_REMOVED_SYMBOLS = {
    'SOL': 'Inverse 50.2% over 4962 trades - not significant',
    'BTC': 'Inverse 52% over large sample - not significant', 
    'AVAX': 'Inverse 48% - actually loses, bearish bias only',
    'TURBO': 'No backtest data - removed until validated',
    'RNDR': 'No backtest data - removed until validated',
    'IQ': 'No backtest data - removed until validated',
    'ILV': 'No backtest data - removed until validated',
    'CHZ': 'No backtest data - removed until validated',
    'ZEC': 'No backtest data - removed until validated',
    'AAVE': 'No backtest data - removed until validated',
    'BNB': 'No backtest data - removed until validated',
    'ADA': 'No backtest data - removed until validated',
    'LTC': 'No backtest data - removed until validated',
}

# ============================================================================
# BLACKLIST - Symbols and strategies that consistently LOSE
# ============================================================================

V3_BLACKLIST = [
    'TGTX', 'SOUN', 'ABCL',  # Stocks with poor performance
    'ZIL', 'MANA', 'SAND', 'RLC', '1INCH',  # Crypto with poor performance
    'IMX', 'APT', 'SUSHI', 'YFI', 'LDO', 'ETC'
]

# Strategies that underperform (45-46% win rate)
V3_AVOID_STRATEGIES = ['RSI', 'RSI_extreme']

# ============================================================================
# STOCK WHITELIST (limited validation data)
# ============================================================================

V3_WHITELIST_STOCKS = ['T', 'BMBL', 'XPO']

V3_STOCK_WIN_RATES = {
    ('T', 'DOWN'): (1.00, 94),      # 100% but only 94 trades - suspicious
    ('BMBL', 'UP'): (0.75, 50),     # 75%, 50 trades
    ('XPO', 'UP'): (0.72, 42),      # 72%, 42 trades
}

# ============================================================================
# HOLD PERIODS
# ============================================================================

DEFAULT_HOLD_HOURS = 72  # Changed from 48h based on backtest

HOLD_PERIOD_MAP = {
    'ETH': 72,   # 3 days
    'XRP': 168,  # 7 days
    'LINK': 72,  # 3 days
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

# Legacy thresholds (pre-V3)
CONFIDENCE_BUY_THRESHOLD = 0.85   # 85% for BUY signal
CONFIDENCE_SELL_THRESHOLD = 0.85  # 85% for SELL signal

# Price movement thresholds
WATCH_THRESHOLD = 0.02            # 2% move for WATCH classification
SIGNIFICANT_MOVE_PCT = 0.03       # 3% move to trigger update alert

# ============================================================================
# TARGET / STOP LOSS DEFAULTS
# ============================================================================

DEFAULT_TARGET_PCT = 0.05         # +5% target
DEFAULT_STOP_LOSS_PCT = 0.02      # -2% stop loss
DEFAULT_RISK_REWARD = 2.5         # Target R/R ratio

# ============================================================================
# TELEGRAM SCHEDULE (Central Time)
# ============================================================================

TOP_10_HOUR = 8                   # 8:00 AM CT - daily TOP 10 message
UPDATE_HOURS = [12, 16, 20]       # 12 PM, 4 PM, 8 PM - update checks

# ============================================================================
# LEARNING / ACCURACY THRESHOLDS
# ============================================================================

MIN_PREDICTIONS_FOR_ACCURACY = 10  # Need 10+ predictions before accuracy matters
ACCURACY_EXCLUDE_THRESHOLD = 0.40  # Exclude symbols with <40% accuracy
ACCURACY_BOOST_THRESHOLD = 0.70    # Boost symbols with >70% accuracy
ACCURACY_BOOST_AMOUNT = 0.15       # 15% confidence boost for high performers

# ============================================================================
# DATA VALIDATION
# ============================================================================

# Minimum valid crypto prices (anything below = corrupt data)
MIN_VALID_PRICES = {
    'BTC': 10000,
    'ETH': 500,
    'SOL': 5,
    'BNB': 100,
    'XRP': 0.10,
    'ADA': 0.05,
    'DOGE': 0.001,
    'LINK': 1.0,
    'AVAX': 5.0,
}

# Maximum valid crypto prices (anything above = corrupt data)
MAX_VALID_PRICES = {
    'BTC': 500000,
    'ETH': 50000,
    'SOL': 1000,
    'BNB': 2000,
    'XRP': 10,
    'ADA': 5,
    'DOGE': 1,
    'LINK': 200,
    'AVAX': 500,
}

# ============================================================================
# API / RATE LIMITING
# ============================================================================

PRICE_TTL_SECONDS = 60            # Cache price data for 60 seconds
NEWS_TTL_SECONDS = 300            # Cache news data for 5 minutes
PREDICTION_COOLDOWN_SECONDS = 60  # Minimum time between predictions for same symbol

# ============================================================================
# BACKTEST PARAMETERS
# ============================================================================

BACKTEST_START_DATE = '2025-01-01'
BACKTEST_END_DATE = '2026-01-31'
BACKTEST_SIGNIFICANCE_THRESHOLD = 0.05  # p-value threshold for "significant"
BACKTEST_MIN_TRADES = 50          # Minimum trades for reliable results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_hold_hours(symbol: str) -> int:
    """Get validated hold period for symbol."""
    return HOLD_PERIOD_MAP.get(symbol.upper(), DEFAULT_HOLD_HOURS)


def get_strategy(symbol: str) -> dict:
    """Get validated strategy config for symbol, or None if not validated."""
    return V3_VALIDATED_STRATEGIES.get(symbol.upper())


def is_validated(symbol: str) -> bool:
    """Check if symbol has backtest-validated edge."""
    return symbol.upper() in V3_VALIDATED_STRATEGIES


def is_removed(symbol: str) -> str:
    """Check if symbol was explicitly removed. Returns reason or empty string."""
    return V3_REMOVED_SYMBOLS.get(symbol.upper(), '')


def is_blacklisted(symbol: str) -> bool:
    """Check if symbol is blacklisted (consistent loser)."""
    return symbol.upper() in V3_BLACKLIST


def validate_price(symbol: str, price: float) -> bool:
    """Check if price is within valid range for symbol."""
    sym = symbol.upper()
    min_price = MIN_VALID_PRICES.get(sym, 0.0001)
    max_price = MAX_VALID_PRICES.get(sym, 1000000)
    return min_price <= price <= max_price
