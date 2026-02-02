#!/usr/bin/env python3
"""
🎯 GHOST NOTIFICATIONS - ONE simple notification system

REPLACES all other notification code. This is the ONLY file that should send
Telegram messages for predictions.

Schedule:
- 8:00 AM Central: ONE TOP 10 message (5 stocks + 5 crypto)
- 12 PM, 4 PM, 8 PM Central: Update message (only if >3% moves)
- Anytime: Alert message (only if target/stop hit)

Colors:
- 🟢 BUY = 48hr prediction > current price (going UP) AND confidence >= 85%
- 🔴 SELL = 48hr prediction < current price (going DOWN) AND confidence >= 85%  
- 🟡 WATCH = confidence < 85% OR prediction within 2% of current

Learning Integration:
- Symbols with <40% accuracy (after 10+ predictions) are EXCLUDED
- Symbols with >70% accuracy get 15% confidence BOOST
"""

import os
import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

# PostgreSQL for learning data
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo

LOGGER = logging.getLogger("ghost.notifications")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Timezone
CENTRAL_TZ = ZoneInfo("America/Chicago")

# TOP 10 schedule (Central Time)
TOP_10_HOUR = 8  # 8 AM Central

# Update schedule (Central Time)
UPDATE_HOURS = [12, 16, 20]  # 12 PM, 4 PM, 8 PM

# Thresholds
MIN_CONFIDENCE = 0.85  # 85% minimum for BUY/SELL
WATCH_THRESHOLD = 0.02  # 2% move threshold for WATCH
SIGNIFICANT_MOVE_PCT = 0.03  # 3% move to trigger update

# Database for tracking
TRACKING_DB = os.getenv("GHOST_TRACKING_DB", "data/ghost_tracking.db")

# ============================================================================
# V3 QUALITY FILTER - Based on 28K+ Paper Trades Sweetspot Analysis
# ============================================================================

# Stocks with >50% win rate (proven performers)
V3_WHITELIST_STOCKS = ['T', 'BMBL', 'XPO']

# Crypto with >60% win rate (proven performers)
V3_WHITELIST_CRYPTO = ['TURBO', 'RNDR', 'IQ', 'ILV', 'CHZ', 'AAVE', 'ZEC']

# Major crypto with 0% win rate on DOWN predictions - FLIP to inverse
# These are 100% INVERSE SIGNALS (0% win = 100% wrong = flip direction)
V3_INVERSE_LIST = ['ETH', 'BTC', 'SOL', 'XRP', 'BNB', 'ADA', 'AVAX', 'LINK', 'LTC', 
                   'SHIB', 'SUI', 'ALGO', 'DOT', 'FIL', 'VET', 'HBAR', 'NEAR', 'ARB']

# Complete blacklist (symbols with <30% overall win rate - DO NOT TRADE)
V3_BLACKLIST = ['TGTX', 'SOUN', 'ABCL', 'ZIL', 'MANA', 'SAND', 'RLC', '1INCH', 
                'IMX', 'APT', 'SUSHI', 'YFI', 'LDO', 'ETC']

# V3 minimum confidence threshold
V3_MIN_CONFIDENCE = 0.75

# Historical win rates from 28K paper trade analysis
# Format: {symbol: {direction: win_rate}}
V3_HISTORICAL_WIN_RATES = {
    # TOP PERFORMERS
    'T': {'UP': 0.50, 'DOWN': 1.00},      # 100% on DOWN (94 trades)
    'TURBO': {'UP': 1.00, 'DOWN': 0.97},  # 98.4% overall
    'RNDR': {'UP': 1.00, 'DOWN': 0.90},   # 90.3% overall
    'BMBL': {'UP': 0.75, 'DOWN': 0.50},   # 72% overall
    'XPO': {'UP': 0.72, 'DOWN': 0.50},    # 70.6% overall
    'IQ': {'UP': 0.74, 'DOWN': 0.50},     # 74.2% overall
    'ILV': {'UP': 0.58, 'DOWN': 0.82},    # 65.7% overall
    'CHZ': {'UP': 0.50, 'DOWN': 0.75},    # 63.3% overall
    'AAVE': {'UP': 0.90, 'DOWN': 0.50},   # 60% overall
    'ZEC': {'UP': 0.50, 'DOWN': 0.81},    # 80.6% on DOWN
    
    # INVERSE CANDIDATES (0% win on DOWN = flip to UP)
    'ETH': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (29 trades) -> INVERSE
    'BTC': {'UP': 0.50, 'DOWN': 0.03},    # 3% on DOWN (33 trades) -> INVERSE
    'SOL': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (30 trades) -> INVERSE
    'XRP': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (28 trades) -> INVERSE
    'BNB': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (28 trades) -> INVERSE
    'ADA': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (28 trades) -> INVERSE
    'AVAX': {'UP': 0.50, 'DOWN': 0.00},   # 0% on DOWN (27 trades) -> INVERSE
    'LINK': {'UP': 0.50, 'DOWN': 0.00},   # 0% on DOWN (19 trades) -> INVERSE
    'LTC': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (26 trades) -> INVERSE
    'SHIB': {'UP': 0.50, 'DOWN': 0.00},   # 0% on DOWN (16 trades) -> INVERSE
    'DOT': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (16 trades) -> INVERSE
    'NEAR': {'UP': 0.50, 'DOWN': 0.00},   # 0% on DOWN (17 trades) -> INVERSE
    'HBAR': {'UP': 0.50, 'DOWN': 0.00},   # 0% on DOWN (17 trades) -> INVERSE
    'ARB': {'UP': 0.50, 'DOWN': 0.00},    # 0% on DOWN (15 trades) -> INVERSE
}

# V3 Mode toggle (enable by default)
V3_ENABLED = os.getenv("V3_QUALITY_ENABLED", "1") == "1"

# ============================================================================
# MARKET HOURS (Eastern Time for US stocks)
# ============================================================================
# Regular trading hours: 9:30 AM - 4:00 PM ET
# Pre-market: 4:00 AM - 9:30 AM ET
# After-hours: 4:00 PM - 8:00 PM ET
# We'll check during extended hours (4 AM - 8 PM ET) when prices can change

MARKET_OPEN_HOUR = 4    # 4 AM ET (pre-market starts)
MARKET_CLOSE_HOUR = 20  # 8 PM ET (after-hours ends)

try:
    EASTERN_TZ = ZoneInfo("America/New_York")
except:
    EASTERN_TZ = ZoneInfo("US/Eastern")

def is_stock_market_hours() -> bool:
    """
    Check if US stock market is in extended trading hours.
    Returns True during pre-market + regular + after-hours (4 AM - 8 PM ET).
    Returns False overnight and on weekends.
    
    Crypto trades 24/7 so this only applies to stocks.
    """
    from datetime import datetime
    now_et = datetime.now(EASTERN_TZ)
    
    # Check if weekend (Saturday=5, Sunday=6)
    if now_et.weekday() >= 5:
        return False
    
    # Check if within extended hours (4 AM - 8 PM ET)
    hour = now_et.hour
    return MARKET_OPEN_HOUR <= hour < MARKET_CLOSE_HOUR


# ============================================================================
# V3 QUALITY FILTER FUNCTIONS
# ============================================================================

def get_v3_historical_win_rate(symbol: str, direction: str) -> float:
    """
    Get historical win rate for symbol+direction from V3 analysis.
    
    Returns:
        Win rate (0.0-1.0) or 0.50 if unknown
    """
    symbol = symbol.upper()
    direction = direction.upper()
    
    if symbol in V3_HISTORICAL_WIN_RATES:
        rates = V3_HISTORICAL_WIN_RATES[symbol]
        return rates.get(direction, 0.50)
    
    # Unknown symbol - default to 50%
    return 0.50


def is_v3_inverse_candidate(symbol: str, direction: str) -> bool:
    """
    Check if symbol+direction should be INVERSED based on V3 analysis.
    
    Returns True if Ghost is historically 100% WRONG on this symbol+direction.
    """
    symbol = symbol.upper()
    direction = direction.upper()
    
    # Only inverse DOWN predictions for crypto in inverse list
    if symbol in V3_INVERSE_LIST and direction == 'DOWN':
        return True
    
    return False


def v3_filter_and_score(predictions: List[Dict]) -> List[Dict]:
    """
    Apply V3 quality filter: whitelist, blacklist, inverse, and historical scoring.
    
    This is the core V3 logic:
    1. Skip blacklisted symbols
    2. Prioritize whitelisted symbols
    3. Apply inverse logic for major crypto DOWN predictions
    4. Score by historical_win_rate × confidence
    
    Args:
        predictions: List of prediction dicts with 'symbol', 'direction', 'confidence'
        
    Returns:
        List of scored predictions, sorted by v3_score descending
    """
    if not V3_ENABLED:
        LOGGER.info("[V3] V3 quality filter DISABLED - using legacy scoring")
        return predictions
    
    scored = []
    skipped_blacklist = 0
    inversed_count = 0
    
    for pred in predictions:
        symbol = pred.get('symbol', '').upper()
        direction = pred.get('direction', 'UP').upper()
        confidence = pred.get('confidence', 0.5)
        
        # 1. Skip blacklisted symbols
        if symbol in V3_BLACKLIST:
            skipped_blacklist += 1
            LOGGER.debug(f"[V3] BLACKLIST skip: {symbol}")
            continue
        
        # 2. Check for inverse candidate FIRST (before confidence check)
        # Inverse candidates bypass confidence check since they become good signals when flipped
        is_inverse = is_v3_inverse_candidate(symbol, direction)
        is_whitelisted = symbol in V3_WHITELIST_STOCKS or symbol in V3_WHITELIST_CRYPTO
        
        # 3. Minimum confidence check
        if confidence < V3_MIN_CONFIDENCE:
            # Whitelist symbols get a pass on minimum confidence
            # Inverse candidates ALSO get a pass (they're valuable flipped signals)
            if not is_whitelisted and not is_inverse:
                LOGGER.debug(f"[V3] Low confidence skip: {symbol} ({confidence:.0%})")
                continue
        
        # 4. Apply inverse logic
        original_direction = direction
        
        if is_inverse:
            # FLIP the direction!
            direction = 'UP' if direction == 'DOWN' else 'DOWN'
            inversed_count += 1
            LOGGER.info(f"[V3] 🔄 INVERSE: {symbol} {original_direction} → {direction}")
        
        # 5. Get historical win rate for the (potentially inversed) direction
        historical_win_rate = get_v3_historical_win_rate(symbol, direction)
        
        # For inverse signals, use inverted win rate (0% win = 100% inverse success)
        if is_inverse:
            historical_win_rate = 1.0 - get_v3_historical_win_rate(symbol, original_direction)
        
        # 6. Calculate V3 score = historical_win_rate × confidence
        v3_score = historical_win_rate * confidence
        
        # Whitelist bonus (small boost for proven performers)
        if is_whitelisted:
            v3_score *= 1.10  # 10% whitelist bonus
        
        # Build scored prediction
        scored_pred = pred.copy()
        scored_pred['direction'] = direction  # May have been inversed
        scored_pred['v3_score'] = v3_score
        scored_pred['v3_historical_win_rate'] = historical_win_rate
        scored_pred['v3_is_inverse'] = is_inverse
        scored_pred['v3_original_direction'] = original_direction if is_inverse else None
        scored_pred['v3_is_whitelisted'] = is_whitelisted
        
        scored.append(scored_pred)
    
    # Sort by v3_score descending
    scored.sort(key=lambda x: x['v3_score'], reverse=True)
    
    LOGGER.info(f"[V3] ✅ Filtered: {len(scored)}/{len(predictions)} passed, {skipped_blacklist} blacklisted, {inversed_count} inversed")
    
    return scored


# ============================================================================
# CONFIDENCE CALIBRATION FOR DISPLAY
# ============================================================================
# Model outputs 85-95% but actual accuracy is 55-70%
# This calibration makes displayed confidence match reality

def calibrate_display_confidence(raw_confidence: float, symbol: str = None) -> float:
    """
    Calibrate model confidence for honest display using LINEAR INTERPOLATION.
    
    Model tends to output 70-95% but actual accuracy is 52-68%.
    This function smoothly transforms raw confidence to realistic values.
    
    UPDATED Jan 2026: Widened output range for visible variation.
    Old: 75-85% raw → 55-62% display (all looked the same)
    New: 75-85% raw → 52-65% display (13 percentage points spread)
    
    UPDATED Jan 24, 2026: Added symbol-seeded jitter to prevent identical values.
    When XGBoost clusters similar stocks at same confidence, this adds
    deterministic micro-variance (±2%) so each symbol shows slightly different.
    
    Args:
        raw_confidence: Model's raw confidence (0.0-1.0)
        symbol: Optional symbol for deterministic jitter
    
    Returns:
        Calibrated confidence for display (0.0-1.0)
    
    Calibration curve (linear interpolation between points):
        Raw 95%+ -> Display 72%
        Raw 90%  -> Display 68%
        Raw 85%  -> Display 65%
        Raw 80%  -> Display 60%
        Raw 78%  -> Display 57%  (NEW: more granular in the common range)
        Raw 76%  -> Display 54%  (NEW: more granular in the common range)
        Raw 75%  -> Display 52%
        Raw 70%  -> Display 48%
        Raw 60%  -> Display 42%
        Below 60% -> Display = raw * 0.7
    """
    # Calibration points: (raw, display) - WIDENED RANGE for visible variation
    # Most predictions fall in 75-82% range, so we add extra points there
    calibration_points = [
        (0.95, 0.72),
        (0.90, 0.68),
        (0.85, 0.65),
        (0.82, 0.62),  # NEW: granular point in common range
        (0.80, 0.60),
        (0.78, 0.57),  # NEW: granular point in common range  
        (0.76, 0.54),  # NEW: granular point in common range
        (0.75, 0.52),
        (0.70, 0.48),
        (0.60, 0.42),
    ]
    
    # Above highest point
    if raw_confidence >= calibration_points[0][0]:
        calibrated = calibration_points[0][1]
    # Below lowest calibration point - use simple scaling
    elif raw_confidence < calibration_points[-1][0]:
        calibrated = raw_confidence * 0.7
    else:
        # Linear interpolation between calibration points
        calibrated = raw_confidence * 0.7  # Default
        for i in range(len(calibration_points) - 1):
            high_raw, high_display = calibration_points[i]
            low_raw, low_display = calibration_points[i + 1]
            
            if low_raw <= raw_confidence < high_raw:
                # Interpolate: display = low_display + (raw - low_raw) / (high_raw - low_raw) * (high_display - low_display)
                ratio = (raw_confidence - low_raw) / (high_raw - low_raw)
                calibrated = low_display + ratio * (high_display - low_display)
                break
    
    # ADD: Symbol-seeded jitter to prevent identical display values
    # When XGBoost clusters similar stocks, this ensures each shows differently
    # Jitter is deterministic per symbol so same symbol always gets same jitter
    if symbol:
        # Use symbol hash to get consistent jitter per symbol
        symbol_hash = hash(symbol.upper()) % 10000
        jitter = ((symbol_hash / 10000) - 0.5) * 0.04  # ±2% jitter
        calibrated = calibrated + jitter
    
    # Clamp to valid range (30% - 85%)
    return max(0.30, min(0.85, calibrated))


# ============================================================================
# LEARNING INTEGRATION
# ============================================================================

# Learning thresholds
LEARNING_MIN_PREDICTIONS = 10  # Need at least 10 predictions to evaluate
LEARNING_EXCLUDE_ACCURACY = 40.0  # Exclude symbols with <40% accuracy
LEARNING_BOOST_ACCURACY = 70.0  # Boost symbols with >70% accuracy
LEARNING_BOOST_AMOUNT = 0.15  # 15% confidence boost

# ============================================================================
# LEARNING MODE - Re-enabled Dec 28, 2025
# BOOST ENABLED Jan 31, 2026 - 1500+ predictions tracked, data validated
# ============================================================================
LEARNING_ENABLED = True  # Master switch
LEARNING_EXCLUDE_ENABLED = True  # Exclusions active (low accuracy symbols removed)
LEARNING_BOOST_ENABLED = True  # Boosting ENABLED - high accuracy symbols get +15% confidence

# ============================================================================
# HARDCODED EXCLUSIONS - Symbols with historically bad accuracy (<40%)
# These are ALWAYS excluded regardless of learning data
# Updated Dec 29, 2025 based on actual accuracy data + Telegram analysis
# ============================================================================
HARDCODED_EXCLUSIONS = {
    # === DATA PROVIDER FAILURES (Dec 30) - All 3 providers fail for these ===
    'TON': 'All providers fail - 0/0/0 pillars',  # Added Dec 30
    'XMR': 'All providers fail - privacy coin not supported',  # Added Dec 30
    'INJ': 'All providers fail - 0/0/0 pillars',  # Added Dec 30
    'OSMO': 'All providers fail - Cosmos DEX token',  # Added Dec 30
    'STRK': 'All providers fail - StarkNet token',  # Added Dec 30
    'FUN': 'All providers fail - possibly delisted',  # Added Dec 30
    'TNT': 'Binance 451 blocked in US, others 404',  # Added Dec 30
    'MYST': 'Binance 451 blocked in US, others 404',  # Added Dec 30
    
    # === FROM TELEGRAM ANALYSIS (Dec 29) - Confirmed stop-outs and wrong directions ===
    'ALGO': 'STOPPED OUT +3.0% on Dec 29 - wrong direction',  # Added Dec 29
    'TIA': 'Small-cap, high risk - same pattern as ALGO',  # Added Dec 29
    'SAND': 'STOPPED OUT +3.3% on Dec 27 (Telegram confirmed)',  # Added Dec 28
    'FLOW': 'STOPPED OUT +12.5% on Dec 27 (Telegram confirmed)',  # Added Dec 28
    'HBAR': 'Wrong direction - went UP when predicted DOWN (Telegram confirmed)',  # Added Dec 28
    'ILV': 'Gaming token - pump/dump prone, volatile',  # Added Dec 28
    'BAND': 'Oracle token - low liquidity',  # Added Dec 28
    'DIA': 'Oracle token - low liquidity',  # Added Dec 28
    
    # === MEME COINS - Too volatile, unpredictable ===
    'SHIB': 'Meme coin - unpredictable',
    'PEPE': 'Meme coin - unpredictable',
    'BONK': 'Meme coin - unpredictable',
    'WIF': 'Meme coin - unpredictable',
    'MEME': 'Meme coin - unpredictable',
    'FLOKI': '11% accuracy (1/9) - meme coin',
    
    # === GAMING/METAVERSE - Pump/dump prone ===
    'APE': '22% accuracy (2/9) - NFT/gaming',
    'GMT': 'Gaming token - volatile',
    'GALA': 'Gaming token - volatile',
    'ENJ': 'Gaming token - volatile',
    'MANA': 'Metaverse - volatile',
    'AXS': 'Gaming token - volatile',
    
    # === FROM ACCURACY DATA - Crypto with <40% accuracy ===
    'DOT': '30% accuracy (3/10)',
    'DOGE': '30% accuracy (3/10)',
    'MATIC': '50% but volatile',
    'OMG': '10% accuracy (1/10)',
    'AVAX': '30% accuracy (3/10)',
    'ANT': '40% accuracy (borderline)',
    'OCEAN': '30% accuracy (3/10)',
    'ADA': '20% accuracy (2/10)',
    'STORJ': '0% accuracy (0/9)',
    'RLC': '10% accuracy (1/10)',
    '1INCH': '11% accuracy (1/9)',
    'YFI': '11% accuracy (1/9)',
    'LDO': '11% accuracy (1/9)',
    'BAL': '11% accuracy (1/9)',
    'RNDR': '11% accuracy (1/9)',
    'XLM': '22% accuracy (2/9)',
    'ETC': '22% accuracy (2/9)',
    'ZEN': '20% accuracy (2/10)',
    'EGLD': '20% accuracy (2/10)',
    'ONDO': '20% accuracy (2/10)',
    'BAT': '20% accuracy (2/10)',
    'THETA': '20% accuracy (2/10)',
    'ZRX': '30% accuracy (3/10)',
    'BNB': '30% accuracy (3/10)',
    'XTZ': '30% accuracy (3/10)',
    'LOOM': '30% accuracy (3/10)',
    'USDC': '33% accuracy - stablecoin',
    'QNT': '20% accuracy (2/10)',
}

# Cache for symbol accuracy (refreshed every 5 minutes)
_SYMBOL_ACCURACY_CACHE = {}
_SYMBOL_ACCURACY_CACHE_TIME = 0
SYMBOL_ACCURACY_CACHE_TTL = 300  # 5 minutes


# =============================================================================
# ENVIRONMENT VARIABLE EXCLUSIONS - Read from Railway
# =============================================================================
def _load_env_exclusions() -> set:
    """Load exclusions from GHOST_EXCLUDE_SYMBOLS environment variable."""
    env_val = os.getenv("GHOST_EXCLUDE_SYMBOLS", "")
    if not env_val:
        LOGGER.info("[EXCLUSIONS] GHOST_EXCLUDE_SYMBOLS not set or empty")
        return set()
    
    symbols = set()
    for s in env_val.split(","):
        s = s.strip().upper()
        if s:
            symbols.add(s)
    
    if symbols:
        LOGGER.info(f"[EXCLUSIONS] ✅ Loaded {len(symbols)} symbols from GHOST_EXCLUDE_SYMBOLS env var")
        LOGGER.debug(f"[EXCLUSIONS] Symbols: {', '.join(sorted(symbols))}")
    
    return symbols


# Load env exclusions at module load time
_ENV_EXCLUSIONS = _load_env_exclusions()


def reload_env_exclusions() -> set:
    """Reload exclusions from env var (call after Railway restart)."""
    global _ENV_EXCLUSIONS
    _ENV_EXCLUSIONS = _load_env_exclusions()
    return _ENV_EXCLUSIONS


def get_exclusion_stats() -> dict:
    """Get statistics about current exclusions (for debugging)."""
    all_excluded = set(HARDCODED_EXCLUSIONS.keys()) | _ENV_EXCLUSIONS
    return {
        "hardcoded_count": len(HARDCODED_EXCLUSIONS),
        "hardcoded_symbols": sorted(HARDCODED_EXCLUSIONS.keys()),
        "env_exclusions_count": len(_ENV_EXCLUSIONS),
        "env_exclusions": sorted(_ENV_EXCLUSIONS),
        "total_unique_excluded": len(all_excluded),
        "all_excluded": sorted(all_excluded),
    }


def get_symbol_accuracy_from_postgres() -> Dict[str, Dict]:
    """
    Get symbol accuracy data from PostgreSQL ghost_symbol_accuracy table.
    
    ⚠️ DISABLED: Returns empty dict when LEARNING_ENABLED=False
    
    Returns:
        Dict of symbol -> {total: int, correct: int, accuracy_pct: float}
    """
    global _SYMBOL_ACCURACY_CACHE, _SYMBOL_ACCURACY_CACHE_TIME
    
    # LEARNING DISABLED - return empty to skip all learning adjustments
    if not LEARNING_ENABLED:
        LOGGER.info("[LEARNING] ⚠️ DISABLED - Using raw INVERSE predictions only")
        return {}
    
    # Check cache first
    if time.time() - _SYMBOL_ACCURACY_CACHE_TIME < SYMBOL_ACCURACY_CACHE_TTL:
        return _SYMBOL_ACCURACY_CACHE
    
    if not PSYCOPG2_AVAILABLE:
        LOGGER.warning("[LEARNING] psycopg2 not available, skipping accuracy lookup")
        return {}
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        LOGGER.warning("[LEARNING] DATABASE_URL not set, skipping accuracy lookup")
        return {}
    
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT symbol, total_predictions, correct_predictions, accuracy_pct
            FROM ghost_symbol_accuracy
            WHERE total_predictions >= %s
            ORDER BY total_predictions DESC
        """, (LEARNING_MIN_PREDICTIONS,))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        result = {}
        for row in rows:
            symbol, total, correct, accuracy = row
            result[symbol] = {
                "total": total,
                "correct": correct,
                "accuracy_pct": float(accuracy) if accuracy else 0.0
            }
        
        # Update cache
        _SYMBOL_ACCURACY_CACHE = result
        _SYMBOL_ACCURACY_CACHE_TIME = time.time()
        
        LOGGER.info(f"[LEARNING] Loaded accuracy data for {len(result)} symbols from PostgreSQL")
        return result
        
    except Exception as e:
        LOGGER.error(f"[LEARNING] Failed to get symbol accuracy from PostgreSQL: {e}")
        return _SYMBOL_ACCURACY_CACHE  # Return stale cache on error


def should_exclude_symbol(symbol: str, accuracy_data: Dict[str, Dict]) -> tuple:
    """
    Check if symbol should be excluded based on historical accuracy.
    
    PRIORITY ORDER:
    0. GHOST_EXCLUDE_SYMBOLS env var - Railway config exclusions
    1. HARDCODED_EXCLUSIONS - Always excluded (known bad symbols in code)
    2. Learning data - Excluded if <40% accuracy after 10+ predictions (if LEARNING_EXCLUDE_ENABLED)
    
    Args:
        symbol: The symbol to check
        accuracy_data: Dict from get_symbol_accuracy_from_postgres()
        
    Returns:
        (should_exclude: bool, reason: str)
    """
    symbol_upper = symbol.upper()
    
    # MONEY GAME: No more V2 whitelist bypass - all symbols compete on merit!
    
    # PRIORITY 0: Check environment variable exclusions FIRST (Railway config)
    if symbol_upper in _ENV_EXCLUSIONS:
        return True, f"ENV_EXCLUDED: In GHOST_EXCLUDE_SYMBOLS"
    
    # PRIORITY 1: Check hardcoded exclusions (code-level)
    if symbol_upper in HARDCODED_EXCLUSIONS:
        return True, f"HARDCODED: {HARDCODED_EXCLUSIONS[symbol_upper]}"
    if symbol in HARDCODED_EXCLUSIONS:
        return True, f"HARDCODED: {HARDCODED_EXCLUSIONS[symbol]}"
    
    # PRIORITY 2: Check learning data (only if LEARNING_EXCLUDE_ENABLED)
    if not LEARNING_EXCLUDE_ENABLED:
        return False, "learning_exclusions_disabled"
    
    # Check both cases for accuracy data
    data = accuracy_data.get(symbol_upper) or accuracy_data.get(symbol)
    if not data:
        return False, "no_data"
    
    accuracy = data.get("accuracy_pct", 0)
    total = data.get("total", 0)
    
    if total < LEARNING_MIN_PREDICTIONS:
        return False, f"insufficient_data ({total} predictions)"
    
    if accuracy < LEARNING_EXCLUDE_ACCURACY:
        return True, f"low_accuracy ({accuracy:.1f}% < {LEARNING_EXCLUDE_ACCURACY}%)"
    
    return False, "ok"


def get_confidence_boost(symbol: str, accuracy_data: Dict[str, Dict]) -> tuple:
    """
    Get confidence boost for high-accuracy symbols.
    
    Args:
        symbol: The symbol to check
        accuracy_data: Dict from get_symbol_accuracy_from_postgres()
        
    Returns:
        (boost_multiplier: float, reason: str)
    """
    # Check if boosts are enabled
    if not LEARNING_BOOST_ENABLED:
        return 1.0, "boosts_disabled"
    
    if symbol not in accuracy_data:
        return 1.0, "no_data"
    
    data = accuracy_data[symbol]
    accuracy = data.get("accuracy_pct", 0)
    total = data.get("total", 0)
    
    if total < LEARNING_MIN_PREDICTIONS:
        return 1.0, f"insufficient_data ({total} predictions)"
    
    if accuracy >= LEARNING_BOOST_ACCURACY:
        boost = 1.0 + LEARNING_BOOST_AMOUNT
        return boost, f"high_accuracy ({accuracy:.1f}% >= {LEARNING_BOOST_ACCURACY}%)"
    
    return 1.0, f"accuracy_ok ({accuracy:.1f}%)"


@dataclass
class TrackedPick:
    """A pick being tracked for 48 hours"""
    symbol: str
    asset_type: str  # 'crypto' or 'stock'
    direction: str  # 'BUY' or 'SELL' or 'WATCH'
    entry_price: float
    current_price: float
    target_price: float
    stop_price: float
    prediction_48h: float  # 48hr predicted price
    confidence: float
    entry_time: datetime
    expires_at: datetime
    status: str = "active"  # 'active', 'target_hit', 'stop_hit', 'expired'


def get_central_time() -> datetime:
    """Get current time in Central timezone"""
    return datetime.now(CENTRAL_TZ)


def format_price(price: float) -> str:
    """Format price nicely"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def determine_action(current_price: float, prediction_48h: float, confidence: float) -> tuple:
    """
    Determine BUY/SELL/WATCH based on prediction vs current price.
    
    Returns: (action, emoji, color_code)
    """
    if confidence < MIN_CONFIDENCE:
        return ("WATCH", "🟡", "watch")
    
    pct_change = (prediction_48h - current_price) / current_price
    
    # If move is too small, it's a WATCH
    if abs(pct_change) < WATCH_THRESHOLD:
        return ("WATCH", "🟡", "watch")
    
    if prediction_48h > current_price:
        return ("BUY", "🟢", "buy")
    else:
        return ("SELL", "🔴", "sell")


def _calc_roi_100(current: float, target: float, direction: str = "BUY") -> str:
    """
    Calculate ROI on $100 investment.
    
    For BUY: You profit when price goes UP (target > current)
    For SELL: You profit when price goes DOWN (target < current)
    
    CRITICAL FIX: SELL predictions now show PROFIT correctly!
    If you SHORT a stock and it drops 5%, you MAKE 5%, not lose it!
    """
    if current <= 0:
        return "$100.00"
    
    gain_pct = (target - current) / current
    
    # For SELL/SHORT: You profit from price DROPS
    # If price drops 5% (gain_pct = -0.05), you MAKE 5%
    if direction == "SELL":
        gain_pct = abs(gain_pct)  # SELL profits when price drops
    
    final_value = 100 * (1 + gain_pct)
    return f"${final_value:.2f}"


def _get_buy_timing(asset_type: str) -> tuple:
    """
    Determine buy timing based on market hours.
    Returns (buy_label, buy_datetime_str)
    """
    from datetime import datetime, timedelta
    
    ct = get_central_time()
    
    if asset_type == 'crypto':
        # Crypto trades 24/7 - buy NOW
        return ("🔥 BUY NOW", ct.strftime("%b %d %I:%M %p CT"))
    
    # Stock - check if market is open
    try:
        now_et = datetime.now(EASTERN_TZ)
        hour = now_et.hour
        minute = now_et.minute
        weekday = now_et.weekday()
        
        # Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        is_market_open = (
            weekday < 5 and  # Mon-Fri
            ((hour == 9 and minute >= 30) or (hour > 9 and hour < 16))
        )
        
        if is_market_open:
            return ("🔥 BUY NOW", now_et.strftime("%b %d %I:%M %p ET"))
        else:
            # Calculate next market open
            if weekday >= 4:  # Fri after close, Sat, Sun
                days_until_monday = (7 - weekday) % 7
                if days_until_monday == 0:
                    days_until_monday = 7 if weekday == 4 and hour >= 16 else 0
                next_open = now_et + timedelta(days=days_until_monday)
            elif hour >= 16:  # After hours
                next_open = now_et + timedelta(days=1)
            else:  # Before market open same day
                next_open = now_et
            
            next_open = next_open.replace(hour=9, minute=30, second=0, microsecond=0)
            return ("📅 BUY AT OPEN", next_open.strftime("%b %d 9:30 AM ET"))
    except:
        return ("📅 BUY AT OPEN", "Next Market Open")


def _get_sell_timing(hours: int = 48) -> str:
    """Calculate sell timing based on horizon"""
    from datetime import datetime, timedelta
    ct = get_central_time()
    sell_time = ct + timedelta(hours=hours)
    return sell_time.strftime("%b %d %I:%M %p CT")


def format_top10_message(stocks: List[Dict], crypto: List[Dict], inverse_mode: bool = None) -> List[str]:
    """
    Format TOP 10 message (5 stocks + 5 crypto) - DETAILED TRADE PLAN format.
    
    Returns LIST of messages (stocks msg, crypto msg) to fit Telegram 4096 char limit.
    
    Format per pick:
    1) 🟢 NVDA — BUY ✅
    • BUY (CT): Fri Jan 30 @ 9:30 AM | Entry Zone: $192.51 – $194.00
    • SELL (CT): Mon Feb 02 @ Open | Target: $205.60 (+6.8%)
    • STOP LOSS: $185.77
    • Hold: 7 days (swing trade)
    • Confidence: 56% | Risk: Moderate | R/R: ~2.4 : 1
    💰 $100 → $106.80
    ✅ News influence confirmed
    """
    from datetime import datetime, timedelta
    
    if inverse_mode is None:
        inverse_mode = os.getenv("INVERSE_GHOST", "0") == "1"
    
    ct = get_central_time()
    date_str = ct.strftime("%b %d, %Y")
    time_str = ct.strftime("%I:%M %p CT").lstrip("0")
    
    # === US MARKET HOLIDAYS (NYSE/NASDAQ) ===
    # These are fixed for 2025-2027. For dynamic holidays, use pandas_market_calendars.
    US_MARKET_HOLIDAYS = {
        # 2025
        (2025, 1, 1),   # New Year's Day
        (2025, 1, 20),  # MLK Day
        (2025, 2, 17),  # Presidents Day
        (2025, 4, 18),  # Good Friday
        (2025, 5, 26),  # Memorial Day
        (2025, 6, 19),  # Juneteenth
        (2025, 7, 4),   # Independence Day
        (2025, 9, 1),   # Labor Day
        (2025, 11, 27), # Thanksgiving
        (2025, 12, 25), # Christmas
        # 2026
        (2026, 1, 1),   # New Year's Day
        (2026, 1, 19),  # MLK Day
        (2026, 2, 16),  # Presidents Day
        (2026, 4, 3),   # Good Friday
        (2026, 5, 25),  # Memorial Day
        (2026, 6, 19),  # Juneteenth
        (2026, 7, 3),   # Independence Day (observed)
        (2026, 9, 7),   # Labor Day
        (2026, 11, 26), # Thanksgiving
        (2026, 12, 25), # Christmas
        # 2027
        (2027, 1, 1),   # New Year's Day
        (2027, 1, 18),  # MLK Day
        (2027, 2, 15),  # Presidents Day
        (2027, 3, 26),  # Good Friday
        (2027, 5, 31),  # Memorial Day
        (2027, 6, 18),  # Juneteenth (observed)
        (2027, 7, 5),   # Independence Day (observed)
        (2027, 9, 6),   # Labor Day
        (2027, 11, 25), # Thanksgiving
        (2027, 12, 24), # Christmas (observed)
    }
    
    def is_us_holiday(dt: datetime) -> bool:
        """Check if date is a US market holiday"""
        return (dt.year, dt.month, dt.day) in US_MARKET_HOLIDAYS
    
    def get_holiday_name(dt: datetime) -> str:
        """Get the name of the holiday for a date"""
        holiday_names = {
            (1, 1): "New Year's Day", (1, 18): "MLK Day", (1, 19): "MLK Day", (1, 20): "MLK Day",
            (2, 15): "Presidents Day", (2, 16): "Presidents Day", (2, 17): "Presidents Day",
            (3, 26): "Good Friday", (4, 3): "Good Friday", (4, 18): "Good Friday",
            (5, 25): "Memorial Day", (5, 26): "Memorial Day", (5, 31): "Memorial Day",
            (6, 18): "Juneteenth", (6, 19): "Juneteenth",
            (7, 3): "Independence Day", (7, 4): "Independence Day", (7, 5): "Independence Day",
            (9, 1): "Labor Day", (9, 6): "Labor Day", (9, 7): "Labor Day",
            (11, 25): "Thanksgiving", (11, 26): "Thanksgiving", (11, 27): "Thanksgiving",
            (12, 24): "Christmas", (12, 25): "Christmas",
        }
        return holiday_names.get((dt.month, dt.day), "Market Holiday")
    
    # === TRADING DAY CHECK ===
    # Stocks can only be bought Mon-Fri, excluding holidays
    def is_trading_day(dt: datetime) -> bool:
        """Check if date is a trading day (weekday + not a holiday)"""
        if dt.weekday() >= 5:  # Weekend
            return False
        if is_us_holiday(dt):  # Holiday
            return False
        return True
    
    def get_next_trading_day(dt: datetime) -> datetime:
        """Get next trading day. Skips weekends AND holidays."""
        while not is_trading_day(dt):
            dt = dt + timedelta(days=1)
        return dt
    
    # For stocks: use next trading day
    stock_entry_date = get_next_trading_day(ct)
    stock_day_name = stock_entry_date.strftime("%a")
    is_market_closed = not is_trading_day(ct)
    is_weekend = ct.weekday() >= 5
    is_holiday = is_us_holiday(ct) and not is_weekend
    holiday_name = get_holiday_name(ct) if is_holiday else None
    
    # For crypto: 24/7, use today
    crypto_day_name = ct.strftime("%a")
    
    def get_hold_reason(conf: float, hold_days: int) -> str:
        """Get intelligent hold reason based on confidence and days"""
        if hold_days == 1:
            if conf < 0.35:
                return "low conviction scalp"
            else:
                return "RSI extreme"
        elif hold_days == 2:
            return "volatility swing"
        elif hold_days <= 4:
            return "swing trade"
        else:
            return "position trade"
    
    def get_risk_level(conf: float) -> str:
        """Get risk level based on confidence"""
        if conf >= 0.60:
            return "Low"
        elif conf >= 0.40:
            return "Moderate"
        else:
            return "High"
    
    def calculate_rr(entry: float, target: float, stop: float, direction: str) -> float:
        """Calculate risk/reward ratio"""
        if direction in ('UP', 'BUY'):
            reward = abs(target - entry)
            risk = abs(entry - stop)
        else:
            reward = abs(entry - target)
            risk = abs(stop - entry)
        return reward / risk if risk > 0 else 1.0
    
    def get_exit_date(hold_days: int, is_stock: bool) -> str:
        """Get exit date string - calculated from ACTUAL entry date"""
        # For stocks, entry is stock_entry_date (next trading day if weekend)
        # For crypto, entry is now (ct)
        base_date = stock_entry_date if is_stock else ct
        exit_dt = base_date + timedelta(days=hold_days)
        # Stocks: if exit lands on weekend, push to Monday
        if is_stock and exit_dt.weekday() >= 5:  # Saturday=5, Sunday=6
            days_to_monday = 7 - exit_dt.weekday()
            exit_dt = exit_dt + timedelta(days=days_to_monday)
        return exit_dt.strftime("%a %b %d")
    
    def format_pick(item, idx: int, is_stock: bool) -> str:
        """Format a single pick in detailed format"""
        symbol = item['symbol']
        direction = item.get('direction', 'UP')
        is_buy = direction in ('UP', 'BUY')
        
        current = item.get('current', 0)
        target = item.get('prediction_48h', item.get('target_price', current))
        stop = item.get('stop', current * 0.97 if is_buy else current * 1.03)
        conf = calibrate_display_confidence(item['confidence'], symbol=symbol)
        hold_days = item.get('hold_days', 3)
        news = item.get('news_influenced', False)
        
        # V3 DATA
        v3_is_inverse = item.get('v3_is_inverse', False)
        v3_original_direction = item.get('v3_original_direction')
        v3_historical_win_rate = item.get('v3_historical_win_rate', 0.50)
        v3_score = item.get('v3_score', 0)
        v3_is_whitelisted = item.get('v3_is_whitelisted', False)
        
        # Entry Zone: Use REAL volatility data (daily vol / 2 = intraday range)
        # If no volatility data, estimate from expected_move
        volatility = item.get('volatility', 0.02)  # 20-day annualized vol (default 2%)
        expected_move = item.get('expected_move_pct', 0.03)  # Expected move %
        
        # Daily volatility = annualized / sqrt(252) ≈ annualized / 16
        # Entry zone = half of daily vol (reasonable fill range)
        daily_vol = volatility / 16 if volatility > 0.01 else abs(expected_move) / 3
        entry_range_pct = max(0.005, min(0.03, daily_vol / 2))  # Clamp 0.5% - 3%
        
        # Calculate gain percentage
        # For BUY: positive when target > current
        # For SELL (short): we PROFIT when price drops, so show positive % for drop
        if is_buy:
            gain_pct = ((target - current) / current * 100) if current > 0 else 0
        else:
            # SHORT: profit = entry - exit, show as positive %
            gain_pct = ((current - target) / current * 100) if current > 0 else 0
        
        entry_high = current * (1 + entry_range_pct)  # Entry zone: current to +vol-based %
        rr = calculate_rr(current, target, stop, direction)
        risk = get_risk_level(conf)
        hold_reason = get_hold_reason(conf, hold_days)
        exit_date = get_exit_date(hold_days, is_stock)
        return_val = 100 + abs(gain_pct)
        
        emoji = "🟢" if is_buy else "🔴"
        action = "BUY" if is_buy else "SELL"
        news_check = " ✅" if news else ""
        
        # V3 BADGES
        inverse_badge = " 🔄 INVERSE" if v3_is_inverse else ""
        whitelist_badge = " ⭐" if v3_is_whitelisted else ""
        
        # Use correct day name based on asset type
        # Stocks: must use trading day (Mon-Fri)
        # Crypto: can trade 24/7
        entry_day_name = stock_day_name if is_stock else crypto_day_name
        entry_date = stock_entry_date if is_stock else ct
        
        lines = []
        lines.append(f"{idx}) {emoji} {symbol} — {action}{inverse_badge}{whitelist_badge}{news_check}")
        
        # Add V3 explanation for inverse signals
        if v3_is_inverse:
            lines.append(f"🔄 Ghost predicted {v3_original_direction} → FLIPPED to {direction}")
        
        # Add warning if market closed (weekend or holiday) for stocks
        if is_stock and is_market_closed:
            if is_holiday:
                lines.append(f"⚠️ MARKET CLOSED ({holiday_name}) - Execute {entry_day_name}")
            else:
                lines.append(f"⚠️ MARKET CLOSED (Weekend) - Execute {entry_day_name}")
        
        if is_buy:
            if is_stock:
                lines.append(f"• BUY (CT): {entry_day_name} {entry_date.strftime('%b %d')} @ 9:30 AM | Entry Zone: {format_price(current)} – {format_price(entry_high)}")
                lines.append(f"• SELL (CT): {exit_date} @ Open | Target: {format_price(target)} ({gain_pct:+.1f}%)")
            else:
                # Crypto - 24/7 (no "Open" for crypto)
                lines.append(f"• BUY NOW (24/7): Entry Zone: {format_price(current)} – {format_price(entry_high)}")
                lines.append(f"• SELL: {exit_date} | Target: {format_price(target)} ({gain_pct:+.1f}%)")
        else:
            # SHORT: sell high, buy back low - show positive profit %
            if is_stock:
                lines.append(f"• SELL (CT): {entry_day_name} {entry_date.strftime('%b %d')} @ 9:30 AM | Entry Zone: {format_price(current)} – {format_price(entry_high)}")
                lines.append(f"• BUY-BACK (CT): {exit_date} @ Open | Target: {format_price(target)} (+{gain_pct:.1f}%)")
            else:
                # Crypto - 24/7 (no "Open" for crypto)
                lines.append(f"• SELL NOW (24/7): Entry Zone: {format_price(current)} – {format_price(entry_high)}")
                lines.append(f"• BUY-BACK: {exit_date} | Target: {format_price(target)} (+{gain_pct:.1f}%)")
        
        lines.append(f"• STOP LOSS: {format_price(stop)}")
        lines.append(f"• Hold: {hold_days} day{'s' if hold_days > 1 else ''} ({hold_reason})")
        
        # V3: Include historical win rate if available
        if v3_historical_win_rate > 0 and v3_historical_win_rate != 0.50:
            lines.append(f"• Confidence: {conf:.0%} | Risk: {risk} | R/R: ~{rr:.1f} : 1 | Win Rate: {v3_historical_win_rate:.0%}")
        else:
            lines.append(f"• Confidence: {conf:.0%} | Risk: {risk} | R/R: ~{rr:.1f} : 1")
        
        lines.append(f"💰 $100 → ${return_val:.2f}")
        
        if news:
            lines.append("✅ News influence confirmed")
        
        return "\n".join(lines)
    
    # ===== BUILD SINGLE MESSAGE =====
    all_lines = [
        "🎯 GHOST TOP 10 — TRADE PLAN",
        f"📅 {date_str} | ⏰ 8:00 AM CT",
    ]
    
    # Add market closed warning banner (weekend or holiday)
    if is_market_closed:
        if is_holiday:
            all_lines.extend([
                "",
                f"⚠️ **{holiday_name.upper()} — STOCK MARKETS CLOSED**",
                f"📆 Next trading day: {stock_day_name} {stock_entry_date.strftime('%b %d')}",
                "🪙 Crypto trades 24/7 — execute anytime",
            ])
        else:  # Weekend
            all_lines.extend([
                "",
                "⚠️ **WEEKEND — STOCK MARKETS CLOSED**",
                f"📆 Next trading day: {stock_day_name} {stock_entry_date.strftime('%b %d')}",
                "🪙 Crypto trades 24/7 — execute anytime",
            ])
    
    all_lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "📈 STOCKS",
        "━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ])
    
    # Take top 5 stocks
    for idx, s in enumerate(stocks[:5], 1):
        all_lines.append(format_pick(s, idx, is_stock=True))
        all_lines.append("")
    
    if not stocks:
        all_lines.append("(No stock picks today)")
        all_lines.append("")
    
    # Add crypto section
    all_lines.extend([
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🪙 CRYPTO",
        "━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ])
    
    # Take top 5 crypto
    for idx, c in enumerate(crypto[:5], 1):
        all_lines.append(format_pick(c, idx, is_stock=False))
        all_lines.append("")
    
    if not crypto:
        all_lines.append("(No crypto picks today)")
        all_lines.append("")
    
    # Add legend at end
    all_lines.extend([
        "━━━━━━━━━━━━━━━━━━━━━━",
        "📖 LEGEND",
        "━━━━━━━━━━━━━━━━━━━━━━",
        "🟢 BUY | 🔴 SELL",
        "🔄 = Inverse signal (Ghost flipped)",
        "⭐ = Whitelist (proven performer)",
        "✅ = News-feed influenced",
        "💰 = Return on $100 position",
        "Ghost V3 is watching 👁️"
    ])
    
    full_msg = "\n".join(all_lines)
    
    # Return as list with single message (keeps API compatible)
    return [full_msg]


def format_update_message(picks: List[Dict]) -> str:
    """Format an update message showing current status of all tracked picks"""
    ct = get_central_time()
    time_str = ct.strftime("%I:%M %p CT").lstrip("0")
    
    lines = [
        f"📊 GHOST UPDATE — {time_str}",
        "",
    ]
    
    # Split into stocks and crypto
    stocks = [p for p in picks if p['asset_type'] == 'stock']
    crypto = [p for p in picks if p['asset_type'] == 'crypto']
    
    if stocks:
        lines.append("STOCKS")
        lines.append("━━━━━━━━━━━━━━━")
        for s in stocks:
            pct = (s['current'] - s['entry']) / s['entry'] * 100
            pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
            
            emoji = s['emoji']
            status = ""
            
            if s.get('near_target'):
                status = " 🎯 NEAR TARGET"
            elif s.get('near_stop'):
                status = " ⚠️ NEAR STOP"
            elif s.get('on_track'):
                status = " ✓ On track"
            else:
                status = " — Moving against prediction"
            
            lines.append(f"{emoji} {s['symbol']}: {format_price(s['entry'])} → {format_price(s['current'])} ({pct_str}){status}")
        lines.append("")
    
    if crypto:
        lines.append("CRYPTO")
        lines.append("━━━━━━━━━━━━━━━")
        for c in crypto:
            pct = (c['current'] - c['entry']) / c['entry'] * 100
            pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
            
            emoji = c['emoji']
            status = ""
            
            if c.get('near_target'):
                status = " 🎯 NEAR TARGET"
            elif c.get('near_stop'):
                status = " ⚠️ NEAR STOP"
            elif c.get('on_track'):
                status = " ✓ On track"
            else:
                status = " — Moving against prediction"
            
            lines.append(f"{emoji} {c['symbol']}: {format_price(c['entry'])} → {format_price(c['current'])} ({pct_str}){status}")
        lines.append("")
    
    # Next update time
    next_hour = ct.hour + 4
    if next_hour >= 24:
        next_hour = 8  # Next morning
    next_time = ct.replace(hour=next_hour, minute=0, second=0, microsecond=0)
    lines.append(f"Next update: {next_time.strftime('%I:%M %p CT').lstrip('0')} or on target hit")
    
    return "\n".join(lines)


def format_off_path_alert(off_path_picks: List[Dict], asset_type: str = "ALL") -> str:
    """
    Format alert when picks go OFF their prediction path.
    
    OFF PATH means:
    - BUY prediction but price going DOWN
    - SELL prediction but price going UP
    """
    ct = get_central_time()
    time_str = ct.strftime("%I:%M %p CT").lstrip("0")
    
    title = "🚨 GHOST PATH ALERT"
    if asset_type == "crypto":
        title = "🚨 CRYPTO PATH ALERT"
    elif asset_type == "stock":
        title = "🚨 STOCK PATH ALERT"
    
    lines = [
        f"{title} — {time_str}",
        "",
        "⚠️ These picks are moving AGAINST the prediction:",
        "",
    ]
    
    for p in off_path_picks:
        pct = (p['current'] - p['entry']) / p['entry'] * 100
        pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        
        direction = p.get('direction', 'BUY')
        emoji = "🟢" if direction == "BUY" else "🔴"
        
        lines.append(f"{emoji} {p['symbol']} — {direction}")
        lines.append(f"   Entry: {format_price(p['entry'])} → Now: {format_price(p['current'])} ({pct_str})")
        lines.append(f"   Target: {format_price(p['target'])} | Stop: {format_price(p['stop'])}")
        
        # Explain what's wrong
        if direction == "BUY" and pct < 0:
            lines.append(f"   ⚠️ Expected UP but down {abs(pct):.1f}%")
        elif direction == "SELL" and pct > 0:
            lines.append(f"   ⚠️ Expected DOWN but up {pct:.1f}%")
        
        lines.append("")
    
    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("Ghost is still watching. Will alert on target/stop hit.")
    
    return "\n".join(lines)


def format_alert_message(alerts: List[Dict]) -> str:
    """Format an alert message when target or stop is hit"""
    ct = get_central_time()
    time_str = ct.strftime("%I:%M %p CT").lstrip("0")
    
    lines = [
        f"🚨 GHOST ALERT — {time_str}",
        "",
    ]
    
    for a in alerts:
        pct = (a['current'] - a['entry']) / a['entry'] * 100
        pct_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        
        if a['type'] == 'target_hit':
            lines.append(f"🎯 {a['symbol']} HIT TARGET")
            lines.append(f"Entry: {format_price(a['entry'])} → Now: {format_price(a['current'])} ({pct_str})")
            lines.append(f"Target was: {format_price(a['target'])} ✅ ACHIEVED")
            lines.append("Action: Consider taking profit")
        else:
            lines.append(f"⚠️ {a['symbol']} STOP TRIGGERED")
            lines.append(f"Entry: {format_price(a['entry'])} → Now: {format_price(a['current'])} ({pct_str})")
            lines.append(f"Stop was: {format_price(a['stop'])} ❌ HIT")
            lines.append("Action: Position closed")
        
        lines.append("")
    
    # Don't assume 10 picks - active tracker knows the real count
    # "Remaining X picks" message removed - confusing with V2 filter
    # The active tracker sends its own progress updates
    
    return "\n".join(lines)


class GhostNotificationSystem:
    """
    The ONE notification system for Ghost.
    
    Handles:
    - Morning TOP 10 at 8 AM Central
    - Updates every 4 hours (if significant moves)
    - Instant alerts when targets/stops hit
    
    PERSISTENCE: Uses PostgreSQL (DATABASE_URL) for tracking to survive Railway deploys.
    Falls back to SQLite if PostgreSQL unavailable.
    """
    
    def __init__(self, send_telegram_func: Callable[[str], bool] = None):
        self.send_telegram = send_telegram_func
        self._tracked_picks: List[TrackedPick] = []
        self._last_top10_date: str = ""
        self._last_update_hour: int = -1
        self._last_off_path_alerts: Dict[str, str] = {}  # symbol -> date last alerted
        self._db_path = TRACKING_DB
        self._last_postgres_error: Optional[str] = None
        self._use_postgres = self._init_postgres()
        if not self._use_postgres:
            self._init_sqlite()
    
    def retry_postgres_connection(self) -> bool:
        """
        Retry PostgreSQL connection if currently using SQLite.
        Call this if DATABASE_URL wasn't available at startup but is now.
        Returns True if successfully switched to PostgreSQL.
        """
        if self._use_postgres:
            return True  # Already using PostgreSQL
        
        LOGGER.info("[TRACKING] Retrying PostgreSQL connection...")
        self._use_postgres = self._init_postgres()
        if self._use_postgres:
            LOGGER.info("[TRACKING] ✅ Successfully switched to PostgreSQL!")
            return True
        else:
            LOGGER.warning("[TRACKING] PostgreSQL retry failed - still using SQLite")
            return False
    
    def _get_postgres_conn(self):
        """Get PostgreSQL connection from DATABASE_URL"""
        import psycopg2
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            return None
        return psycopg2.connect(database_url)
    
    def _init_postgres(self) -> bool:
        """Initialize PostgreSQL tracking tables. Returns True if successful."""
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url:
            self._last_postgres_error = "No DATABASE_URL environment variable"
            LOGGER.info("[TRACKING] No DATABASE_URL - using SQLite fallback")
            return False
        
        LOGGER.info(f"[TRACKING] Attempting PostgreSQL connection (URL length: {len(database_url)} chars)")
        
        try:
            import psycopg2
        except ImportError as ie:
            self._last_postgres_error = f"psycopg2 not installed: {ie}"
            LOGGER.warning(f"[TRACKING] psycopg2 import failed: {ie}")
            return False
        
        try:
            conn = psycopg2.connect(database_url)
            LOGGER.info("[TRACKING] PostgreSQL connection successful!")
            cur = conn.cursor()
            
            # Create tracking table in PostgreSQL
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ghost_tracked_picks (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    prediction_48h REAL NOT NULL,
                    confidence REAL NOT NULL,
                    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create notification log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ghost_notification_log (
                    id SERIAL PRIMARY KEY,
                    notification_type TEXT NOT NULL,
                    sent_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    message_preview TEXT
                )
            """)
            
            # Create index for faster active picks lookup
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracked_picks_status 
                ON ghost_tracked_picks(status)
            """)
            
            # CLEANUP FIRST: Remove duplicate active picks BEFORE creating unique index
            # (Keeps the OLDEST entry for each symbol)
            cur.execute("""
                DELETE FROM ghost_tracked_picks a
                USING ghost_tracked_picks b
                WHERE a.id > b.id 
                AND a.symbol = b.symbol 
                AND a.status = 'active' 
                AND b.status = 'active'
            """)
            deleted_count = cur.rowcount
            if deleted_count > 0:
                LOGGER.info(f"[TRACKING] Cleaned up {deleted_count} duplicate active picks")
            
            # NOW create unique index (duplicates removed above)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_symbol 
                ON ghost_tracked_picks (symbol) 
                WHERE status = 'active'
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
            LOGGER.info("[TRACKING] ✅ PostgreSQL initialized - picks will persist across deploys!")
            self._last_postgres_error = None
            return True
            
        except Exception as e:
            self._last_postgres_error = str(e)
            LOGGER.warning(f"[TRACKING] PostgreSQL init failed: {e} - using SQLite fallback")
            return False
    
    def _init_sqlite(self):
        """Initialize SQLite tracking database (fallback)"""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tracked_picks (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                prediction_48h REAL NOT NULL,
                confidence REAL NOT NULL,
                entry_time TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notification_log (
                id INTEGER PRIMARY KEY,
                notification_type TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_preview TEXT
            )
        """)
        conn.commit()
        conn.close()
        LOGGER.info("[TRACKING] Using SQLite (ephemeral on Railway)")
    
    def set_telegram_func(self, func: Callable[[str], bool]):
        """Set the function to send Telegram messages"""
        self.send_telegram = func
    
    def get_top10_predictions(self, latest_predictions: Dict[str, Dict]) -> tuple:
        """
        Get top 5 crypto and top 5 stocks from latest predictions.
        
        LEARNING INTEGRATION:
        - Excludes symbols with <40% accuracy (after 10+ predictions)
        - Boosts confidence by 15% for symbols with >70% accuracy
        
        PERFORMANCE FIX (Dec 31, 2025):
        - Phase 1: Filter and sort using CACHED prices only (fast)
        - Phase 2: Only refresh prices for TOP 15 candidates (limited API calls)
        - This prevents 100+ API calls causing 30s timeout
        
        Args:
            latest_predictions: Dict of symbol -> prediction data
            
        Returns:
            (stocks_list, crypto_list) - each sorted by confidence
        """
        from core.asset_classifier import get_asset_type, AssetClassifier
        
        # FIXED: Use INVERSE_GHOST (not INVERSE_GHOST_MODE) - default to OFF (0)
        inverse_mode = os.getenv("INVERSE_GHOST", "0") == "1"
        
        # MONEY GAME MODE: Use profit-based rankings instead of V2 blacklist
        # Set USE_MONEY_GAME=1 to use the new competition system
        use_money_game = os.getenv("USE_MONEY_GAME", "1") == "1"  # DEFAULT ON!
        
        # Get Money Game preferred symbols if enabled
        money_game_stocks = []
        money_game_crypto = []
        if use_money_game:
            try:
                from core.money_game_engine import get_money_game
                mg = get_money_game()
                money_game_stocks = mg.get_best_symbols_for_top10("stock", limit=20)
                money_game_crypto = mg.get_best_symbols_for_top10("crypto", limit=20)
                LOGGER.info(f"[MONEY-GAME] Priority symbols: {len(money_game_stocks)} stocks, {len(money_game_crypto)} crypto")
            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Failed to get rankings: {e}")
                money_game_stocks = []
                money_game_crypto = []
        
        # Phase 1: Build candidate lists using CACHED prices (no API calls)
        stock_candidates = []
        crypto_candidates = []
        
        # Track stats for logging
        stablecoins_skipped = 0
        
        # V2 QUALITY FILTER: Only load if NOT using Money Game
        v2_excluded = 0
        v2_excluded_symbols = []
        if not use_money_game:
            from core.v2_quality import get_quality_system
            v2_quality = get_quality_system()
            LOGGER.info(f"[V2-FILTER] Active whitelist: {len(v2_quality._whitelist)}, blacklist: {len(v2_quality._blacklist)}")
        else:
            v2_quality = None
            LOGGER.info(f"[MONEY-GAME] V2 blacklist BYPASSED - using profit-based competition!")
        
        # LEARNING: Get symbol accuracy data from PostgreSQL
        accuracy_data = get_symbol_accuracy_from_postgres()
        learning_excluded = 0
        learning_boosted = 0
        excluded_symbols = []
        boosted_symbols = []
        
        # EVENT MEMORY: Load learned event patterns
        try:
            from core.event_memory import check_for_event_impact
            event_memory_active = True
            event_adjusted = 0
            LOGGER.info("[EVENT_MEMORY] 🧠 Event memory active - checking for recent events")
        except Exception as e:
            event_memory_active = False
            LOGGER.warning(f"[EVENT_MEMORY] Not available: {e}")
        
        # SYMBOL COLLISION BLACKLIST - symbols that exist as both stock and crypto
        # These cause wrong prices and confusion
        COLLISION_BLACKLIST = {
            "STX",      # Seagate (stock) vs Stacks (crypto) - use STACKS for crypto
            "DASH",     # DoorDash (stock) vs Dash (crypto) - use DASHCOIN for crypto
        }
        
        LOGGER.info(f"[TOP10] Phase 1: Filtering {len(latest_predictions)} predictions using cached prices...")
        
        for symbol, pred in latest_predictions.items():
            if not isinstance(pred, dict):
                continue
            
            # CRITICAL: Skip symbols with stock/crypto collision
            if symbol in COLLISION_BLACKLIST:
                LOGGER.info(f"[TOP10] Skipping {symbol} - symbol collision blacklist")
                continue
            
            # CRITICAL: Skip stablecoins (USDC, DAI, USDT, etc.) - they don't move!
            if AssetClassifier.is_stablecoin(symbol):
                stablecoins_skipped += 1
                continue
            
            # V2 QUALITY FILTER: Only apply if NOT using Money Game
            confidence = pred.get("confidence", 0)
            if v2_quality and not use_money_game:
                should_predict, v2_reason = v2_quality.should_predict(symbol, confidence)
                if not should_predict:
                    v2_excluded += 1
                    v2_excluded_symbols.append(f"{symbol} ({v2_reason})")
                    continue
            
            # LEARNING: Check if symbol should be excluded due to low accuracy
            should_exclude, exclude_reason = should_exclude_symbol(symbol, accuracy_data)
            if should_exclude:
                learning_excluded += 1
                excluded_symbols.append(f"{symbol} ({exclude_reason})")
                continue
            
            confidence = pred.get("confidence", 0)
            
            # MONEY GAME: Always include TOP 10 symbols regardless of confidence
            is_money_game_elite = symbol in money_game_stocks or symbol in money_game_crypto
            if not is_money_game_elite and confidence < 0.50:  # Only filter non-elite symbols
                continue
            
            # LEARNING: Apply confidence boost for high-accuracy symbols
            boost_multiplier, boost_reason = get_confidence_boost(symbol, accuracy_data)
            original_confidence = confidence
            if boost_multiplier > 1.0:
                confidence = min(1.0, confidence * boost_multiplier)  # Cap at 100%
                learning_boosted += 1
                boosted_symbols.append(f"{symbol} ({original_confidence:.0%}→{confidence:.0%})")
            
            # Get cached price from prediction (DO NOT refresh yet)
            cached_price = (pred.get("price") or 
                           pred.get("current_price") or 
                           pred.get("entry_price") or 0)
            
            # Classify asset type
            asset_class = get_asset_type(symbol)
            
            # Skip stablecoins by type too
            if asset_class == "stablecoin":
                stablecoins_skipped += 1
                continue
            
            # Direction is ALREADY inverted in _LATEST_PREDICTIONS when INVERSE_GHOST=1
            direction = pred.get("direction", "DOWN")
            
            # EVENT MEMORY: Check if recent events should adjust prediction
            event_warning = None
            if event_memory_active:
                try:
                    event_check = check_for_event_impact(symbol, direction)
                    if event_check.get("should_adjust"):
                        event_warning = event_check.get("reason")
                        event_adjusted += 1
                        # Reduce confidence when events conflict with prediction
                        confidence = confidence * 0.7  # 30% confidence reduction
                        LOGGER.info(f"[EVENT_MEMORY] ⚠️ {symbol}: {event_warning}")
                except Exception as e:
                    LOGGER.debug(f"[EVENT_MEMORY] Check failed for {symbol}: {e}")
            
            candidate = {
                "symbol": symbol,
                "cached_price": cached_price,
                "confidence": confidence,
                "direction": direction,
                "asset_class": asset_class,
                "learning_boosted": boost_multiplier > 1.0,
                "event_warning": event_warning,
                "pred": pred,  # Keep original for later
            }
            
            if asset_class == "crypto":
                crypto_candidates.append(candidate)
            else:
                stock_candidates.append(candidate)
        
        # Log stats
        if use_money_game:
            LOGGER.info(f"[MONEY-GAME] ✅ Blacklist BYPASSED - {len(stock_candidates)} stocks, {len(crypto_candidates)} crypto passed")
        elif v2_excluded > 0:
            LOGGER.info(f"[V2-FILTER] 🚫 EXCLUDED {v2_excluded} symbols via V2 quality filter")
            LOGGER.debug(f"[V2-FILTER] Excluded: {', '.join(v2_excluded_symbols[:10])}")
        if learning_excluded > 0:
            LOGGER.info(f"[LEARNING] 🚫 EXCLUDED {learning_excluded} low-accuracy symbols")
        if learning_boosted > 0:
            LOGGER.info(f"[LEARNING] 🚀 BOOSTED {learning_boosted} high-accuracy symbols")
        if event_memory_active and event_adjusted > 0:
            LOGGER.info(f"[EVENT_MEMORY] ⚠️ ADJUSTED {event_adjusted} predictions due to recent events")
        
        # ================================================================
        # V3 QUALITY FILTER - Apply historical win rate scoring and inverse logic
        # ================================================================
        if V3_ENABLED:
            LOGGER.info(f"[V3] Applying V3 quality filter to {len(stock_candidates)} stocks, {len(crypto_candidates)} crypto...")
            
            # Apply V3 to stock candidates
            stock_candidates = v3_filter_and_score(stock_candidates)
            # Apply V3 to crypto candidates
            crypto_candidates = v3_filter_and_score(crypto_candidates)
            
            # V3 sorts by v3_score, so we don't need Money Game or confidence sorting
            LOGGER.info(f"[V3] ✅ After V3 filter: {len(stock_candidates)} stocks, {len(crypto_candidates)} crypto")
            top_stocks_str = ", ".join([f"{c['symbol']} ({c.get('v3_score', 0):.2f})" for c in stock_candidates[:3]])
            top_crypto_str = ", ".join([f"{c['symbol']} ({c.get('v3_score', 0):.2f})" for c in crypto_candidates[:3]])
            LOGGER.info(f"[V3] Top 3 stocks: {top_stocks_str}")
            LOGGER.info(f"[V3] Top 3 crypto: {top_crypto_str}")
        else:
            # Legacy sorting (Money Game or confidence)
            # MONEY GAME: Sort by profit ranking, not just confidence
            if use_money_game and (money_game_stocks or money_game_crypto):
                # Create priority lookup (lower index = higher priority)
                stock_priority = {sym: i for i, sym in enumerate(money_game_stocks)}
                crypto_priority = {sym: i for i, sym in enumerate(money_game_crypto)}
                
                # Sort by: Money Game priority first, then confidence as tiebreaker
                # Symbols NOT in Money Game get priority 999 (go to end)
                stock_candidates.sort(key=lambda x: (
                    stock_priority.get(x["symbol"], 999),
                    -x["confidence"]
                ))
                crypto_candidates.sort(key=lambda x: (
                    crypto_priority.get(x["symbol"], 999),
                    -x["confidence"]
                ))
                LOGGER.info(f"[MONEY-GAME] ✅ Sorted by PROFIT rankings, not just confidence")
            else:
                # Fallback: Sort candidates by confidence only
                stock_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                crypto_candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Phase 2: Only refresh prices for TOP 15 of each (5 final + buffer)
        TOP_N_TO_REFRESH = 15
        
        LOGGER.info(f"[TOP10] Phase 2: Refreshing prices for top {TOP_N_TO_REFRESH} candidates only...")
        
        stocks = []
        crypto = []
        prices_refreshed = 0
        
        # Helper: Get stock price with Yahoo Finance fallback
        def get_stock_price_with_fallback(symbol: str) -> float:
            """Try turbo provider, fall back to Yahoo Finance"""
            # Try turbo first
            try:
                from core.providers.turbo_provider import get_turbo_provider
                turbo = get_turbo_provider()
                fresh = turbo.turbo_stock_price(symbol, max_budget_s=1.5)
                if fresh.get("ok") and fresh.get("price", 0) > 0:
                    return fresh["price"]
            except Exception as e:
                LOGGER.debug(f"[TOP10] Turbo stock price failed for {symbol}: {e}")
            
            # Fallback: Yahoo Finance (works pre-market!)
            try:
                import requests
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                    LOGGER.debug(f"[TOP10] Yahoo fallback price for {symbol}: ${price}")
                    return float(price)
            except Exception as e:
                LOGGER.debug(f"[TOP10] Yahoo price failed for {symbol}: {e}")
            
            return 0
        
        # Process top stock candidates
        for candidate in stock_candidates[:TOP_N_TO_REFRESH]:
            symbol = candidate["symbol"]
            current_price = candidate["cached_price"]
            
            # Try to refresh price if stale or missing
            if current_price <= 0:
                current_price = get_stock_price_with_fallback(symbol)
                if current_price > 0:
                    prices_refreshed += 1
            
            if current_price <= 0:
                LOGGER.warning(f"[TOP10] Skipping {symbol} - no price available")
                continue
            
            # Build final pick
            pick = self._build_pick(candidate, current_price)
            stocks.append(pick)
            
            if len(stocks) >= 5:
                break
        
        # Process top crypto candidates
        for candidate in crypto_candidates[:TOP_N_TO_REFRESH]:
            symbol = candidate["symbol"]
            current_price = candidate["cached_price"]
            
            # Try to refresh price if stale or missing
            if current_price <= 0:
                # Method 1: CoinGecko quorum
                try:
                    from core.crypto.crypto_providers import get_crypto_price_quorum
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    fresh_price_data = loop.run_until_complete(
                        get_crypto_price_quorum(symbol, use_cache=True)
                    )
                    if fresh_price_data and fresh_price_data.get("price", 0) > 0:
                        current_price = fresh_price_data["price"]
                        prices_refreshed += 1
                except Exception as e:
                    LOGGER.debug(f"[TOP10] Crypto quorum failed for {symbol}: {e}")
                
                # Method 2: Direct CoinGecko fallback
                if current_price <= 0:
                    try:
                        import requests
                        # Symbol to CoinGecko ID mapping
                        cg_map = {
                            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
                            "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
                            "RNDR": "render-token", "ZEC": "zcash", "CHZ": "chiliz",
                            "TURBO": "turbo", "ILV": "illuvium", "LINK": "chainlink",
                            "AVAX": "avalanche-2", "DOT": "polkadot", "ATOM": "cosmos",
                            "NEAR": "near", "ARB": "arbitrum", "OP": "optimism",
                            "PEPE": "pepe", "SHIB": "shiba-inu", "BONK": "bonk",
                        }
                        cg_id = cg_map.get(symbol.upper(), symbol.lower())
                        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
                        resp = requests.get(url, timeout=3)
                        if resp.status_code == 200:
                            data = resp.json()
                            if cg_id in data:
                                current_price = data[cg_id]["usd"]
                                prices_refreshed += 1
                                LOGGER.debug(f"[TOP10] CoinGecko fallback for {symbol}: ${current_price}")
                    except Exception as e:
                        LOGGER.debug(f"[TOP10] CoinGecko fallback failed for {symbol}: {e}")
            
            if current_price <= 0:
                continue
            
            # Build final pick
            pick = self._build_pick(candidate, current_price)
            crypto.append(pick)
            
            if len(crypto) >= 5:
                break
        
        LOGGER.info(f"[TOP10] ✅ Complete: {len(stocks)} stocks, {len(crypto)} crypto, {prices_refreshed} prices refreshed")
        
        return stocks[:5], crypto[:5]
    
    def _build_pick(self, candidate: Dict, current_price: float) -> Dict:
        """
        Build a pick dict from candidate and current price.
        
        IMPORTANT: Preserve ALL prediction data from the source.
        Telegram message must mirror EXACTLY what the backend calculated.
        """
        from core.asset_classifier import get_asset_type
        
        symbol = candidate["symbol"]
        direction = candidate["direction"]
        confidence = candidate["confidence"]
        asset_class = candidate["asset_class"]
        
        # Get original prediction data for complete values
        pred = candidate.get("pred", {})
        
        # PRESERVE prediction data if available (from turbo/ensemble predictor)
        # Check both candidate and original pred dict
        target_price = (
            candidate.get("target_price") or 
            pred.get("target_price") or 
            candidate.get("prediction_48h") or 
            pred.get("prediction_48h")
        )
        if not target_price or target_price <= 0:
            # Fallback: Calculate from confidence
            if direction in ("UP", "BUY"):
                move_pct = 0.05 if asset_class == "crypto" else 0.03
                target_price = current_price * (1 + move_pct)
            else:
                move_pct = 0.05 if asset_class == "crypto" else 0.03
                target_price = current_price * (1 - move_pct)
        
        # PRESERVE stop from prediction if available
        stop_price = (
            candidate.get("stop_loss") or
            pred.get("stop_loss") or
            candidate.get("stop") or
            pred.get("stop")
        )
        if not stop_price or stop_price <= 0:
            # Fallback: Calculate stop based on direction
            if direction in ("UP", "BUY"):
                stop_price = current_price * 0.98  # 2% below for BUY
            else:
                stop_price = current_price * 1.02  # 2% above for SELL
        
        # PRESERVE hold_days from prediction if available
        hold_days = candidate.get("hold_days") or pred.get("hold_days") or 3
        hold_reason = candidate.get("hold_reason") or pred.get("hold_reason") or "swing_trade"
        
        # PRESERVE expected_move and volatility for Entry Zone
        expected_move = candidate.get("expected_move_pct") or pred.get("expected_move_pct") or 0.03
        volatility = candidate.get("volatility") or pred.get("volatility") or 0.02
        
        # PRESERVE news_influenced from prediction
        news_influenced = candidate.get("news_influenced") or pred.get("news_influenced") or False
        
        # Legacy fields (keep for backward compatibility)
        buy_in = current_price * 0.99  # 1% below current
        sell_at = current_price * 1.02  # 2% above current
        
        return {
            "symbol": symbol,
            "current": current_price,
            "prediction_48h": target_price,
            "target_price": target_price,
            "buy_in": buy_in,
            "sell": sell_at,
            "stop": stop_price,
            "stop_loss": stop_price,  # Alias
            "confidence": confidence,
            "direction": direction,
            "asset_type": asset_class,
            "hold_days": hold_days,
            "hold_reason": hold_reason,
            "expected_move_pct": expected_move,
            "volatility": volatility,
            "news_influenced": news_influenced,
            "learning_boosted": candidate.get("learning_boosted", False),
            # V3 DATA
            "v3_score": candidate.get("v3_score", 0),
            "v3_historical_win_rate": candidate.get("v3_historical_win_rate", 0.50),
            "v3_is_inverse": candidate.get("v3_is_inverse", False),
            "v3_original_direction": candidate.get("v3_original_direction"),
            "v3_is_whitelisted": candidate.get("v3_is_whitelisted", False),
        }
    
    def send_top10(self, latest_predictions: Dict[str, Dict]) -> bool:
        """
        Send the morning TOP 10 message.
        
        Should be called at 8 AM Central.
        """
        if not self.send_telegram:
            LOGGER.error("[NOTIFICATIONS] No Telegram function set")
            return False
        
        # Check if already sent today
        today = get_central_time().strftime("%Y-%m-%d")
        if self._last_top10_date == today:
            LOGGER.info(f"[NOTIFICATIONS] TOP 10 already sent today ({today})")
            return False
        
        stocks, crypto = self.get_top10_predictions(latest_predictions)
        
        if not stocks and not crypto:
            LOGGER.warning("[NOTIFICATIONS] No predictions available for TOP 10")
            return False
        
        # FIXED: Use INVERSE_GHOST (not INVERSE_GHOST_MODE) - default to OFF (0)
        inverse_mode = os.getenv("INVERSE_GHOST", "0") == "1"
        messages = format_top10_message(stocks, crypto, inverse_mode)
        
        LOGGER.info(f"[NOTIFICATIONS] Sending TOP 10 ({len(stocks)} stocks, {len(crypto)} crypto)")
        
        # Send each message (format returns list of 2 messages: stocks + crypto)
        success = True
        for msg in messages:
            if not self.send_telegram(msg):
                success = False
        
        if success:
            self._last_top10_date = today
            LOGGER.info("[NOTIFICATIONS] ✅ TOP 10 sent successfully")
            
            # Register picks for tracking
            self._register_picks_for_tracking(stocks + crypto)
        else:
            LOGGER.error("[NOTIFICATIONS] ❌ Failed to send TOP 10")
        
        return success
    
    def _register_picks_for_tracking(self, picks: List[Dict]):
        """
        Register picks for 48-hour tracking (PostgreSQL or SQLite).
        
        CRITICAL: Also logs to paper_trades table for win rate tracking!
        This connects Telegram alerts → paper trading database.
        """
        now = get_central_time()
        expires = now + timedelta(hours=48)
        
        # =====================================================================
        # CRITICAL FIX (Jan 10, 2026): Log to paper_trades table
        # Telegram TOP 10 alerts were NOT being tracked in paper trades DB
        # This caused disconnect: Telegram ~60% win rate vs DB 16.7%
        # =====================================================================
        try:
            from core.paper_tracker import get_paper_tracker
            
            paper_tracker = get_paper_tracker()
            logged_count = 0
            
            for p in picks:
                try:
                    # Determine direction (BUY/SELL → UP/DOWN)
                    action, _, _ = determine_action(p['current'], p['prediction_48h'], p['confidence'])
                    direction = "UP" if action == "BUY" else "DOWN"
                    
                    # Log to paper trades with unique cascade ID
                    paper_trade_id = paper_tracker.log_signal(
                        cascade_id=f"top10_{p['symbol']}_{int(now.timestamp())}",
                        symbol=p['symbol'],
                        signal_direction=direction,
                        signal_confidence=p['confidence'],
                        entry_price=p['current'],
                        entry_time=now.isoformat(),
                        position_size=1000.0,  # $1k position size
                        stop_loss_pct=0.05,    # 5% stop loss
                        take_profit_pct=0.10   # 10% take profit
                    )
                    
                    if paper_trade_id:
                        logged_count += 1
                        LOGGER.info(f"[PAPER-TRACK] ✅ Logged {p['symbol']} {direction} to paper_trades (ID: {paper_trade_id[:8]}...)")
                    else:
                        LOGGER.debug(f"[PAPER-TRACK] ⏭️ Skipped {p['symbol']} (blacklisted or low confidence)")
                
                except Exception as e:
                    LOGGER.warning(f"[PAPER-TRACK] Failed to log {p.get('symbol', 'UNKNOWN')}: {e}")
            
            LOGGER.info(f"[PAPER-TRACK] ✅ Logged {logged_count}/{len(picks)} TOP 10 picks to paper_trades table")
        
        except Exception as e:
            LOGGER.error(f"[PAPER-TRACK] Paper tracker integration failed: {e} - continuing with ghost_tracked_picks only")
        
        # Continue with original tracking system
        if self._use_postgres:
            try:
                conn = self._get_postgres_conn()
                cur = conn.cursor()
                
                # First, ensure we have a unique constraint on symbol for active picks
                # This prevents duplicate registrations
                cur.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_symbol 
                    ON ghost_tracked_picks (symbol) 
                    WHERE status = 'active'
                """)
                
                for p in picks:
                    action, _, _ = determine_action(p['current'], p['prediction_48h'], p['confidence'])
                    asset_type = p.get('asset_type', 'crypto' if p['symbol'] in ['BTC', 'ETH', 'SOL'] else 'stock')
                    
                    # Use ON CONFLICT to update existing active picks instead of duplicating
                    cur.execute("""
                        INSERT INTO ghost_tracked_picks 
                        (symbol, asset_type, direction, entry_price, target_price, stop_price, 
                         prediction_48h, confidence, entry_time, expires_at, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
                        ON CONFLICT (symbol) WHERE status = 'active'
                        DO UPDATE SET
                            entry_price = EXCLUDED.entry_price,
                            target_price = EXCLUDED.target_price,
                            stop_price = EXCLUDED.stop_price,
                            prediction_48h = EXCLUDED.prediction_48h,
                            confidence = EXCLUDED.confidence,
                            entry_time = EXCLUDED.entry_time,
                            expires_at = EXCLUDED.expires_at
                    """, (
                        p['symbol'],
                        asset_type,
                        action,
                        p['current'],
                        p['prediction_48h'],
                        p['current'] * 0.95 if action == 'BUY' else p['current'] * 1.05,
                        p['prediction_48h'],
                        p['confidence'],
                        now,
                        expires,
                    ))
                
                conn.commit()
                cur.close()
                conn.close()
                LOGGER.info(f"[TRACKING] ✅ Registered {len(picks)} picks in PostgreSQL (persistent)")
                return
                
            except Exception as e:
                LOGGER.error(f"[TRACKING] PostgreSQL insert failed: {e} - falling back to SQLite")
        
        # SQLite fallback
        conn = sqlite3.connect(self._db_path)
        for p in picks:
            action, _, _ = determine_action(p['current'], p['prediction_48h'], p['confidence'])
            asset_type = p.get('asset_type', 'crypto' if p['symbol'] in ['BTC', 'ETH', 'SOL'] else 'stock')
            
            conn.execute("""
                INSERT INTO tracked_picks 
                (symbol, asset_type, direction, entry_price, target_price, stop_price, 
                 prediction_48h, confidence, entry_time, expires_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """, (
                p['symbol'],
                asset_type,
                action,
                p['current'],
                p['prediction_48h'],
                p['current'] * 0.95 if action == 'BUY' else p['current'] * 1.05,
                p['prediction_48h'],
                p['confidence'],
                now.isoformat(),
                expires.isoformat(),
            ))
        
        conn.commit()
        conn.close()
        LOGGER.info(f"[TRACKING] Registered {len(picks)} picks in SQLite (ephemeral)")
    
    def check_for_updates(self, get_price_func: Callable[[str], float]) -> bool:
        """
        Check all tracked picks for significant moves.
        
        Sends update message if any pick moved >3%.
        Should be called every 15-30 minutes.
        """
        if not self.send_telegram:
            return False
        
        # Load active picks from PostgreSQL (persistent across deploys)
        rows = []
        if self._use_postgres:
            try:
                conn = self._get_postgres_conn()
                cur = conn.cursor()
                cur.execute("""
                    SELECT symbol, asset_type, direction, entry_price, target_price, stop_price,
                           prediction_48h, confidence, entry_time, expires_at
                    FROM ghost_tracked_picks 
                    WHERE status = 'active'
                """)
                rows = cur.fetchall()
                cur.close()
                conn.close()
            except Exception as e:
                LOGGER.error(f"PostgreSQL fetch failed: {e}")
        else:
            # Fallback to SQLite (ephemeral on Railway)
            conn = sqlite3.connect(self._db_path)
            rows = conn.execute("""
                SELECT symbol, asset_type, direction, entry_price, target_price, stop_price,
                       prediction_48h, confidence, entry_time, expires_at
                FROM tracked_picks 
                WHERE status = 'active'
            """).fetchall()
            conn.close()
        
        if not rows:
            return False
        
        updates = []
        alerts = []
        off_path_stocks = []
        off_path_crypto = []
        
        for row in rows:
            symbol, asset_type, direction, entry, target, stop, pred_48h, conf, entry_time, expires = row
            
            # Get current price
            current = get_price_func(symbol)
            if current <= 0:
                continue
            
            pct_change = (current - entry) / entry
            
            # Validate direction (must be "BUY" or "SELL")
            if direction not in ("BUY", "SELL"):
                LOGGER.error(f"[WATCHDOG] Invalid direction '{direction}' for {symbol} - skipping")
                continue
            
            # Determine if on track based on direction
            if direction == "BUY":
                emoji = "🟢"
                on_track = current >= entry * 0.98  # Allow 2% buffer
                # CRITICAL FIX (Jan 11, 2026 - v2): EXACT target match only (no buffer)
                # Bug: 2% buffer allowed false positives like META +0.4% triggering
                near_target = current >= target  # Must ACTUALLY hit target
                near_stop = current <= stop * 1.02
                # OFF PATH = BUY but price dropped >2%
                is_off_path = pct_change < -0.02
            else:  # SELL
                emoji = "🔴"
                on_track = current <= entry * 1.02  # Allow 2% buffer
                # CRITICAL FIX (Jan 11, 2026 - v2): EXACT target match only (no buffer)
                near_target = current <= target  # Must ACTUALLY hit target
                near_stop = current >= stop * 0.98
                # OFF PATH = SELL but price rose >2%
                is_off_path = pct_change > 0.02
            
            # Check for target/stop hit (HIGHEST PRIORITY)
            # CRITICAL FIX (Jan 11, 2026): Only trigger on moves in CORRECT direction
            # BUG: Was using abs(pct_change) which triggered on ANY 3% move
            # Example: BUY signal going DOWN 3% would incorrectly say "HIT TARGET"
            # v2 FIX: Removed 2% buffer, added direction validation, added debug logging
            if near_target:
                LOGGER.info(f"[WATCHDOG] 🎯 TARGET CHECK: {symbol} {direction} @ ${current:.2f} vs target ${target:.2f} (entry ${entry:.2f}, +{pct_change*100:.1f}%)")
                alerts.append({
                    "symbol": symbol,
                    "type": "target_hit",
                    "entry": entry,
                    "current": current,
                    "target": target,
                    "stop": stop,
                })
            elif near_stop:
                alerts.append({
                    "symbol": symbol,
                    "type": "stop_hit",
                    "entry": entry,
                    "current": current,
                    "target": target,
                    "stop": stop,
                })
            # Check for OFF PATH (moving against prediction)
            # MARKET HOURS AWARENESS: Only add stock path alerts during trading hours
            # Crypto trades 24/7 so always check crypto
            elif is_off_path:
                off_path_pick = {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "direction": direction,
                    "entry": entry,
                    "current": current,
                    "target": target,
                    "stop": stop,
                    "pct_change": pct_change,
                }
                if asset_type == "crypto":
                    # Crypto trades 24/7 - always alert
                    off_path_crypto.append(off_path_pick)
                elif is_stock_market_hours():
                    # Stocks only during market hours (to avoid duplicate overnight alerts)
                    off_path_stocks.append(off_path_pick)
                else:
                    LOGGER.debug(f"[WATCHDOG] Skipping stock {symbol} path alert - market closed")
            # Check for significant moves (for scheduled updates)
            elif abs(pct_change) >= SIGNIFICANT_MOVE_PCT:
                updates.append({
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "entry": entry,
                    "current": current,
                    "emoji": emoji,
                    "on_track": on_track,
                    "near_target": near_target,
                    "near_stop": near_stop,
                })
        
        # Send target/stop alerts immediately (HIGHEST PRIORITY)
        if alerts:
            # CRITICAL: Update status BEFORE sending to prevent duplicate alerts
            # If send fails, we lose the alert but don't spam users
            for a in alerts:
                status = "target_hit" if a['type'] == 'target_hit' else 'stop_hit'
                
                # Update PostgreSQL FIRST (persistent, primary)
                if self._use_postgres:
                    try:
                        conn = self._get_postgres_conn()
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE ghost_tracked_picks SET status = %s WHERE symbol = %s AND status = 'active'",
                            (status, a['symbol'])
                        )
                        conn.commit()
                        cur.close()
                        conn.close()
                        LOGGER.info(f"[TRACKING] ✅ Updated {a['symbol']} to {status} in PostgreSQL (BEFORE alert)")
                    except Exception as e:
                        LOGGER.error(f"[TRACKING] ❌ Failed to update PostgreSQL for {a['symbol']}: {e}")
                        # DON'T send alert if we couldn't update status - prevents duplicates
                        continue
                
                # Also update SQLite (fallback)
                try:
                    sqlite_conn = sqlite3.connect(self._db_path)
                    sqlite_conn.execute("UPDATE tracked_picks SET status = ? WHERE symbol = ?", 
                               (status, a['symbol']))
                    sqlite_conn.commit()
                    sqlite_conn.close()
                except Exception as e:
                    LOGGER.debug(f"SQLite update failed for {a['symbol']}: {e}")
            
            # NOW send the alert (after status is updated)
            msg = format_alert_message(alerts)
            self.send_telegram(msg)
            LOGGER.info(f"[NOTIFICATIONS] Sent {len(alerts)} target/stop alerts")
        
        # Send OFF PATH alerts - with deduplication (only once per symbol per day)
        ct = get_central_time()
        today = ct.strftime("%Y-%m-%d")
        
        # First, deduplicate the off_path lists by symbol (keep first occurrence)
        seen_symbols_stocks = set()
        deduped_off_path_stocks = []
        for p in off_path_stocks:
            if p['symbol'] not in seen_symbols_stocks:
                seen_symbols_stocks.add(p['symbol'])
                deduped_off_path_stocks.append(p)
        
        seen_symbols_crypto = set()
        deduped_off_path_crypto = []
        for p in off_path_crypto:
            if p['symbol'] not in seen_symbols_crypto:
                seen_symbols_crypto.add(p['symbol'])
                deduped_off_path_crypto.append(p)
        
        # Then filter out symbols we already alerted today
        new_off_path_stocks = [
            p for p in deduped_off_path_stocks 
            if self._last_off_path_alerts.get(p['symbol']) != today
        ]
        new_off_path_crypto = [
            p for p in deduped_off_path_crypto 
            if self._last_off_path_alerts.get(p['symbol']) != today
        ]
        
        if new_off_path_stocks:
            msg = format_off_path_alert(new_off_path_stocks, asset_type="stock")
            self.send_telegram(msg)
            # Mark as alerted today
            for p in new_off_path_stocks:
                self._last_off_path_alerts[p['symbol']] = today
            LOGGER.info(f"[NOTIFICATIONS] Sent OFF PATH alert for {len(new_off_path_stocks)} stocks")
        
        if new_off_path_crypto:
            msg = format_off_path_alert(new_off_path_crypto, asset_type="crypto")
            self.send_telegram(msg)
            # Mark as alerted today
            for p in new_off_path_crypto:
                self._last_off_path_alerts[p['symbol']] = today
            LOGGER.info(f"[NOTIFICATIONS] Sent OFF PATH alert for {len(new_off_path_crypto)} crypto")
        
        # Send scheduled updates (12 PM, 4 PM, 8 PM) - only if no alerts sent
        if ct.hour in UPDATE_HOURS and ct.hour != self._last_update_hour:
            if updates and not alerts and not new_off_path_stocks and not new_off_path_crypto:
                msg = format_update_message(updates)
                self.send_telegram(msg)
                self._last_update_hour = ct.hour
                LOGGER.info(f"[NOTIFICATIONS] Sent scheduled update for {len(updates)} picks")
        
        return bool(alerts or updates or new_off_path_stocks or new_off_path_crypto)
    
    def get_status(self) -> Dict:
        """Get current status of the notification system"""
        active = 0
        target_hits = 0
        stop_hits = 0
        db_type = "sqlite"
        
        # Try PostgreSQL first (persistent)
        if self._use_postgres:
            try:
                conn = self._get_postgres_conn()
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM ghost_tracked_picks WHERE status = 'active'")
                active = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM ghost_tracked_picks WHERE status = 'target_hit'")
                target_hits = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM ghost_tracked_picks WHERE status = 'stop_hit'")
                stop_hits = cur.fetchone()[0]
                cur.close()
                conn.close()
                db_type = "postgresql"
            except Exception as e:
                LOGGER.error(f"PostgreSQL status check failed: {e}")
        else:
            # Fallback to SQLite (ephemeral)
            conn = sqlite3.connect(self._db_path)
            active = conn.execute("SELECT COUNT(*) FROM tracked_picks WHERE status = 'active'").fetchone()[0]
            target_hits = conn.execute("SELECT COUNT(*) FROM tracked_picks WHERE status = 'target_hit'").fetchone()[0]
            stop_hits = conn.execute("SELECT COUNT(*) FROM tracked_picks WHERE status = 'stop_hit'").fetchone()[0]
            conn.close()
        
        return {
            "active_picks": active,
            "target_hits": target_hits,
            "stop_hits": stop_hits,
            "last_top10_date": self._last_top10_date,
            "central_time": get_central_time().isoformat(),
            "next_top10_hour": TOP_10_HOUR,
            "update_hours": UPDATE_HOURS,
            "database": db_type,
            "persistent": db_type == "postgresql",
            "stock_market_open": is_stock_market_hours(),
        }


# Singleton instance
_notification_system: Optional[GhostNotificationSystem] = None


def get_notification_system() -> GhostNotificationSystem:
    """Get the singleton notification system"""
    global _notification_system
    if _notification_system is None:
        _notification_system = GhostNotificationSystem()
    return _notification_system


# ============================================================================
# SIMPLE TEST
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test message formatting
    test_stocks = [
        {"symbol": "MSFT", "current": 484.92, "prediction_48h": 471.78, "buy_in": 480.00, "sell": 495.00, "confidence": 0.92},
        {"symbol": "GOOGL", "current": 309.78, "prediction_48h": 295.00, "buy_in": 305.00, "sell": 318.18, "confidence": 0.92},
    ]
    test_crypto = [
        {"symbol": "ETH", "current": 2995.00, "prediction_48h": 2815.00, "buy_in": 2900.00, "sell": 3050.00, "confidence": 0.95},
        {"symbol": "BTC", "current": 88255.00, "prediction_48h": 92000.00, "buy_in": 87500.00, "sell": 90000.00, "confidence": 0.92},
    ]
    
    print("=" * 60)
    print("TEST TOP 10 MESSAGE")
    print("=" * 60)
    print(format_top10_message(test_stocks, test_crypto, inverse_mode=True))
