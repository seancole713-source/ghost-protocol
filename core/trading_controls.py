"""
Ghost Protocol Trading Controls
================================
Blacklist/Whitelist management based on historical performance.

This module prevents trading assets with consistently poor performance (0-20% win rate)
and prioritizes assets with proven historical success (>50% win rate).

Historical Performance Analysis (from database reality check Jan 9, 2026):
- System claimed 80% win rate → Database showed 16.7% reality
- Major cryptos (SOL, ETH, BTC, XRP) had 0-3% accuracy
- Some assets (CHZ, ZEC, T) had 100% accuracy
- Root cause: Model bias + trading assets it can't predict

Solution: Stop trading losers, focus on proven winners.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# BLACKLIST: AUTO-POPULATED from live paper trade accuracy
# ============================================================================
# Static blacklist cleared Feb 7 2026 (old data from broken model).
# Now uses dynamic auto-blacklist: if a symbol has 20+ resolved paper trades
# and win rate < 35%, it gets blocked automatically.
# The auto_blacklist_check() function queries the database at runtime.
STATIC_BLACKLIST = {
    # Add symbols here only for non-accuracy reasons (e.g., delisted, illiquid)
}

# Dynamic blacklist cache (refreshed every 30 minutes)
_AUTO_BLACKLIST_CACHE = {}  # {symbol: win_rate}
_AUTO_BLACKLIST_LAST_REFRESH = 0
AUTO_BLACKLIST_MIN_TRADES = 20       # Need 20+ resolved trades
AUTO_BLACKLIST_MAX_WIN_RATE = 0.35   # Block if win rate < 35%
AUTO_BLACKLIST_REFRESH_SECONDS = 1800  # Refresh every 30 min


def _refresh_auto_blacklist():
    """Query paper_trades for symbols with proven poor accuracy."""
    global _AUTO_BLACKLIST_CACHE, _AUTO_BLACKLIST_LAST_REFRESH
    import time as _time
    now = _time.time()
    if now - _AUTO_BLACKLIST_LAST_REFRESH < AUTO_BLACKLIST_REFRESH_SECONDS:
        return  # Cache still fresh
    
    try:
        from core.paper_tracker import get_paper_tracker
        tracker = get_paper_tracker()
        conn = tracker._get_connection()
        
        # Use compatible SQL (works on both PostgreSQL and SQLite)
        # IMPORTANT: Only consider trades from the last 14 days to avoid
        # penalizing symbols based on the old broken model's data.
        # The retrained model (Feb 7 2026) needs a clean slate.
        if tracker.use_postgres:
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol,
                       SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) as resolved,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
                FROM paper_trades
                WHERE outcome IN ('WIN', 'LOSS')
                  AND entry_time::timestamptz > NOW() - INTERVAL '14 days'
                GROUP BY symbol
                HAVING SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) >= %s
            """, (AUTO_BLACKLIST_MIN_TRADES,))
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        else:
            cur = conn.execute("""
                SELECT symbol,
                       SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) as resolved,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
                FROM paper_trades
                WHERE outcome IN ('WIN', 'LOSS')
                  AND entry_time > datetime('now', '-14 days')
                GROUP BY symbol
                HAVING SUM(CASE WHEN outcome IN ('WIN','LOSS') THEN 1 ELSE 0 END) >= ?
            """, (AUTO_BLACKLIST_MIN_TRADES,))
            rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        
        new_blacklist = {}
        for row in rows:
            sym = row['symbol']
            resolved = row['resolved']
            wins = row['wins']
            win_rate = wins / resolved if resolved > 0 else 0
            if win_rate < AUTO_BLACKLIST_MAX_WIN_RATE:
                new_blacklist[sym] = win_rate
        
        _AUTO_BLACKLIST_CACHE = new_blacklist
        _AUTO_BLACKLIST_LAST_REFRESH = now
        if new_blacklist:
            logger.info(f"[AUTO-BLACKLIST] Refreshed: {len(new_blacklist)} symbols blocked (< {AUTO_BLACKLIST_MAX_WIN_RATE:.0%} WR with {AUTO_BLACKLIST_MIN_TRADES}+ trades)")
    except Exception as e:
        logger.warning(f"[AUTO-BLACKLIST] Refresh failed (using cache): {e}")


# Backward compatibility alias
BLACKLIST = STATIC_BLACKLIST

# ============================================================================
# WHITELIST: Symbols with historically strong performance — PRIORITIZE
# ============================================================================
# Kept as reference but will be rebuilt with fresh data from retrained model.
# V3 validated strategies (ETH, XRP, LINK, CHZ, PANW, NET, etc.) bypass 
# this entirely via V3_VALIDATED_STRATEGIES in should_trade().
WHITELIST = {
    # Will be populated by accuracy tracker as new data arrives
}

# ============================================================================
# TRADING PARAMETERS
# ============================================================================
MIN_CONFIDENCE = 0.30  # Paper trade threshold — matches wolf_app.py paper trade gate
WHITELIST_ONLY_MODE = False  # True = only trade whitelist, False = also allow unknown assets


def should_trade(symbol: str, confidence: float) -> Tuple[bool, str]:
    """
    Check if we should trade this symbol based on historical performance.
    
    Args:
        symbol: Trading symbol (e.g., "BTC", "CHZ")
        confidence: Model confidence (0.0-1.0)
    
    Returns:
        (can_trade: bool, reason: str)
        
    Examples:
        >>> should_trade("SOL", 0.85)
        (False, "Blacklisted: 0.0% historical win rate - Model cannot predict SOL")
        
        >>> should_trade("CHZ", 0.75)
        (True, "Whitelisted: 100.0% historical win rate")
        
        >>> should_trade("AAPL", 0.65)
        (False, "Low confidence: 65.0% < 70.0% threshold")
    """
    symbol = symbol.upper()
    
    # V3 BYPASS (HIGHEST PRIORITY): If symbol has a validated strategy with p < 0.05,
    # skip ALL checks - the backtest proves the edge exists
    # This overrides even the blacklist, because V3 strategies like ghost_inverse
    # INTENTIONALLY trade symbols the base model gets wrong (that's why they're blacklisted)
    try:
        from core.ghost_notifications import V3_VALIDATED_STRATEGIES
        if symbol in V3_VALIDATED_STRATEGIES:
            v3_config = V3_VALIDATED_STRATEGIES[symbol]
            return True, f"V3 validated: {v3_config.get('strategy')} @ {v3_config.get('win_rate', 0):.1%} win rate (p={v3_config.get('p_value')})"
    except ImportError:
        pass
    
    # Check 1: Static blacklist (for non-V3 symbols only)
    if symbol in STATIC_BLACKLIST:
        return False, f"Static blacklisted: {symbol} (delisted/illiquid)"
    
    # Check 1b: Dynamic auto-blacklist from live paper trade accuracy
    _refresh_auto_blacklist()
    if symbol in _AUTO_BLACKLIST_CACHE:
        wr = _AUTO_BLACKLIST_CACHE[symbol]
        return False, f"Auto-blacklisted: {wr:.0%} win rate over {AUTO_BLACKLIST_MIN_TRADES}+ trades"
    
    # Check 2: Confidence threshold (for non-V3 symbols)
    if confidence < MIN_CONFIDENCE:
        return False, f"Low confidence: {confidence:.1%} < {MIN_CONFIDENCE:.1%} threshold"
    
    # Check 3: Whitelist-only mode
    if WHITELIST_ONLY_MODE and symbol not in WHITELIST:
        return False, f"Not in whitelist - Only trading proven winners (set WHITELIST_ONLY_MODE=False to disable)"
    
    # Check 4: If whitelisted, always approve (already passed confidence check)
    if symbol in WHITELIST:
        win_rate = WHITELIST[symbol]
        return True, f"Whitelisted: {win_rate:.1%} historical win rate"
    
    # Check 5: Unknown asset with sufficient confidence
    return True, f"Unknown asset with {confidence:.1%} confidence - Proceeding cautiously"


def get_position_multiplier(symbol: str, confidence: float) -> float:
    """
    Get position size multiplier based on confidence and historical performance.
    
    Args:
        symbol: Trading symbol
        confidence: Model confidence (0.0-1.0)
    
    Returns:
        float: Position size multiplier (0.5 - 2.0)
        
    Strategy:
    - High confidence (≥80%) + High historical win rate (≥90%): 2.0x position
    - High confidence (≥80%) + Medium win rate (70-89%): 1.5x position
    - Medium confidence (70-79%) + High win rate (≥90%): 1.5x position
    - Medium confidence (70-79%) + Medium win rate: 1.0x position
    - Lower confidence (65-69%): 0.7x position
    - Below 65%: 0.5x position (or don't trade)
    
    Examples:
        >>> get_position_multiplier("CHZ", 0.85)  # Perfect history + high confidence
        2.0
        
        >>> get_position_multiplier("AAVE", 0.75)  # 64% history + medium confidence
        1.0
        
        >>> get_position_multiplier("UNKNOWN", 0.72)  # Unknown asset
        0.8
    """
    symbol = symbol.upper()
    
    # Get historical win rate (default to 0.5 for unknown)
    hist_wr = WHITELIST.get(symbol, 0.5)
    
    # Confidence multiplier
    if confidence >= 0.80:
        conf_mult = 1.5
    elif confidence >= 0.70:
        conf_mult = 1.0
    elif confidence >= 0.65:
        conf_mult = 0.7
    else:
        conf_mult = 0.5
    
    # Historical performance multiplier
    if hist_wr >= 0.90:
        hist_mult = 1.5
    elif hist_wr >= 0.70:
        hist_mult = 1.2
    elif hist_wr >= 0.50:
        hist_mult = 1.0
    else:
        hist_mult = 0.8  # Unknown or mediocre
    
    # Combined multiplier (capped at 2.0x)
    multiplier = min(conf_mult * hist_mult, 2.0)
    
    logger.debug(
        f"[{symbol}] Position multiplier: {multiplier:.2f}x "
        f"(confidence={confidence:.1%} × history={hist_wr:.1%})"
    )
    
    return multiplier


def get_confidence_adjustment(symbol: str, base_confidence: float) -> float:
    """
    Adjust confidence based on historical performance.
    
    Args:
        symbol: Trading symbol
        base_confidence: Model's raw confidence (0.0-1.0)
    
    Returns:
        float: Adjusted confidence (0.0-0.95)
        
    Strategy:
    - Blacklisted assets: 0.0 (force no trade)
    - Perfect history (100%): +10-15% confidence boost
    - Excellent history (>90%): +5-10% confidence boost
    - Good history (70-90%): +3-5% confidence boost
    - Unknown/mediocre: -5% confidence penalty (conservative)
    
    Examples:
        >>> get_confidence_adjustment("SOL", 0.75)
        0.0  # Blacklisted
        
        >>> get_confidence_adjustment("CHZ", 0.65)
        0.80  # 100% history, +15% boost (capped at 0.85)
        
        >>> get_confidence_adjustment("AAPL", 0.70)
        0.665  # Unknown, -5% penalty
    """
    symbol = symbol.upper()
    
    # Blacklist: Force zero confidence
    if symbol in BLACKLIST:
        logger.warning(f"[{symbol}] BLACKLISTED - Confidence forced to 0.0")
        return 0.0
    
    # Whitelist: Boost confidence based on historical performance
    if symbol in WHITELIST:
        win_rate = WHITELIST[symbol]
        
        if win_rate >= 0.95:
            # Perfect or near-perfect history: significant boost
            boost = 0.15
        elif win_rate >= 0.90:
            # Excellent history: strong boost
            boost = 0.10
        elif win_rate >= 0.80:
            # Very good history: moderate boost
            boost = 0.07
        elif win_rate >= 0.70:
            # Good history: small boost
            boost = 0.05
        else:
            # Decent history: minimal boost
            boost = 0.03
        
        adjusted = min(base_confidence + boost, 0.85)  # Cap at 85% (markets are uncertain)
        
        logger.info(
            f"[{symbol}] Confidence boost: {base_confidence:.1%} → {adjusted:.1%} "
            f"(+{boost:.1%} for {win_rate:.1%} historical)"
        )
        
        return adjusted
    
    # Unknown asset: Slight penalty (be conservative)
    adjusted = base_confidence * 0.95
    logger.debug(f"[{symbol}] Unknown asset - Conservative adjustment: {base_confidence:.1%} → {adjusted:.1%}")
    
    return adjusted


def get_trading_stats() -> dict:
    """
    Get current trading control statistics.
    
    Returns:
        dict with blacklist/whitelist counts and settings
    """
    return {
        "blacklist_count": len(BLACKLIST),
        "whitelist_count": len(WHITELIST),
        "min_confidence": MIN_CONFIDENCE,
        "whitelist_only_mode": WHITELIST_ONLY_MODE,
        "blacklist": sorted(list(BLACKLIST)),
        "whitelist_symbols": sorted(list(WHITELIST.keys())),
        "whitelist_detail": {
            symbol: f"{rate:.1%}" for symbol, rate in sorted(WHITELIST.items(), key=lambda x: -x[1])
        }
    }


# ============================================================================
# TESTING / VALIDATION
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("=== Ghost Protocol Trading Controls ===\n")
    
    # Test blacklist
    print("1. Testing BLACKLIST:")
    for symbol in ["SOL", "BTC", "ETH", "XRP"]:
        can_trade, reason = should_trade(symbol, 0.85)
        print(f"   {symbol}: {can_trade} - {reason}")
    
    print("\n2. Testing WHITELIST:")
    for symbol in ["CHZ", "ZEC", "ICP", "AAVE"]:
        can_trade, reason = should_trade(symbol, 0.75)
        print(f"   {symbol}: {can_trade} - {reason}")
    
    print("\n3. Testing UNKNOWN assets:")
    for symbol in ["AAPL", "GOOGL", "TSLA"]:
        can_trade, reason = should_trade(symbol, 0.72)
        print(f"   {symbol}: {can_trade} - {reason}")
    
    print("\n4. Testing CONFIDENCE adjustments:")
    for symbol, base_conf in [("SOL", 0.80), ("CHZ", 0.65), ("AAPL", 0.70)]:
        adjusted = get_confidence_adjustment(symbol, base_conf)
        print(f"   {symbol}: {base_conf:.1%} → {adjusted:.1%}")
    
    print("\n5. Testing POSITION sizing:")
    for symbol, conf in [("CHZ", 0.85), ("AAVE", 0.75), ("AAPL", 0.72)]:
        mult = get_position_multiplier(symbol, conf)
        print(f"   {symbol} @ {conf:.1%}: {mult:.2f}x position")
    
    print("\n6. Trading Statistics:")
    stats = get_trading_stats()
    print(f"   Blacklist: {stats['blacklist_count']} assets")
    print(f"   Whitelist: {stats['whitelist_count']} assets")
    print(f"   Min Confidence: {stats['min_confidence']:.1%}")
    print(f"   Whitelist-Only Mode: {stats['whitelist_only_mode']}")
