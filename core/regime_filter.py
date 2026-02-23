"""
Regime Filter — Gate crypto BUYs during market dumps
=====================================================

This is the #1 improvement for Ghost's win rate. Without it, Ghost issues
BUY signals into dumps and gets correlated stop-outs (e.g., Feb 21: ICP/ILV/CHZ
all BUY → all stopped → -$13.50 in one morning).

Architecture:
    V3 pipeline produces (stocks, crypto) pick lists
    → THIS FILTER gates crypto BUYs when BTC is dumping
    → Only clean picks reach format_top10_message()

Regime detection:
    - BTC 24h change (fast signal) — from crypto_providers quorum
    - BTC 7d change (trend signal) — from Binance OHLCV or CoinGecko
    - SPY regime for stocks — from Polygon (existing market_gates)

Actions when bearish:
    LEVEL 1 (CAUTION): BTC -3% to -5% 24h → suppress lowest-confidence crypto BUY
    LEVEL 2 (BEARISH): BTC -5% to -8% 24h OR -8% 7d → suppress ALL crypto BUYs
    LEVEL 3 (CRASH):   BTC -8%+ 24h → suppress ALL crypto BUYs, send crash alert

Stocks get their own gate: SPY below 20-day MA → suppress stock BUYs.
SELL signals are NEVER suppressed — selling into a dump is the right trade.

Created: Feb 23, 2026
Reason: 7W/17L (29% WR), 80% of losses are correlated BUY stops in crypto dumps
"""

import os
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

LOGGER = logging.getLogger("ghost.regime")

# ============================================================================
# CONFIGURATION — tunable via env vars, defaults based on backtest analysis
# ============================================================================

# Master switch
REGIME_ENABLED = os.getenv("REGIME_FILTER_V2", "1") == "1"

# BTC 24h thresholds (fast signal — catches intraday dumps)
BTC_24H_CAUTION  = float(os.getenv("BTC_24H_CAUTION", "-3.0"))   # -3% = caution
BTC_24H_BEARISH  = float(os.getenv("BTC_24H_BEARISH", "-5.0"))   # -5% = bearish
BTC_24H_CRASH    = float(os.getenv("BTC_24H_CRASH", "-8.0"))     # -8% = crash

# BTC 7d threshold (trend signal — catches multi-day bleeds)
BTC_7D_BEARISH   = float(os.getenv("BTC_7D_BEARISH", "-8.0"))    # -8% over 7d = bearish

# SPY threshold for stocks
SPY_BELOW_MA_PENALTY = float(os.getenv("SPY_MA_PENALTY", "0.10"))  # -10% confidence

# Cache TTL (don't hammer APIs every call)
CACHE_TTL_SECONDS = int(os.getenv("REGIME_CACHE_TTL", "300"))  # 5 min cache

# ============================================================================
# REGIME STATE — cached to avoid repeated API calls
# ============================================================================

_regime_cache: Dict[str, Any] = {
    "btc_24h_pct": None,
    "btc_7d_pct": None,
    "btc_price": None,
    "spy_regime": None,
    "level": "UNKNOWN",       # CLEAR / CAUTION / BEARISH / CRASH
    "last_update": 0,
    "last_error": None,
}


# ============================================================================
# CORE: Fetch BTC regime data
# ============================================================================

async def _fetch_btc_24h() -> Optional[float]:
    """
    Get BTC 24h price change % using the existing crypto provider quorum.
    Fast — uses cached data from CoinGecko/Binance/Coinbase.
    """
    try:
        from core.crypto.crypto_providers import get_crypto_price_quorum
        result = await get_crypto_price_quorum("BTC", use_cache=True)
        if result and result.get("change_24h_pct") is not None:
            pct = float(result["change_24h_pct"])
            LOGGER.info(f"₿ BTC 24h: {pct:+.2f}% (price: ${result.get('price', 0):,.0f})")
            return pct
    except Exception as e:
        LOGGER.warning(f"BTC 24h fetch failed: {e}")
    return None


async def _fetch_btc_7d() -> Optional[float]:
    """
    Get BTC 7-day price change %. Uses Binance OHLCV as primary,
    CoinGecko historical as fallback.
    """
    # Try Binance first (no API key needed, fast)
    try:
        from core.providers.binance_ohlcv import get_binance_ohlcv
        bars = get_binance_ohlcv("BTC", interval="1d", limit=8)
        if bars and len(bars) >= 2:
            price_now = bars[-1]["close"]
            price_7d = bars[0]["close"]
            pct = ((price_now - price_7d) / price_7d) * 100
            LOGGER.info(f"₿ BTC 7d: {pct:+.2f}% (${price_7d:,.0f} → ${price_now:,.0f})")
            return round(pct, 2)
    except Exception as e:
        LOGGER.warning(f"Binance BTC 7d failed: {e}")

    # Fallback: CoinGecko historical
    try:
        from core.crypto.crypto_providers import CoinGeckoProvider
        provider = CoinGeckoProvider()
        history = provider.get_historical("BTC", days=8)
        if history and len(history) >= 2:
            price_now = history[-1]["price"]
            price_7d = history[0]["price"]
            pct = ((price_now - price_7d) / price_7d) * 100
            LOGGER.info(f"₿ BTC 7d (CoinGecko): {pct:+.2f}%")
            return round(pct, 2)
    except Exception as e:
        LOGGER.warning(f"CoinGecko BTC 7d also failed: {e}")

    return None


async def _fetch_spy_regime() -> Optional[str]:
    """
    Get SPY regime using existing market_gates infrastructure.
    Returns 'bull', 'bear', or 'unknown'.
    """
    try:
        from core.market_gates import RegimeFilter
        rf = RegimeFilter()
        spy_data = await rf.get_spy_regime()
        regime = spy_data.get("regime", "unknown")
        LOGGER.info(f"📊 SPY regime: {regime.upper()}")
        return regime
    except Exception as e:
        LOGGER.warning(f"SPY regime fetch failed: {e}")
    return "unknown"


# ============================================================================
# CORE: Determine regime level
# ============================================================================

async def get_regime_state(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get current market regime. Cached for CACHE_TTL_SECONDS.

    Returns dict with:
        level: CLEAR / CAUTION / BEARISH / CRASH
        btc_24h_pct: float
        btc_7d_pct: float
        btc_price: float
        spy_regime: bull / bear / unknown
        reason: human-readable explanation
    """
    global _regime_cache

    # Return cache if fresh
    now = time.time()
    if not force_refresh and _regime_cache["last_update"] > 0:
        age = now - _regime_cache["last_update"]
        if age < CACHE_TTL_SECONDS and _regime_cache["level"] != "UNKNOWN":
            return _regime_cache

    # Fetch all data concurrently
    try:
        btc_24h, btc_7d, spy = await asyncio.gather(
            _fetch_btc_24h(),
            _fetch_btc_7d(),
            _fetch_spy_regime(),
            return_exceptions=True,
        )

        # Handle exceptions from gather
        if isinstance(btc_24h, Exception):
            LOGGER.error(f"BTC 24h exception: {btc_24h}")
            btc_24h = None
        if isinstance(btc_7d, Exception):
            LOGGER.error(f"BTC 7d exception: {btc_7d}")
            btc_7d = None
        if isinstance(spy, Exception):
            LOGGER.error(f"SPY exception: {spy}")
            spy = "unknown"

    except Exception as e:
        LOGGER.error(f"Regime data fetch failed: {e}")
        btc_24h, btc_7d, spy = None, None, "unknown"

    # Determine level from BTC data
    level = "CLEAR"
    reason = "Market conditions normal"

    if btc_24h is not None:
        if btc_24h <= BTC_24H_CRASH:
            level = "CRASH"
            reason = f"BTC crashed {btc_24h:+.1f}% in 24h"
        elif btc_24h <= BTC_24H_BEARISH:
            level = "BEARISH"
            reason = f"BTC down {btc_24h:+.1f}% in 24h"
        elif btc_24h <= BTC_24H_CAUTION:
            level = "CAUTION"
            reason = f"BTC dipping {btc_24h:+.1f}% in 24h"

    # 7d trend can upgrade level (but not downgrade)
    if btc_7d is not None and btc_7d <= BTC_7D_BEARISH:
        if level in ("CLEAR", "CAUTION"):
            level = "BEARISH"
            reason = f"BTC down {btc_7d:+.1f}% over 7 days"

    # Update cache
    _regime_cache = {
        "btc_24h_pct": btc_24h,
        "btc_7d_pct": btc_7d,
        "btc_price": None,  # Will be filled from quorum if available
        "spy_regime": spy,
        "level": level,
        "reason": reason,
        "last_update": now,
        "last_error": None,
    }

    LOGGER.info(f"🌡️ REGIME: {level} — {reason}")
    return _regime_cache


# ============================================================================
# CORE: Apply regime filter to pick lists
# ============================================================================

def _is_buy_direction(pick: Dict) -> bool:
    """Check if a pick is a BUY (UP) direction."""
    d = pick.get("direction", "").upper()
    return d in ("UP", "BUY")


async def apply_regime_filter(
    stocks: List[Dict],
    crypto: List[Dict],
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """
    Apply regime filter to stock and crypto pick lists.

    This is the ONLY function that send_top10() needs to call.

    Rules:
        CLEAR   → pass everything through unchanged
        CAUTION → drop lowest-confidence crypto BUY (keep the rest)
        BEARISH → drop ALL crypto BUYs (keep SELLs)
        CRASH   → drop ALL crypto BUYs + send crash alert

    Stock BUYs: Suppressed when SPY regime is 'bear'.
    SELL signals: NEVER filtered (selling into a dump is correct).

    Args:
        stocks: list of stock pick dicts from V3 pipeline
        crypto: list of crypto pick dicts from V3 pipeline

    Returns:
        (filtered_stocks, filtered_crypto, regime_info)
    """
    if not REGIME_ENABLED:
        return stocks, crypto, {
            "enabled": False,
            "level": "DISABLED",
            "filtered_count": 0,
        }

    regime = await get_regime_state()
    level = regime.get("level", "CLEAR")

    filtered_crypto = list(crypto)
    filtered_stocks = list(stocks)
    removed_crypto: List[Dict] = []
    removed_stocks: List[Dict] = []

    # ── Crypto regime gating ──────────────────────────────────
    if level == "CLEAR":
        pass  # All signals pass through

    elif level == "CAUTION":
        # Drop the weakest crypto BUY (keep strongest, keep all SELLs)
        buys = [p for p in filtered_crypto if _is_buy_direction(p)]
        sells = [p for p in filtered_crypto if not _is_buy_direction(p)]

        if len(buys) > 1:
            # Sort by confidence ascending, remove the weakest
            buys_sorted = sorted(buys, key=lambda p: p.get("confidence", 0))
            weakest = buys_sorted[0]
            removed_crypto.append(weakest)
            buys = buys_sorted[1:]
            LOGGER.info(
                f"⚠️ REGIME CAUTION: Dropped weakest crypto BUY "
                f"{weakest['symbol']} ({weakest.get('confidence', 0):.0%})"
            )

        filtered_crypto = sells + buys

    elif level in ("BEARISH", "CRASH"):
        # Drop ALL crypto BUYs — keep only SELLs
        for p in filtered_crypto:
            if _is_buy_direction(p):
                removed_crypto.append(p)
                LOGGER.warning(
                    f"🚫 REGIME {level}: Blocked crypto BUY "
                    f"{p['symbol']} ({p.get('confidence', 0):.0%})"
                )
        filtered_crypto = [p for p in filtered_crypto if not _is_buy_direction(p)]

    # ── Stock regime gating ───────────────────────────────────
    spy_regime = regime.get("spy_regime", "unknown")
    if spy_regime == "bear":
        for p in filtered_stocks:
            if _is_buy_direction(p):
                removed_stocks.append(p)
                LOGGER.warning(
                    f"🚫 SPY BEAR: Blocked stock BUY "
                    f"{p['symbol']} ({p.get('confidence', 0):.0%})"
                )
        filtered_stocks = [p for p in filtered_stocks if not _is_buy_direction(p)]

    # Build regime info for logging/debug
    regime_info = {
        "enabled": True,
        "level": level,
        "reason": regime.get("reason", ""),
        "btc_24h_pct": regime.get("btc_24h_pct"),
        "btc_7d_pct": regime.get("btc_7d_pct"),
        "spy_regime": spy_regime,
        "crypto_before": len(crypto),
        "crypto_after": len(filtered_crypto),
        "crypto_removed": [p["symbol"] for p in removed_crypto],
        "stocks_before": len(stocks),
        "stocks_after": len(filtered_stocks),
        "stocks_removed": [p["symbol"] for p in removed_stocks],
        "filtered_count": len(removed_crypto) + len(removed_stocks),
    }

    if regime_info["filtered_count"] > 0:
        LOGGER.warning(
            f"🌡️ REGIME FILTER: {level} — removed {regime_info['filtered_count']} picks "
            f"(crypto: {[p['symbol'] for p in removed_crypto]}, "
            f"stocks: {[p['symbol'] for p in removed_stocks]})"
        )
    else:
        LOGGER.info(f"🌡️ REGIME FILTER: {level} — all picks passed")

    return filtered_stocks, filtered_crypto, regime_info


# ============================================================================
# TELEGRAM: Regime alert message
# ============================================================================

def format_regime_alert(regime_info: Dict[str, Any]) -> Optional[str]:
    """
    Format a Telegram message when regime filter removes picks.
    Returns None if no picks were filtered (no alert needed).
    """
    if regime_info.get("filtered_count", 0) == 0:
        return None

    level = regime_info.get("level", "UNKNOWN")
    reason = regime_info.get("reason", "")
    btc_24h = regime_info.get("btc_24h_pct")
    btc_7d = regime_info.get("btc_7d_pct")

    # Level emoji
    level_emoji = {
        "CAUTION": "⚠️",
        "BEARISH": "🔴",
        "CRASH": "🚨",
    }.get(level, "⚠️")

    lines = [
        f"{level_emoji} <b>REGIME FILTER — {level}</b>",
        f"",
        f"<b>Reason:</b> {reason}",
    ]

    if btc_24h is not None:
        lines.append(f"₿ BTC 24h: {btc_24h:+.1f}%")
    if btc_7d is not None:
        lines.append(f"₿ BTC 7d: {btc_7d:+.1f}%")

    spy = regime_info.get("spy_regime")
    if spy and spy != "unknown":
        lines.append(f"📊 SPY: {spy.upper()}")

    # What was filtered
    crypto_removed = regime_info.get("crypto_removed", [])
    stocks_removed = regime_info.get("stocks_removed", [])

    if crypto_removed:
        lines.append(f"")
        lines.append(f"<b>Crypto BUYs blocked:</b>")
        for sym in crypto_removed:
            lines.append(f"  🚫 {sym}")

    if stocks_removed:
        lines.append(f"")
        lines.append(f"<b>Stock BUYs blocked:</b>")
        for sym in stocks_removed:
            lines.append(f"  🚫 {sym}")

    remaining = (
        regime_info.get("crypto_after", 0) + regime_info.get("stocks_after", 0)
    )
    lines.append(f"")
    lines.append(f"<b>Remaining picks:</b> {remaining}")
    lines.append(f"")
    lines.append(f"<i>Regime filter protects against correlated BUY stops in dumps.</i>")

    return "\n".join(lines)


# ============================================================================
# DEBUG: Status endpoint data
# ============================================================================

async def get_regime_debug() -> Dict[str, Any]:
    """
    Get current regime status for debug endpoint.
    Forces a fresh fetch (ignores cache).
    """
    regime = await get_regime_state(force_refresh=True)

    return {
        "enabled": REGIME_ENABLED,
        "config": {
            "btc_24h_caution": BTC_24H_CAUTION,
            "btc_24h_bearish": BTC_24H_BEARISH,
            "btc_24h_crash": BTC_24H_CRASH,
            "btc_7d_bearish": BTC_7D_BEARISH,
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
        },
        "current": {
            "level": regime.get("level"),
            "reason": regime.get("reason"),
            "btc_24h_pct": regime.get("btc_24h_pct"),
            "btc_7d_pct": regime.get("btc_7d_pct"),
            "spy_regime": regime.get("spy_regime"),
            "last_update": datetime.fromtimestamp(
                regime.get("last_update", 0)
            ).isoformat() if regime.get("last_update") else None,
        },
        "action_matrix": {
            "CLEAR": "All picks pass through",
            "CAUTION": "Drop weakest crypto BUY",
            "BEARISH": "Block ALL crypto BUYs (SELLs pass)",
            "CRASH": "Block ALL crypto BUYs + crash alert",
        },
    }
