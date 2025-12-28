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
# LEARNING INTEGRATION
# ============================================================================

# Learning thresholds
LEARNING_MIN_PREDICTIONS = 10  # Need at least 10 predictions to evaluate
LEARNING_EXCLUDE_ACCURACY = 40.0  # Exclude symbols with <40% accuracy
LEARNING_BOOST_ACCURACY = 70.0  # Boost symbols with >70% accuracy
LEARNING_BOOST_AMOUNT = 0.15  # 15% confidence boost

# ============================================================================
# LEARNING MODE - Re-enabled Dec 28, 2025
# Only EXCLUSIONS are active (no boosting until data is fully validated)
# ============================================================================
LEARNING_ENABLED = True  # Master switch
LEARNING_EXCLUDE_ENABLED = True  # Exclusions active (low accuracy symbols removed)
LEARNING_BOOST_ENABLED = False  # Boosting disabled until data validated

# ============================================================================
# HARDCODED EXCLUSIONS - Symbols with historically bad accuracy (<40%)
# These are ALWAYS excluded regardless of learning data
# Updated Dec 28, 2025 based on actual accuracy data + Telegram analysis
# ============================================================================
HARDCODED_EXCLUSIONS = {
    # === FROM TELEGRAM ANALYSIS (Dec 28) - Confirmed stop-outs and wrong directions ===
    'SAND': 'STOPPED OUT +3.3% on Dec 27 (Telegram confirmed)',  # Added Dec 28
    'FLOW': 'STOPPED OUT +12.5% on Dec 27 (Telegram confirmed)',  # Added Dec 28
    'HBAR': 'Wrong direction - went UP when predicted DOWN (Telegram confirmed)',  # Added Dec 28
    'ILV': 'Gaming token - pump/dump prone, volatile',  # Added Dec 28
    'BAND': 'Oracle token - low liquidity',  # Added Dec 28
    'DIA': 'Oracle token - low liquidity',  # Added Dec 28
    
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
    'FLOKI': '11% accuracy (1/9)',
    'LDO': '11% accuracy (1/9)',
    'BAL': '11% accuracy (1/9)',
    'RNDR': '11% accuracy (1/9)',
    'XLM': '22% accuracy (2/9)',
    'ETC': '22% accuracy (2/9)',
    'APE': '22% accuracy (2/9)',
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
    1. HARDCODED_EXCLUSIONS - Always excluded (known bad symbols)
    2. Learning data - Excluded if <40% accuracy after 10+ predictions (if LEARNING_EXCLUDE_ENABLED)
    
    Args:
        symbol: The symbol to check
        accuracy_data: Dict from get_symbol_accuracy_from_postgres()
        
    Returns:
        (should_exclude: bool, reason: str)
    """
    # PRIORITY 1: Check hardcoded exclusions FIRST (always applies)
    if symbol in HARDCODED_EXCLUSIONS:
        return True, f"HARDCODED: {HARDCODED_EXCLUSIONS[symbol]}"
    
    # PRIORITY 2: Check learning data (only if LEARNING_EXCLUDE_ENABLED)
    if not LEARNING_EXCLUDE_ENABLED:
        return False, "learning_exclusions_disabled"
    
    if symbol not in accuracy_data:
        return False, "no_data"
    
    data = accuracy_data[symbol]
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


def format_top10_message(stocks: List[Dict], crypto: List[Dict], inverse_mode: bool = True) -> str:
    """
    Format the TOP 10 message in the EXACT format requested.
    
    Args:
        stocks: List of top 5 stock predictions
        crypto: List of top 5 crypto predictions  
        inverse_mode: If True, show "INVERSE GHOST" in title
    """
    ct = get_central_time()
    date_str = ct.strftime("%b %d, %Y")
    
    title = "🎯 INVERSE GHOST TOP 10" if inverse_mode else "🎯 GHOST TOP 10"
    
    lines = [
        f"{title} — {date_str}",
        "",
        "📈 STOCKS (5)",
        "━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]
    
    if stocks:
        for i, s in enumerate(stocks[:5], 1):
            # Use the direction already calculated (which has inverse applied)
            direction = s.get('direction', 'DOWN')
            
            # In INVERSE mode with INVERSE_GHOST=1, raw predictions are flipped
            # So if direction is "DOWN", it means Ghost raw said UP (we inverted to DOWN = SELL)
            # If direction is "UP", it means Ghost raw said DOWN (we inverted to UP = BUY)
            if direction == "UP":
                action = "BUY"
                emoji = "🟢"
            else:
                action = "SELL"
                emoji = "🔴"
            
            lines.append(f"{i}. {emoji} {s['symbol']} — {action}")
            lines.append(f"   Current: {format_price(s['current'])}")
            lines.append(f"   Buy In: {format_price(s['buy_in'])}")
            lines.append(f"   Sell: {format_price(s['sell'])}")
            lines.append(f"   48hr Prediction: {format_price(s['prediction_48h'])}")
            lines.append(f"   Confidence: {s['confidence']:.0%}")
            lines.append("")
    else:
        lines.append("   (No stock picks today)")
        lines.append("")
    
    lines.append("📊 CRYPTO (5)")
    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")
    
    if crypto:
        for i, c in enumerate(crypto[:5], 1):
            # Use the direction already calculated (which has inverse applied)
            direction = c.get('direction', 'DOWN')
            
            if direction == "UP":
                action = "BUY"
                emoji = "🟢"
            else:
                action = "SELL"
                emoji = "🔴"
            
            lines.append(f"{i}. {emoji} {c['symbol']} — {action}")
            lines.append(f"   Current: {format_price(c['current'])}")
            lines.append(f"   Buy In: {format_price(c['buy_in'])}")
            lines.append(f"   Sell: {format_price(c['sell'])}")
            lines.append(f"   48hr Prediction: {format_price(c['prediction_48h'])}")
            lines.append(f"   Confidence: {c['confidence']:.0%}")
            lines.append("")
    else:
        lines.append("   (No crypto picks today)")
        lines.append("")
    
    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("⏱️ 48hr Tracking Active")
    lines.append("📊 Updates on significant moves (>3%)")
    lines.append("🎯 Alerts when targets hit")
    lines.append("")
    lines.append("Ghost is watching.")
    
    return "\n".join(lines)


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
    
    remaining = 10 - len(alerts)
    if remaining > 0:
        lines.append(f"Remaining {remaining} picks still tracking...")
    
    return "\n".join(lines)


class GhostNotificationSystem:
    """
    The ONE notification system for Ghost.
    
    Handles:
    - Morning TOP 10 at 8 AM Central
    - Updates every 4 hours (if significant moves)
    - Instant alerts when targets/stops hit
    """
    
    def __init__(self, send_telegram_func: Callable[[str], bool] = None):
        self.send_telegram = send_telegram_func
        self._tracked_picks: List[TrackedPick] = []
        self._last_top10_date: str = ""
        self._last_update_hour: int = -1
        self._db_path = TRACKING_DB
        self._init_db()
    
    def _init_db(self):
        """Initialize tracking database"""
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
    
    def set_telegram_func(self, func: Callable[[str], bool]):
        """Set the function to send Telegram messages"""
        self.send_telegram = func
    
    def get_top10_predictions(self, latest_predictions: Dict[str, Dict]) -> tuple:
        """
        Get top 5 crypto and top 5 stocks from latest predictions.
        
        LEARNING INTEGRATION:
        - Excludes symbols with <40% accuracy (after 10+ predictions)
        - Boosts confidence by 15% for symbols with >70% accuracy
        
        Args:
            latest_predictions: Dict of symbol -> prediction data
            
        Returns:
            (stocks_list, crypto_list) - each sorted by confidence
        """
        from core.asset_classifier import get_asset_type, AssetClassifier
        
        inverse_mode = os.getenv("INVERSE_GHOST_MODE", "1") == "1"
        
        stocks = []
        crypto = []
        
        # Track price refresh attempts for logging
        prices_refreshed = 0
        prices_stale_skipped = 0
        stablecoins_skipped = 0
        
        # LEARNING: Get symbol accuracy data from PostgreSQL
        accuracy_data = get_symbol_accuracy_from_postgres()
        learning_excluded = 0
        learning_boosted = 0
        excluded_symbols = []
        boosted_symbols = []
        
        for symbol, pred in latest_predictions.items():
            if not isinstance(pred, dict):
                continue
            
            # CRITICAL: Skip stablecoins (USDC, DAI, USDT, etc.) - they don't move!
            if AssetClassifier.is_stablecoin(symbol):
                stablecoins_skipped += 1
                LOGGER.debug(f"[NOTIFICATIONS] Skipping stablecoin: {symbol}")
                continue
            
            # LEARNING: Check if symbol should be excluded due to low accuracy
            should_exclude, exclude_reason = should_exclude_symbol(symbol, accuracy_data)
            if should_exclude:
                learning_excluded += 1
                excluded_symbols.append(f"{symbol} ({exclude_reason})")
                LOGGER.info(f"[LEARNING] EXCLUDED {symbol}: {exclude_reason}")
                continue
            
            confidence = pred.get("confidence", 0)
            if confidence < 0.70:  # At least 70% to consider
                continue
            
            # LEARNING: Apply confidence boost for high-accuracy symbols
            boost_multiplier, boost_reason = get_confidence_boost(symbol, accuracy_data)
            original_confidence = confidence
            if boost_multiplier > 1.0:
                confidence = min(1.0, confidence * boost_multiplier)  # Cap at 100%
                learning_boosted += 1
                boosted_symbols.append(f"{symbol} ({original_confidence:.0%}→{confidence:.0%})")
                LOGGER.info(f"[LEARNING] BOOSTED {symbol}: {original_confidence:.0%} → {confidence:.0%} ({boost_reason})")
            
            # Get current price from prediction (may be missing or stale)
            current_price = (pred.get("price") or 
                           pred.get("current_price") or 
                           pred.get("entry_price") or 0)
            
            # Classify asset type early (needed for price refresh)
            asset_class = get_asset_type(symbol)
            
            # Skip stablecoins by type too (belt and suspenders)
            if asset_class == "stablecoin":
                stablecoins_skipped += 1
                continue
            
            # ALWAYS try to get fresh price if we don't have one, or if price is stale
            price_timestamp = pred.get("run_at") or pred.get("timestamp") or 0
            price_age_seconds = time.time() - price_timestamp if price_timestamp > 0 else 999999
            need_fresh_price = current_price <= 0 or price_age_seconds > 300  # 5 min
            
            if need_fresh_price:
                try:
                    if asset_class == "crypto":
                        from core.crypto.crypto_providers import get_crypto_price_quorum
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        fresh_price_data = loop.run_until_complete(
                            get_crypto_price_quorum(symbol, use_cache=False)
                        )
                        if fresh_price_data and fresh_price_data.get("price", 0) > 0:
                            current_price = fresh_price_data["price"]
                            prices_refreshed += 1
                            LOGGER.debug(f"[NOTIFICATIONS] Refreshed {symbol} price: ${current_price:.4f}")
                    else:
                        # For stocks, use turbo provider
                        from core.providers.turbo_provider import get_turbo_provider
                        turbo = get_turbo_provider()
                        fresh_price_data = turbo.turbo_stock_price(symbol, max_budget_s=2.0)
                        if fresh_price_data.get("ok") and fresh_price_data.get("price", 0) > 0:
                            current_price = fresh_price_data["price"]
                            prices_refreshed += 1
                            LOGGER.debug(f"[NOTIFICATIONS] Refreshed {symbol} price: ${current_price:.2f}")
                except Exception as e:
                    LOGGER.warning(f"[NOTIFICATIONS] Failed to refresh {symbol} price: {e}")
            
            # Skip if we still don't have a valid price
            if current_price <= 0:
                LOGGER.debug(f"[NOTIFICATIONS] Skipping {symbol} - no valid price available")
                continue
            
            # Direction is ALREADY inverted in _LATEST_PREDICTIONS when INVERSE_GHOST=1
            # DO NOT invert again here - that was causing double-inversion bug
            direction = pred.get("direction", "DOWN")
            
            # Calculate 48hr prediction price
            # If direction is UP, price goes higher; if DOWN, price goes lower
            if direction == "UP":
                # Estimate 48hr target (typically 3-6% move)
                move_pct = 0.05 if get_asset_type(symbol) == "crypto" else 0.03
                prediction_48h = current_price * (1 + move_pct)
            else:
                move_pct = 0.05 if get_asset_type(symbol) == "crypto" else 0.03
                prediction_48h = current_price * (1 - move_pct)
            
            # Calculate buy-in and sell prices (entry zone)
            buy_in = current_price * 0.99  # 1% below current
            sell_at = current_price * 1.02  # 2% above current (for profit taking)
            
            pick = {
                "symbol": symbol,
                "current": current_price,
                "prediction_48h": prediction_48h,
                "buy_in": buy_in,
                "sell": sell_at,
                "confidence": confidence,
                "direction": direction,
                "learning_boosted": boost_multiplier > 1.0,  # Flag for boosted picks
            }
            
            # Add to appropriate list based on asset_class (already computed above)
            if asset_class == "crypto":
                crypto.append(pick)
            else:
                stocks.append(pick)
        
        # Log learning stats
        if learning_excluded > 0:
            LOGGER.info(f"[LEARNING] 🚫 EXCLUDED {learning_excluded} low-accuracy symbols: {', '.join(excluded_symbols[:5])}")
        if learning_boosted > 0:
            LOGGER.info(f"[LEARNING] 🚀 BOOSTED {learning_boosted} high-accuracy symbols: {', '.join(boosted_symbols[:5])}")
        
        # Log skip stats
        if stablecoins_skipped > 0:
            LOGGER.info(f"[NOTIFICATIONS] Skipped {stablecoins_skipped} stablecoins from predictions")
        
        # Sort by confidence, take top 5
        stocks.sort(key=lambda x: x["confidence"], reverse=True)
        crypto.sort(key=lambda x: x["confidence"], reverse=True)
        
        return stocks[:5], crypto[:5]
    
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
        
        inverse_mode = os.getenv("INVERSE_GHOST_MODE", "1") == "1"
        message = format_top10_message(stocks, crypto, inverse_mode)
        
        LOGGER.info(f"[NOTIFICATIONS] Sending TOP 10 ({len(stocks)} stocks, {len(crypto)} crypto)")
        
        success = self.send_telegram(message)
        
        if success:
            self._last_top10_date = today
            LOGGER.info("[NOTIFICATIONS] ✅ TOP 10 sent successfully")
            
            # Register picks for tracking
            self._register_picks_for_tracking(stocks + crypto)
        else:
            LOGGER.error("[NOTIFICATIONS] ❌ Failed to send TOP 10")
        
        return success
    
    def _register_picks_for_tracking(self, picks: List[Dict]):
        """Register picks for 48-hour tracking"""
        conn = sqlite3.connect(self._db_path)
        now = get_central_time()
        expires = now + timedelta(hours=48)
        
        for p in picks:
            action, _, _ = determine_action(p['current'], p['prediction_48h'], p['confidence'])
            
            conn.execute("""
                INSERT INTO tracked_picks 
                (symbol, asset_type, direction, entry_price, target_price, stop_price, 
                 prediction_48h, confidence, entry_time, expires_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """, (
                p['symbol'],
                'crypto' if p.get('asset_type') == 'crypto' else 'stock',
                action,
                p['current'],
                p['prediction_48h'],  # Target is the 48hr prediction
                p['current'] * 0.95 if action == 'BUY' else p['current'] * 1.05,  # 5% stop
                p['prediction_48h'],
                p['confidence'],
                now.isoformat(),
                expires.isoformat(),
            ))
        
        conn.commit()
        conn.close()
        LOGGER.info(f"[NOTIFICATIONS] Registered {len(picks)} picks for 48h tracking")
    
    def check_for_updates(self, get_price_func: Callable[[str], float]) -> bool:
        """
        Check all tracked picks for significant moves.
        
        Sends update message if any pick moved >3%.
        Should be called every 15-30 minutes.
        """
        if not self.send_telegram:
            return False
        
        # Load active picks from DB
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
            
            # Determine if on track based on direction
            if direction == "BUY":
                emoji = "🟢"
                on_track = current >= entry * 0.98  # Allow 2% buffer
                near_target = current >= target * 0.98
                near_stop = current <= stop * 1.02
                # OFF PATH = BUY but price dropped >2%
                is_off_path = pct_change < -0.02
            else:  # SELL
                emoji = "🔴"
                on_track = current <= entry * 1.02  # Allow 2% buffer
                near_target = current <= target * 1.02
                near_stop = current >= stop * 0.98
                # OFF PATH = SELL but price rose >2%
                is_off_path = pct_change > 0.02
            
            # Check for target/stop hit (HIGHEST PRIORITY)
            if near_target and abs(pct_change) >= 0.02:
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
                    off_path_crypto.append(off_path_pick)
                else:
                    off_path_stocks.append(off_path_pick)
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
            msg = format_alert_message(alerts)
            self.send_telegram(msg)
            LOGGER.info(f"[NOTIFICATIONS] Sent {len(alerts)} target/stop alerts")
            
            # Update status in DB
            conn = sqlite3.connect(self._db_path)
            for a in alerts:
                status = "target_hit" if a['type'] == 'target_hit' else 'stop_hit'
                conn.execute("UPDATE tracked_picks SET status = ? WHERE symbol = ?", 
                           (status, a['symbol']))
            conn.commit()
            conn.close()
        
        # Send OFF PATH alerts immediately (separate for stocks vs crypto)
        if off_path_stocks:
            msg = format_off_path_alert(off_path_stocks, asset_type="stock")
            self.send_telegram(msg)
            LOGGER.info(f"[NOTIFICATIONS] Sent OFF PATH alert for {len(off_path_stocks)} stocks")
        
        if off_path_crypto:
            msg = format_off_path_alert(off_path_crypto, asset_type="crypto")
            self.send_telegram(msg)
            LOGGER.info(f"[NOTIFICATIONS] Sent OFF PATH alert for {len(off_path_crypto)} crypto")
        
        # Send scheduled updates (12 PM, 4 PM, 8 PM) - only if no alerts sent
        ct = get_central_time()
        if ct.hour in UPDATE_HOURS and ct.hour != self._last_update_hour:
            if updates and not alerts and not off_path_stocks and not off_path_crypto:
                msg = format_update_message(updates)
                self.send_telegram(msg)
                self._last_update_hour = ct.hour
                LOGGER.info(f"[NOTIFICATIONS] Sent scheduled update for {len(updates)} picks")
        
        return bool(alerts or updates or off_path_stocks or off_path_crypto)
    
    def get_status(self) -> Dict:
        """Get current status of the notification system"""
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
