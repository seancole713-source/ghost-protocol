#!/usr/bin/env python3
"""
🎯 GHOST ACTIVE TRACKING SYSTEM - The Missing Loop

THIS IS THE FIX FOR: "Ghost sends ONE prediction then goes silent"

The Full Loop:
1. 5 AM: Daily TOP 10 scan → ONE consolidated message (5 stocks + 5 crypto)
2. Every 4 hours: Update check → ONE message if significant changes (>2%)
3. Instant: Alert when target OR stop hit
4. 48h: Final results message with WIN/LOSS/NEUTRAL breakdown
5. REPEAT with yesterday's picks still tracking (overlap)

NO MORE SPAM. Every notification MATTERS.

Author: Ghost System
"""

import os
import time
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading

LOGGER = logging.getLogger("ghost.active_tracking")

# ============================================================================
# CONFIGURATION (all can be overridden via env vars)
# ============================================================================

# When to send daily TOP 10 (5 AM local)
DAILY_TOP_10_HOUR = int(os.getenv("GHOST_TOP_10_HOUR", "5"))

# How often to check for updates (4 hours)
UPDATE_CHECK_INTERVAL_HOURS = int(os.getenv("GHOST_UPDATE_INTERVAL_HOURS", "4"))

# Prediction tracking duration
PREDICTION_HORIZON_HOURS = 48

# Significance thresholds for updates (only notify if MEANINGFUL change)
SIGNIFICANT_MOVE_PCT = float(os.getenv("GHOST_SIGNIFICANT_MOVE_PCT", "3.0"))  # 3%+ change since last update
NEAR_TARGET_PCT = float(os.getenv("GHOST_NEAR_TARGET_PCT", "1.0"))  # Within 1% of target
NEAR_STOP_PCT = float(os.getenv("GHOST_NEAR_STOP_PCT", "0.5"))  # Within 0.5% of stop - DANGER

# How many top picks per category
TOP_STOCKS_COUNT = int(os.getenv("GHOST_TOP_STOCKS", "5"))
TOP_CRYPTO_COUNT = int(os.getenv("GHOST_TOP_CRYPTO", "5"))

# Minimum confidence to make the TOP 10
MIN_TOP_10_CONFIDENCE = float(os.getenv("GHOST_TOP_10_MIN_CONF", "0.85"))

# Database path
ACTIVE_TRACKING_DB = os.getenv("GHOST_TRACKING_DB", "data/active_tracking.db")


class TrackingStatus(Enum):
    ACTIVE = "active"
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    EXPIRED = "expired"


class TrackingOutcome(Enum):
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    NEUTRAL = "neutral"


@dataclass
class ActivePick:
    """A prediction being actively tracked for 48 hours"""
    pick_id: int
    symbol: str
    asset_type: str  # 'crypto' or 'stock'
    direction: str   # 'UP' or 'DOWN'
    entry_price: float
    target_price: float
    stop_price: float
    confidence: float
    created_at: datetime
    expires_at: datetime
    batch_date: str  # YYYY-MM-DD format
    status: TrackingStatus = TrackingStatus.ACTIVE
    outcome: TrackingOutcome = TrackingOutcome.PENDING
    current_price: float = 0.0
    last_notified_price: float = 0.0
    last_notification_at: Optional[datetime] = None
    target_hit_at: Optional[datetime] = None
    stop_hit_at: Optional[datetime] = None
    final_price: Optional[float] = None
    reasons: str = ""
    
    @property
    def hours_remaining(self) -> float:
        """Hours until this prediction expires"""
        if self.status != TrackingStatus.ACTIVE:
            return 0.0
        remaining = (self.expires_at - datetime.utcnow()).total_seconds() / 3600
        return max(0, remaining)
    
    @property
    def pct_change(self) -> float:
        """Current % change from entry"""
        if self.entry_price <= 0:
            return 0.0
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def pct_to_target(self) -> float:
        """% distance to target (positive = still need to move, 0 = hit)"""
        if self.entry_price <= 0:
            return 0.0
        total_move = abs(self.target_price - self.entry_price)
        if total_move == 0:
            return 0.0
            
        if self.direction == "DOWN":
            if self.current_price <= self.target_price:
                return 0.0  # Already hit
            return ((self.current_price - self.target_price) / self.entry_price) * 100
        else:
            if self.current_price >= self.target_price:
                return 0.0  # Already hit
            return ((self.target_price - self.current_price) / self.entry_price) * 100
    
    @property
    def pct_to_stop(self) -> float:
        """% distance to stop (positive = still safe, 0 = hit)"""
        if self.entry_price <= 0:
            return float('inf')
            
        if self.direction == "DOWN":
            # For DOWN, stop is above entry
            if self.current_price >= self.stop_price:
                return 0.0  # Already hit
            return ((self.stop_price - self.current_price) / self.entry_price) * 100
        else:
            # For UP, stop is below entry
            if self.current_price <= self.stop_price:
                return 0.0  # Already hit
            return ((self.current_price - self.stop_price) / self.entry_price) * 100
    
    @property
    def is_on_track(self) -> bool:
        """Is the prediction moving in the right direction?"""
        if self.direction == "DOWN":
            return self.current_price < self.entry_price
        else:
            return self.current_price > self.entry_price
    
    def check_target_hit(self) -> bool:
        """Check if target price was hit"""
        if self.direction == "DOWN":
            return self.current_price <= self.target_price
        else:
            return self.current_price >= self.target_price
    
    def check_stop_hit(self) -> bool:
        """Check if stop price was hit"""
        if self.direction == "DOWN":
            return self.current_price >= self.stop_price
        else:
            return self.current_price <= self.stop_price


class ActiveTrackingSystem:
    """
    The brain of Ghost's continuous tracking loop.
    
    This is what was MISSING:
    - Before: Send prediction → disappear → user has no idea what happened
    - After: Send TOP 10 → Track 48h → Update on significant changes → 
             Alert on target/stop → Final results → REPEAT
    """
    
    def __init__(self, db_path: str = ACTIVE_TRACKING_DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        
        # In-memory cache of active picks
        self._active_picks: Dict[str, ActivePick] = {}
        self._load_active_from_db()
        
        # Track last daily TOP 10 date
        self._last_top_10_date: Optional[str] = None
        
        LOGGER.info(f"[ACTIVE TRACKING] Initialized with {len(self._active_picks)} active picks")
    
    def _init_db(self):
        """Create the tracking database tables"""
        os.makedirs(os.path.dirname(self.db_path) or "data", exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_picks (
                    pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    batch_date TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    outcome TEXT DEFAULT 'pending',
                    current_price REAL DEFAULT 0,
                    last_notified_price REAL DEFAULT 0,
                    last_notification_at TEXT,
                    target_hit_at TEXT,
                    stop_hit_at TEXT,
                    final_price REAL,
                    reasons TEXT,
                    UNIQUE(symbol, batch_date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_batches (
                    batch_date TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    stocks_count INTEGER DEFAULT 0,
                    crypto_count INTEGER DEFAULT 0,
                    total_wins INTEGER DEFAULT 0,
                    total_losses INTEGER DEFAULT 0,
                    total_neutral INTEGER DEFAULT 0,
                    top_10_sent INTEGER DEFAULT 0,
                    final_results_sent INTEGER DEFAULT 0,
                    message_id TEXT
                )
            """)
            
            # Create indexes for fast queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_picks_status 
                ON active_picks(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_picks_expires 
                ON active_picks(expires_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_active_picks_batch 
                ON active_picks(batch_date)
            """)
            
            conn.commit()
    
    def _load_active_from_db(self):
        """Load active picks from database into memory (MONEY GAME - all symbols compete)"""
        # MONEY GAME: No more V2 whitelist filtering - all symbols compete!
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM active_picks 
                WHERE status = 'active' 
                AND datetime(expires_at) > datetime('now')
            """).fetchall()
            
            loaded = 0
            
            for row in rows:
                symbol = row['symbol']
                pick = self._row_to_pick(row)
                self._active_picks[symbol] = pick
                loaded += 1
            
            LOGGER.info(f"[MONEY GAME] Loaded {loaded} active picks - all symbols compete!")
    
    def _row_to_pick(self, row) -> ActivePick:
        """Convert database row to ActivePick object"""
        return ActivePick(
            pick_id=row['pick_id'],
            symbol=row['symbol'],
            asset_type=row['asset_type'],
            direction=row['direction'],
            entry_price=row['entry_price'],
            target_price=row['target_price'],
            stop_price=row['stop_price'],
            confidence=row['confidence'],
            created_at=datetime.fromisoformat(row['created_at']),
            expires_at=datetime.fromisoformat(row['expires_at']),
            batch_date=row['batch_date'],
            status=TrackingStatus(row['status']),
            outcome=TrackingOutcome(row['outcome']),
            current_price=row['current_price'] or row['entry_price'],
            last_notified_price=row['last_notified_price'] or row['entry_price'],
            last_notification_at=datetime.fromisoformat(row['last_notification_at']) if row['last_notification_at'] else None,
            target_hit_at=datetime.fromisoformat(row['target_hit_at']) if row['target_hit_at'] else None,
            stop_hit_at=datetime.fromisoformat(row['stop_hit_at']) if row['stop_hit_at'] else None,
            final_price=row['final_price'],
            reasons=row['reasons'] or "",
        )
    
    def add_pick(self, pick: ActivePick) -> bool:
        """Add a pick to be tracked"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO active_picks 
                        (symbol, asset_type, direction, entry_price, target_price, 
                         stop_price, confidence, created_at, expires_at, batch_date,
                         status, outcome, current_price, last_notified_price, reasons)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pick.symbol, pick.asset_type, pick.direction,
                        pick.entry_price, pick.target_price, pick.stop_price, 
                        pick.confidence, pick.created_at.isoformat(), 
                        pick.expires_at.isoformat(), pick.batch_date,
                        pick.status.value, pick.outcome.value, pick.current_price,
                        pick.last_notified_price, pick.reasons
                    ))
                    # Get the ID that was assigned
                    if pick.pick_id == 0:
                        pick.pick_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.commit()
                
                self._active_picks[pick.symbol] = pick
                LOGGER.info(f"[ACTIVE TRACKING] Added {pick.symbol} ({pick.asset_type}) to tracking until {pick.expires_at}")
                return True
                
            except Exception as e:
                LOGGER.error(f"[ACTIVE TRACKING] Failed to add {pick.symbol}: {e}")
                return False
    
    def update_price(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Update price for a tracked pick.
        Returns event dict if something SIGNIFICANT happened (target hit, stop hit, or expiry).
        """
        with self._lock:
            pick = self._active_picks.get(symbol)
            if not pick or pick.status != TrackingStatus.ACTIVE:
                return None
            
            old_price = pick.current_price
            pick.current_price = current_price
            event = None
            
            # Check if target hit - INSTANT ALERT
            if pick.check_target_hit():
                pick.status = TrackingStatus.TARGET_HIT
                pick.outcome = TrackingOutcome.WIN
                pick.target_hit_at = datetime.utcnow()
                pick.final_price = current_price
                event = {
                    "type": "TARGET_HIT",
                    "symbol": symbol,
                    "pick": pick,
                    "pct_change": pick.pct_change,
                    "priority": "HIGH",
                }
                LOGGER.info(f"[ACTIVE TRACKING] 🎯 TARGET HIT: {symbol} at {current_price} ({pick.pct_change:+.1f}%)")
            
            # Check if stop hit - INSTANT ALERT
            elif pick.check_stop_hit():
                pick.status = TrackingStatus.STOP_HIT
                pick.outcome = TrackingOutcome.LOSS
                pick.stop_hit_at = datetime.utcnow()
                pick.final_price = current_price
                event = {
                    "type": "STOP_HIT",
                    "symbol": symbol,
                    "pick": pick,
                    "pct_change": pick.pct_change,
                    "priority": "HIGH",
                }
                LOGGER.info(f"[ACTIVE TRACKING] 🛑 STOP HIT: {symbol} at {current_price} ({pick.pct_change:+.1f}%)")
            
            # Check if expired (48h up)
            elif pick.hours_remaining <= 0:
                pick.status = TrackingStatus.EXPIRED
                # Determine outcome based on final position
                if pick.is_on_track:
                    pick.outcome = TrackingOutcome.WIN  # Moved in right direction but didn't hit target
                else:
                    pick.outcome = TrackingOutcome.NEUTRAL  # Didn't move much or wrong direction
                pick.final_price = current_price
                event = {
                    "type": "EXPIRED",
                    "symbol": symbol,
                    "pick": pick,
                    "pct_change": pick.pct_change,
                    "priority": "NORMAL",
                }
                LOGGER.info(f"[ACTIVE TRACKING] ⏱️ EXPIRED: {symbol} at {current_price} ({pick.pct_change:+.1f}%)")
            
            # Save to DB
            self._save_pick(pick)
            
            # Remove from active cache if no longer active
            if pick.status != TrackingStatus.ACTIVE:
                del self._active_picks[symbol]
            
            return event
    
    def _save_pick(self, pick: ActivePick):
        """Save pick state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE active_picks SET
                        status = ?,
                        outcome = ?,
                        current_price = ?,
                        last_notified_price = ?,
                        last_notification_at = ?,
                        target_hit_at = ?,
                        stop_hit_at = ?,
                        final_price = ?
                    WHERE pick_id = ?
                """, (
                    pick.status.value, pick.outcome.value, pick.current_price,
                    pick.last_notified_price,
                    pick.last_notification_at.isoformat() if pick.last_notification_at else None,
                    pick.target_hit_at.isoformat() if pick.target_hit_at else None,
                    pick.stop_hit_at.isoformat() if pick.stop_hit_at else None,
                    pick.final_price,
                    pick.pick_id
                ))
                conn.commit()
        except Exception as e:
            LOGGER.error(f"[ACTIVE TRACKING] Failed to save {pick.symbol}: {e}")
    
    def get_active_picks(self) -> List[ActivePick]:
        """Get all currently active picks"""
        with self._lock:
            return list(self._active_picks.values())
    
    def get_picks_needing_update(self) -> List[ActivePick]:
        """
        Get picks that have SIGNIFICANT changes worth notifying about.
        Only returns picks where something MEANINGFUL happened.
        
        Criteria:
        1. Near target (within NEAR_TARGET_PCT) - Good news!
        2. Near stop (within NEAR_STOP_PCT) - Warning!
        3. Large move since last notification (>SIGNIFICANT_MOVE_PCT)
        """
        needs_update = []
        
        for pick in self.get_active_picks():
            # Don't spam - minimum 2 hours between updates for same pick
            if pick.last_notification_at:
                hours_since = (datetime.utcnow() - pick.last_notification_at).total_seconds() / 3600
                if hours_since < 2:
                    continue
            
            # Priority 1: Near target (almost WIN!)
            if pick.is_on_track and pick.pct_to_target <= NEAR_TARGET_PCT:
                needs_update.append(pick)
                continue
            
            # Priority 2: Near stop (DANGER - about to LOSE)
            if not pick.is_on_track and pick.pct_to_stop <= NEAR_STOP_PCT:
                needs_update.append(pick)
                continue
            
            # Priority 3: Significant price change since last notification
            if pick.last_notified_price > 0:
                pct_since_notify = abs((pick.current_price - pick.last_notified_price) / pick.last_notified_price) * 100
                if pct_since_notify >= SIGNIFICANT_MOVE_PCT:
                    needs_update.append(pick)
                    continue
        
        return needs_update
    
    def mark_notified(self, picks: List[ActivePick]):
        """Mark picks as notified to prevent spam"""
        for pick in picks:
            pick.last_notification_at = datetime.utcnow()
            pick.last_notified_price = pick.current_price
            self._save_pick(pick)
    
    def get_batch_results(self, batch_date: str) -> List[ActivePick]:
        """Get all picks from a specific batch date (for final results)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM active_picks 
                WHERE batch_date = ?
            """, (batch_date,)).fetchall()
            
            return [self._row_to_pick(row) for row in rows]
    
    def get_expired_batches_needing_results(self) -> List[str]:
        """Get batch dates where all picks have expired but results not yet sent"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Find batches that are 48h old and haven't had results sent
            rows = conn.execute("""
                SELECT DISTINCT ap.batch_date 
                FROM active_picks ap
                LEFT JOIN daily_batches db ON ap.batch_date = db.batch_date
                WHERE datetime(ap.expires_at) <= datetime('now')
                AND (db.final_results_sent IS NULL OR db.final_results_sent = 0)
                AND ap.batch_date NOT IN (
                    SELECT batch_date FROM daily_batches WHERE final_results_sent = 1
                )
            """).fetchall()
            
            return [row['batch_date'] for row in rows]
    
    def mark_results_sent(self, batch_date: str, wins: int, losses: int, neutral: int):
        """Mark that final results have been sent for a batch"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_batches 
                    (batch_date, created_at, total_wins, total_losses, total_neutral, final_results_sent)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (batch_date, datetime.utcnow().isoformat(), wins, losses, neutral))
                conn.commit()
        except Exception as e:
            LOGGER.error(f"[ACTIVE TRACKING] Failed to mark results sent for {batch_date}: {e}")
    
    def has_sent_top_10_today(self) -> bool:
        """Check if we've already sent TOP 10 today"""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT top_10_sent FROM daily_batches WHERE batch_date = ?
            """, (today,)).fetchone()
            return result is not None and result[0] == 1
    
    def mark_top_10_sent(self, batch_date: str, stocks_count: int, crypto_count: int):
        """Mark that TOP 10 has been sent for today"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_batches 
                    (batch_date, created_at, stocks_count, crypto_count, top_10_sent)
                    VALUES (?, ?, ?, ?, 1)
                """, (batch_date, datetime.utcnow().isoformat(), stocks_count, crypto_count))
                conn.commit()
        except Exception as e:
            LOGGER.error(f"[ACTIVE TRACKING] Failed to mark TOP 10 sent: {e}")
    
    def get_running_stats(self) -> Tuple[int, int, int]:
        """Get running win/loss/neutral counts (all time)"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT 
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome = 'neutral' THEN 1 ELSE 0 END) as neutral
                FROM active_picks
                WHERE status != 'active'
            """).fetchone()
            
            return (result[0] or 0, result[1] or 0, result[2] or 0)


# ============================================================================
# MESSAGE FORMATTERS - ONE message per event type, NO SPAM
# ============================================================================

def format_daily_top_10(
    stocks: List[ActivePick],
    crypto: List[ActivePick],
    date_str: str,
    inverse_mode: bool = None
) -> str:
    """
    Format the daily TOP 10 as ONE consolidated message.
    This is the ONLY message sent at 5 AM.
    """
    # Read from env var if not specified (default OFF)
    if inverse_mode is None:
        inverse_mode = os.getenv("INVERSE_GHOST", "0") == "1"
    direction_label = "INVERSE GHOST" if inverse_mode else "GHOST"
    
    lines = [
        f"🔮 **{direction_label} TOP 10 — {date_str}**",
        "",
    ]
    
    if stocks:
        lines.append("📈 **STOCKS:**")
        for i, p in enumerate(stocks, 1):
            emoji = "🔴" if p.direction == "DOWN" else "🟢"
            target_pct = ((p.target_price - p.entry_price) / p.entry_price) * 100
            lines.append(
                f"{i}. **{p.symbol}** {emoji} {p.direction} | "
                f"${p.entry_price:,.2f} → ${p.target_price:,.2f} ({target_pct:+.1f}%) | "
                f"{p.confidence*100:.0f}%"
            )
        lines.append("")
    
    if crypto:
        lines.append("🪙 **CRYPTO:**")
        for i, p in enumerate(crypto, 1):
            emoji = "🔴" if p.direction == "DOWN" else "🟢"
            target_pct = ((p.target_price - p.entry_price) / p.entry_price) * 100
            
            # Smart price formatting
            if p.entry_price >= 1000:
                entry_fmt = f"${p.entry_price:,.0f}"
                target_fmt = f"${p.target_price:,.0f}"
            elif p.entry_price >= 1:
                entry_fmt = f"${p.entry_price:,.2f}"
                target_fmt = f"${p.target_price:,.2f}"
            elif p.entry_price >= 0.01:
                entry_fmt = f"${p.entry_price:.4f}"
                target_fmt = f"${p.target_price:.4f}"
            else:
                entry_fmt = f"${p.entry_price:.8f}"
                target_fmt = f"${p.target_price:.8f}"
            
            lines.append(
                f"{i}. **{p.symbol}** {emoji} {p.direction} | "
                f"{entry_fmt} → {target_fmt} ({target_pct:+.1f}%) | "
                f"{p.confidence*100:.0f}%"
            )
        lines.append("")
    
    lines.extend([
        "⏱️ **48-Hour Tracking Started**",
        "📊 Updates ONLY if significant changes (>3%)",
        "🎯 Instant alerts on target/stop hit",
        "",
        "_Ghost is watching. You'll hear from us._"
    ])
    
    return "\n".join(lines)


def format_significant_update(
    attention_needed: List[ActivePick],
    all_active: List[ActivePick]
) -> Optional[str]:
    """
    Format update message ONLY if something SIGNIFICANT happened.
    Returns None if nothing worth reporting.
    """
    if not attention_needed:
        return None
    
    lines = [
        "📊 **GHOST UPDATE**",
        "",
    ]
    
    # Separate by urgency
    near_target = [p for p in attention_needed if p.is_on_track and p.pct_to_target <= NEAR_TARGET_PCT]
    near_stop = [p for p in attention_needed if not p.is_on_track and p.pct_to_stop <= NEAR_STOP_PCT]
    big_moves = [p for p in attention_needed if p not in near_target and p not in near_stop]
    
    if near_target:
        lines.append("🎯 **ALMOST AT TARGET:**")
        for p in near_target:
            pct_away = p.pct_to_target
            lines.append(f"• **{p.symbol}** only {pct_away:.1f}% away from target! ({p.pct_change:+.1f}% so far)")
        lines.append("")
    
    if near_stop:
        lines.append("⚠️ **WARNING - APPROACHING STOP:**")
        for p in near_stop:
            pct_away = p.pct_to_stop
            lines.append(f"• **{p.symbol}** only {pct_away:.1f}% from stop ({p.pct_change:+.1f}%)")
        lines.append("")
    
    if big_moves:
        lines.append("📈 **SIGNIFICANT MOVES:**")
        for p in big_moves:
            direction_emoji = "✅" if p.is_on_track else "⚠️"
            lines.append(f"• {direction_emoji} **{p.symbol}** {p.pct_change:+.1f}%")
        lines.append("")
    
    # Count others with no significant change
    others = len(all_active) - len(attention_needed)
    if others > 0:
        lines.append(f"_Other {others} picks: No significant change_")
        lines.append("")
    
    # Average time remaining
    if all_active:
        avg_hours = sum(p.hours_remaining for p in all_active) / len(all_active)
        lines.append(f"⏱️ Avg time remaining: {avg_hours:.0f}h")
    
    return "\n".join(lines)


def format_target_hit(pick: ActivePick) -> str:
    """Format instant alert when target is hit - WIN!"""
    # Smart price formatting
    if pick.entry_price >= 1000:
        entry_fmt = f"${pick.entry_price:,.0f}"
        target_fmt = f"${pick.target_price:,.0f}"
    elif pick.entry_price >= 1:
        entry_fmt = f"${pick.entry_price:,.2f}"
        target_fmt = f"${pick.target_price:,.2f}"
    elif pick.entry_price >= 0.01:
        entry_fmt = f"${pick.entry_price:.4f}"
        target_fmt = f"${pick.target_price:.4f}"
    else:
        entry_fmt = f"${pick.entry_price:.8f}"
        target_fmt = f"${pick.target_price:.8f}"
    
    hours_taken = PREDICTION_HORIZON_HOURS - pick.hours_remaining
    hours_early = pick.hours_remaining
    
    return "\n".join([
        "🎯 **TARGET HIT!**",
        "",
        f"**{pick.symbol}** reached target!",
        "",
        f"Entry: {entry_fmt}",
        f"Target: {target_fmt}",
        f"Change: **{pick.pct_change:+.1f}%**",
        "",
        f"✅ **RESULT: WIN**",
        f"⏱️ Time: {hours_taken:.0f}h ({hours_early:.0f}h early!)",
    ])


def format_stop_hit(pick: ActivePick) -> str:
    """Format instant alert when stop is hit - LOSS"""
    # Smart price formatting
    if pick.entry_price >= 1000:
        entry_fmt = f"${pick.entry_price:,.0f}"
        stop_fmt = f"${pick.stop_price:,.0f}"
        current_fmt = f"${pick.current_price:,.0f}"
    elif pick.entry_price >= 1:
        entry_fmt = f"${pick.entry_price:,.2f}"
        stop_fmt = f"${pick.stop_price:,.2f}"
        current_fmt = f"${pick.current_price:,.2f}"
    elif pick.entry_price >= 0.01:
        entry_fmt = f"${pick.entry_price:.4f}"
        stop_fmt = f"${pick.stop_price:.4f}"
        current_fmt = f"${pick.current_price:.4f}"
    else:
        entry_fmt = f"${pick.entry_price:.8f}"
        stop_fmt = f"${pick.stop_price:.8f}"
        current_fmt = f"${pick.current_price:.8f}"
    
    return "\n".join([
        "🛑 **STOP HIT**",
        "",
        f"**{pick.symbol}** hit stop loss",
        "",
        f"Entry: {entry_fmt}",
        f"Stop: {stop_fmt}",
        f"Final: {current_fmt}",
        f"Change: **{pick.pct_change:+.1f}%**",
        "",
        "❌ **RESULT: LOSS**",
        "_Position closed to limit losses_",
    ])


def format_final_results(
    picks: List[ActivePick],
    batch_date: str,
    running_wins: int,
    running_losses: int,
    running_neutral: int
) -> str:
    """Format the final results message when 48h completes"""
    wins = [p for p in picks if p.outcome == TrackingOutcome.WIN]
    losses = [p for p in picks if p.outcome == TrackingOutcome.LOSS]
    neutral = [p for p in picks if p.outcome == TrackingOutcome.NEUTRAL]
    
    total = len(picks)
    win_count = len(wins)
    loss_count = len(losses)
    
    # Calculate win rate (wins / (wins + losses), excluding neutral)
    decisive = win_count + loss_count
    win_pct = (win_count / decisive * 100) if decisive > 0 else 0
    
    # Grade based on win rate
    if win_pct >= 70:
        grade = "🔥 EXCELLENT"
    elif win_pct >= 60:
        grade = "✅ GOOD"
    elif win_pct >= 50:
        grade = "➖ AVERAGE"
    else:
        grade = "❌ POOR"
    
    lines = [
        f"📊 **GHOST RESULTS — {batch_date}**",
        "",
        f"**FINAL: {win_count}W / {loss_count}L ({win_pct:.0f}% win rate)** {grade}",
        "",
    ]
    
    if wins:
        lines.append("✅ **WINNERS:**")
        for p in wins:
            status = "🎯 Target" if p.status == TrackingStatus.TARGET_HIT else "📈 On Track"
            lines.append(f"• {p.symbol} {p.pct_change:+.1f}% ({status})")
        lines.append("")
    
    if losses:
        lines.append("❌ **LOSERS:**")
        for p in losses:
            lines.append(f"• {p.symbol} {p.pct_change:+.1f}% (Stopped)")
        lines.append("")
    
    if neutral:
        lines.append("➖ **NEUTRAL:**")
        for p in neutral:
            lines.append(f"• {p.symbol} {p.pct_change:+.1f}%")
        lines.append("")
    
    # Running accuracy (all time)
    total_running = running_wins + running_losses
    if total_running > 0:
        running_pct = (running_wins / total_running) * 100
        lines.append(f"📈 **All-Time: {running_wins}W/{running_losses}L ({running_pct:.0f}%)**")
    
    lines.append("")
    lines.append("_Next Top 10 at 5 AM_")
    
    return "\n".join(lines)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_tracker: Optional[ActiveTrackingSystem] = None
_tracker_lock = threading.Lock()


def get_active_tracker() -> ActiveTrackingSystem:
    """Get or create the singleton tracker instance"""
    global _tracker
    with _tracker_lock:
        if _tracker is None:
            _tracker = ActiveTrackingSystem()
        return _tracker


# ============================================================================
# HIGH-LEVEL ORCHESTRATION FUNCTIONS
# ============================================================================

# DISABLED - Use ghost_notifications.py instead.
# This function had WRONG color logic: used p.direction instead of comparing prices
async def send_daily_top_10(
    get_high_conf_predictions: Callable,
    send_telegram: Callable,
    inverse_mode: bool = None
) -> bool:
    """
    DISABLED - This function had wrong color logic.
    
    The bug was:
        emoji = "🔴" if p.direction == "DOWN" else "🟢"
    
    Should be:
        emoji = "🔴" if target_price < entry_price else "🟢"
    
    Use ghost_notifications.py instead which has correct logic.
    """
    LOGGER.warning("[ACTIVE TRACKING] send_daily_top_10 DISABLED - use ghost_notifications.py")
    return False


async def _send_daily_top_10_ORIGINAL_DISABLED(
    get_high_conf_predictions: Callable,
    send_telegram: Callable,
    inverse_mode: bool = None
) -> bool:
    """
    OLD CODE - DISABLED.
    
    Send the daily TOP 10 picks (5 stocks + 5 crypto).
    
    Args:
        get_high_conf_predictions: async func() -> List[dict] with keys:
            symbol, asset_type, direction, entry_price, target_price, 
            stop_price, confidence, reasons
        send_telegram: func(message: str) -> bool
        inverse_mode: Whether Ghost is in inverse mode
    
    Returns:
        True if sent successfully
    """
    tracker = get_active_tracker()
    
    # Check if we already sent today
    if tracker.has_sent_top_10_today():
        LOGGER.info("[ACTIVE TRACKING] TOP 10 already sent today")
        return False
    
    today_str = datetime.utcnow().strftime("%B %d, %Y")
    batch_date = datetime.utcnow().strftime("%Y-%m-%d")
    
    LOGGER.info(f"[ACTIVE TRACKING] 🌅 Starting daily TOP 10 scan for {today_str}")
    
    try:
        # Get all high-confidence predictions
        all_preds = await get_high_conf_predictions()
        
        if not all_preds:
            LOGGER.warning("[ACTIVE TRACKING] No predictions available")
            return False
        
        # Filter for MIN_TOP_10_CONFIDENCE
        high_conf = [p for p in all_preds if p.get('confidence', 0) >= MIN_TOP_10_CONFIDENCE]
        
        # Separate stocks and crypto
        stocks = [p for p in high_conf if p.get('asset_type') == 'stock']
        crypto = [p for p in high_conf if p.get('asset_type') == 'crypto']
        
        # Sort by confidence (best first)
        stocks.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        crypto.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Take TOP N
        top_stocks = stocks[:TOP_STOCKS_COUNT]
        top_crypto = crypto[:TOP_CRYPTO_COUNT]
        
        if not top_stocks and not top_crypto:
            LOGGER.warning(f"[ACTIVE TRACKING] No {MIN_TOP_10_CONFIDENCE*100:.0f}%+ confidence predictions")
            return False
        
        # Create ActivePick objects and add to tracking
        now = datetime.utcnow()
        expires = now + timedelta(hours=PREDICTION_HORIZON_HOURS)
        
        stock_picks = []
        for p in top_stocks:
            pick = ActivePick(
                pick_id=0,  # Will be assigned by DB
                symbol=p['symbol'],
                asset_type='stock',
                direction=p.get('direction', 'DOWN'),
                entry_price=p.get('entry_price', p.get('price', 0)),
                target_price=p.get('target_price', 0),
                stop_price=p.get('stop_price', p.get('stop_loss', 0)),
                confidence=p.get('confidence', 0),
                created_at=now,
                expires_at=expires,
                batch_date=batch_date,
                current_price=p.get('entry_price', p.get('price', 0)),
                last_notified_price=p.get('entry_price', p.get('price', 0)),
                reasons=str(p.get('reasons', '')),
            )
            tracker.add_pick(pick)
            stock_picks.append(pick)
        
        crypto_picks = []
        for p in top_crypto:
            pick = ActivePick(
                pick_id=0,
                symbol=p['symbol'],
                asset_type='crypto',
                direction=p.get('direction', 'DOWN'),
                entry_price=p.get('entry_price', p.get('price', 0)),
                target_price=p.get('target_price', 0),
                stop_price=p.get('stop_price', p.get('stop_loss', 0)),
                confidence=p.get('confidence', 0),
                created_at=now,
                expires_at=expires,
                batch_date=batch_date,
                current_price=p.get('entry_price', p.get('price', 0)),
                last_notified_price=p.get('entry_price', p.get('price', 0)),
                reasons=str(p.get('reasons', '')),
            )
            tracker.add_pick(pick)
            crypto_picks.append(pick)
        
        # Format and send ONE message
        message = format_daily_top_10(stock_picks, crypto_picks, today_str, inverse_mode)
        success = send_telegram(message)
        
        if success:
            tracker.mark_top_10_sent(batch_date, len(stock_picks), len(crypto_picks))
            LOGGER.info(f"[ACTIVE TRACKING] ✅ Sent TOP 10: {len(stock_picks)} stocks, {len(crypto_picks)} crypto")
        else:
            LOGGER.error("[ACTIVE TRACKING] ❌ Failed to send TOP 10")
        
        return success
        
    except Exception as e:
        LOGGER.error(f"[ACTIVE TRACKING] Daily TOP 10 failed: {e}", exc_info=True)
        return False


async def check_and_update_prices(
    get_price: Callable,
    send_telegram: Callable
) -> Dict[str, int]:
    """
    Check all active picks for price changes.
    Send instant alerts for target/stop hits.
    Send consolidated update if significant changes.
    
    Args:
        get_price: async func(symbol: str) -> float
        send_telegram: func(message: str) -> bool
    
    Returns:
        Dict with counts: {"target_hits": N, "stop_hits": N, "updates_sent": N}
    """
    tracker = get_active_tracker()
    results = {"target_hits": 0, "stop_hits": 0, "updates_sent": 0}
    
    active = tracker.get_active_picks()
    if not active:
        return results
    
    LOGGER.info(f"[ACTIVE TRACKING] Checking {len(active)} active picks")
    
    instant_alerts = []
    
    # Update all prices and collect events
    for pick in active:
        try:
            price = await get_price(pick.symbol)
            if price and price > 0:
                event = tracker.update_price(pick.symbol, price)
                
                if event:
                    if event['type'] == 'TARGET_HIT':
                        results["target_hits"] += 1
                        instant_alerts.append(('target', event['pick']))
                    elif event['type'] == 'STOP_HIT':
                        results["stop_hits"] += 1
                        instant_alerts.append(('stop', event['pick']))
                        
        except Exception as e:
            LOGGER.warning(f"[ACTIVE TRACKING] Failed to get price for {pick.symbol}: {e}")
    
    # Send instant alerts for target/stop hits
    for alert_type, pick in instant_alerts:
        if alert_type == 'target':
            msg = format_target_hit(pick)
        else:
            msg = format_stop_hit(pick)
        send_telegram(msg)
    
    # Check if any remaining active picks need significant update
    remaining_active = tracker.get_active_picks()
    needs_update = tracker.get_picks_needing_update()
    
    if needs_update:
        msg = format_significant_update(needs_update, remaining_active)
        if msg:
            send_telegram(msg)
            tracker.mark_notified(needs_update)
            results["updates_sent"] = 1
    
    LOGGER.info(f"[ACTIVE TRACKING] Update check complete: {results}")
    return results


async def check_and_send_final_results(send_telegram: Callable) -> int:
    """
    Check for expired batches and send final results.
    
    Returns:
        Number of final results messages sent
    """
    tracker = get_active_tracker()
    results_sent = 0
    
    expired_batches = tracker.get_expired_batches_needing_results()
    
    for batch_date in expired_batches:
        picks = tracker.get_batch_results(batch_date)
        
        if not picks:
            continue
        
        # Check if ALL picks in batch are resolved
        still_active = [p for p in picks if p.status == TrackingStatus.ACTIVE]
        if still_active:
            continue  # Batch not fully expired yet
        
        # Get running stats
        wins, losses, neutral = tracker.get_running_stats()
        
        # Format and send
        msg = format_final_results(picks, batch_date, wins, losses, neutral)
        success = send_telegram(msg)
        
        if success:
            batch_wins = len([p for p in picks if p.outcome == TrackingOutcome.WIN])
            batch_losses = len([p for p in picks if p.outcome == TrackingOutcome.LOSS])
            batch_neutral = len([p for p in picks if p.outcome == TrackingOutcome.NEUTRAL])
            tracker.mark_results_sent(batch_date, batch_wins, batch_losses, batch_neutral)
            results_sent += 1
            LOGGER.info(f"[ACTIVE TRACKING] ✅ Sent final results for {batch_date}")
    
    return results_sent


# ============================================================================
# MAIN SCHEDULER LOOP (to be started as background task in wolf_app)
# ============================================================================

async def active_tracking_scheduler(
    get_predictions: Callable,
    get_price: Callable,
    send_telegram: Callable,
    inverse_mode: bool = None,
    check_interval_minutes: int = 5
):
    """
    Main scheduler loop for active tracking.
    
    This runs continuously and handles:
    1. Daily TOP 10 at 5 AM
    2. Price updates every 5 minutes (but only notifies on significant changes)
    3. Final results when 48h expires
    
    Args:
        get_predictions: async func() -> List[dict]
        get_price: async func(symbol) -> float
        send_telegram: func(message) -> bool
        inverse_mode: Whether Ghost is in inverse mode
        check_interval_minutes: How often to check prices (default 5)
    """
    LOGGER.info("[ACTIVE TRACKING] 🚀 Starting Active Tracking Scheduler")
    
    last_top_10_date = None
    last_update_check = datetime.utcnow()
    
    while True:
        try:
            now = datetime.utcnow()
            current_hour = now.hour
            current_date = now.strftime("%Y-%m-%d")
            
            # ========================================
            # TASK 1: Daily TOP 10 at 5 AM
            # NOTE: TOP 10 is now handled by ghost_notifications.py
            #       This scheduler only handles price tracking updates
            # ========================================
            # DISABLED - send_daily_top_10 had wrong color logic
            # Now using ghost_notifications.py instead
            # if current_hour == DAILY_TOP_10_HOUR and last_top_10_date != current_date:
            #     LOGGER.info("[ACTIVE TRACKING] 🌅 Time for daily TOP 10!")
            #     await send_daily_top_10(get_predictions, send_telegram, inverse_mode)
            #     last_top_10_date = current_date
            
            # ========================================
            # TASK 2: Price updates every check_interval_minutes
            # ========================================
            minutes_since_update = (now - last_update_check).total_seconds() / 60
            if minutes_since_update >= check_interval_minutes:
                await check_and_update_prices(get_price, send_telegram)
                last_update_check = now
            
            # ========================================
            # TASK 3: Check for 48h expirations (every hour)
            # ========================================
            if now.minute < 5:  # First 5 minutes of each hour
                await check_and_send_final_results(send_telegram)
            
            # Sleep before next check
            await asyncio.sleep(60)  # Check every minute, but only act when needed
            
        except asyncio.CancelledError:
            LOGGER.info("[ACTIVE TRACKING] Scheduler cancelled")
            break
        except Exception as e:
            LOGGER.error(f"[ACTIVE TRACKING] Scheduler error: {e}", exc_info=True)
            await asyncio.sleep(60)


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    print("🎯 Ghost Active Tracking System")
    print("=" * 50)
    
    tracker = get_active_tracker()
    
    # Show current state
    active = tracker.get_active_picks()
    print(f"\nActive Picks: {len(active)}")
    for pick in active:
        print(f"  • {pick.symbol} ({pick.asset_type}): {pick.direction} @ ${pick.entry_price:.2f}")
        print(f"    Target: ${pick.target_price:.2f} | Stop: ${pick.stop_price:.2f}")
        print(f"    Hours remaining: {pick.hours_remaining:.1f}")
    
    # Show running stats
    wins, losses, neutral = tracker.get_running_stats()
    total = wins + losses
    if total > 0:
        print(f"\nRunning Stats: {wins}W / {losses}L ({wins/total*100:.0f}% win rate)")
    else:
        print("\nNo completed predictions yet")
