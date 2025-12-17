#!/usr/bin/env python3
"""
Ghost Protocol - Trade Journal
================================

Track your actual trades vs Ghost's signals.

This helps you:
1. See which signals you actually executed
2. Track P&L for real trades
3. Compare your results vs Ghost's predictions
4. Learn from winning and losing trades

Usage:
    from core.trade_journal import get_trade_journal
    
    journal = get_trade_journal()
    
    # Log a trade entry
    trade_id = journal.log_entry(
        symbol="BTC",
        direction="SHORT",
        entry_price=87250.00,
        position_size=2500.00,  # $2500 position
        stop_loss=88800.00,
        take_profit=84500.00,
        cascade_id="370bf39d-...",  # Link to Ghost signal
        notes="6h final call - high confidence"
    )
    
    # Log trade exit
    journal.log_exit(
        trade_id=trade_id,
        exit_price=84800.00,
        exit_reason="TARGET_HIT"
    )
    
    # Get journal entries
    trades = journal.get_trades(limit=50)
    stats = journal.get_stats(days=30)
"""

import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("core.trade_journal")

# Database path
JOURNAL_DB_PATH = Path("./data/ghost_predictions.db")


class TradeJournal:
    """
    Track your actual trades and compare to Ghost's signals.
    """
    
    def __init__(self, db_path: Path | str | None = None):
        """
        Initialize trade journal.
        
        Args:
            db_path: Path to SQLite database (defaults to ghost_predictions.db)
        """
        self.db_path = Path(db_path) if db_path else JOURNAL_DB_PATH
        self._ensure_journal_table()
    
    def _ensure_journal_table(self):
        """Create trade_journal table if it doesn't exist"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_journal (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,  -- LONG, SHORT
                    
                    -- Entry details
                    entry_time INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    position_size REAL NOT NULL,  -- Dollar amount
                    stop_loss REAL,
                    take_profit REAL,
                    
                    -- Exit details
                    exit_time INTEGER,
                    exit_price REAL,
                    exit_reason TEXT,  -- TARGET_HIT, STOP_HIT, MANUAL, TIMEOUT
                    
                    -- P&L
                    profit_loss REAL,  -- Dollar P&L
                    profit_loss_pct REAL,  -- Percentage return
                    
                    -- Ghost signal link
                    cascade_id TEXT,  -- Link to prediction_cascades
                    prediction_id INTEGER,  -- Link to ghost_predictions
                    ghost_confidence REAL,
                    ghost_direction TEXT,
                    
                    -- User notes
                    entry_notes TEXT,
                    exit_notes TEXT,
                    tags TEXT,  -- Comma-separated: "scalp,reversal,6h_final"
                    
                    -- Metadata
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER
                )
            """)
            conn.commit()
            conn.close()
            
            LOGGER.info("✅ Trade journal table ready")
        
        except Exception as e:
            LOGGER.error(f"Failed to create trade journal table: {e}")
    
    def log_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        position_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        cascade_id: Optional[str] = None,
        prediction_id: Optional[int] = None,
        ghost_confidence: Optional[float] = None,
        ghost_direction: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> str:
        """
        Log trade entry.
        
        Args:
            symbol: Asset symbol (BTC, ETH, etc.)
            direction: LONG or SHORT
            entry_price: Entry price
            position_size: Position size in dollars
            stop_loss: Stop loss price
            take_profit: Take profit target
            cascade_id: Link to Ghost cascade
            prediction_id: Link to Ghost prediction
            ghost_confidence: Ghost's confidence at entry
            ghost_direction: Ghost's direction (UP/DOWN)
            notes: Entry notes
            tags: List of tags (e.g., ["scalp", "6h_final"])
        
        Returns:
            trade_id: Unique trade identifier
        """
        try:
            trade_id = str(uuid.uuid4())
            now = int(time.time())
            
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                INSERT INTO trade_journal (
                    trade_id, symbol, direction,
                    entry_time, entry_price, position_size,
                    stop_loss, take_profit,
                    cascade_id, prediction_id,
                    ghost_confidence, ghost_direction,
                    entry_notes, tags,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, symbol, direction.upper(),
                now, entry_price, position_size,
                stop_loss, take_profit,
                cascade_id, prediction_id,
                ghost_confidence, ghost_direction,
                notes, ",".join(tags) if tags else None,
                now, now
            ))
            conn.commit()
            conn.close()
            
            LOGGER.info(f"✅ Trade logged: {symbol} {direction} @ ${entry_price:,.2f} (${position_size:,.2f})")
            return trade_id
        
        except Exception as e:
            LOGGER.error(f"Failed to log trade entry: {e}", exc_info=True)
            raise
    
    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "MANUAL",
        notes: Optional[str] = None
    ):
        """
        Log trade exit and calculate P&L.
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: TARGET_HIT, STOP_HIT, MANUAL, TIMEOUT
            notes: Exit notes
        """
        try:
            now = int(time.time())
            
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            # Get trade details
            cursor = conn.execute("""
                SELECT * FROM trade_journal WHERE trade_id = ?
            """, (trade_id,))
            trade = cursor.fetchone()
            
            if not trade:
                raise ValueError(f"Trade {trade_id} not found")
            
            # Calculate P&L
            direction = trade['direction']
            entry_price = trade['entry_price']
            position_size = trade['position_size']
            
            if direction == 'LONG':
                price_change_pct = (exit_price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - exit_price) / entry_price
            
            profit_loss = position_size * price_change_pct
            profit_loss_pct = price_change_pct * 100
            
            # Update trade
            conn.execute("""
                UPDATE trade_journal
                SET exit_time = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    profit_loss = ?,
                    profit_loss_pct = ?,
                    exit_notes = ?,
                    updated_at = ?
                WHERE trade_id = ?
            """, (
                now, exit_price, exit_reason,
                profit_loss, profit_loss_pct,
                notes, now, trade_id
            ))
            conn.commit()
            conn.close()
            
            result_emoji = "✅" if profit_loss > 0 else "❌"
            LOGGER.info(
                f"{result_emoji} Trade closed: {trade['symbol']} "
                f"${profit_loss:+,.2f} ({profit_loss_pct:+.2f}%)"
            )
        
        except Exception as e:
            LOGGER.error(f"Failed to log trade exit: {e}", exc_info=True)
            raise
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 50,
        include_open: bool = True
    ) -> list[dict[str, Any]]:
        """
        Get trade journal entries.
        
        Args:
            symbol: Filter by symbol
            days: Show trades from last N days
            limit: Max number of trades to return
            include_open: Include open trades (not yet exited)
        
        Returns:
            List of trade dictionaries
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM trade_journal WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if days:
                cutoff = int(time.time() - (days * 86400))
                query += " AND entry_time >= ?"
                params.append(cutoff)
            
            if not include_open:
                query += " AND exit_time IS NOT NULL"
            
            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        
        except Exception as e:
            LOGGER.error(f"Failed to get trades: {e}")
            return []
    
    def get_stats(self, days: int = 30) -> dict[str, Any]:
        """
        Get trading statistics.
        
        Args:
            days: Calculate stats for last N days
        
        Returns:
            Statistics dictionary
        """
        try:
            cutoff = int(time.time() - (days * 86400))
            
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            # Get all closed trades
            cursor = conn.execute("""
                SELECT * FROM trade_journal
                WHERE entry_time >= ?
                AND exit_time IS NOT NULL
                ORDER BY entry_time DESC
            """, (cutoff,))
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return {
                    "total_trades": 0,
                    "win_rate": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "total_pnl": 0,
                    "total_pnl_pct": 0,
                    "best_trade": None,
                    "worst_trade": None
                }
            
            # Calculate stats
            total_trades = len(trades)
            winners = [t for t in trades if t['profit_loss'] > 0]
            losers = [t for t in trades if t['profit_loss'] < 0]
            
            total_pnl = sum(t['profit_loss'] for t in trades)
            avg_win = sum(t['profit_loss'] for t in winners) / len(winners) if winners else 0
            avg_loss = sum(t['profit_loss'] for t in losers) / len(losers) if losers else 0
            
            best_trade = max(trades, key=lambda t: t['profit_loss'])
            worst_trade = min(trades, key=lambda t: t['profit_loss'])
            
            return {
                "total_trades": total_trades,
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": len(winners) / total_trades if total_trades > 0 else 0,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "total_pnl": total_pnl,
                "total_pnl_pct": sum(t['profit_loss_pct'] for t in trades) / total_trades if total_trades > 0 else 0,
                "best_trade": dict(best_trade),
                "worst_trade": dict(worst_trade),
                "period_days": days
            }
        
        except Exception as e:
            LOGGER.error(f"Failed to calculate stats: {e}")
            return {}


# Singleton instance
_TRADE_JOURNAL = None


def get_trade_journal() -> TradeJournal:
    """Get singleton trade journal instance"""
    global _TRADE_JOURNAL
    if _TRADE_JOURNAL is None:
        _TRADE_JOURNAL = TradeJournal()
    return _TRADE_JOURNAL
