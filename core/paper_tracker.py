"""
Paper Trading Tracker - Automatically track EVERY Ghost signal

This is different from trade_journal.py:
- trade_journal = YOU manually log trades YOU execute
- paper_tracker = AUTO-logs EVERY Ghost signal, tracks what WOULD happen

Purpose: Prove Ghost's accuracy with 30+ days of tracked signals.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

LOGGER = logging.getLogger("paper_tracker")

# PostgreSQL support for production
DATABASE_URL = os.getenv("DATABASE_URL")


class PaperTracker:
    """
    Automatically track all Ghost signals and their outcomes.
    
    Supports both PostgreSQL (production) and SQLite (local dev).
    
    Workflow:
    1. When cascade hits 6h (final call), auto-log as paper trade
    2. Wait 48h from original entry
    3. Check actual price movement
    4. Calculate hypothetical P&L
    5. Track accuracy stats
    """
    
    def __init__(self, db_path: str = "data/ghost_predictions.db"):
        self.db_path = db_path
        self.use_postgres = bool(DATABASE_URL)
        self._ensure_table()
    
    def _get_postgres_connection(self):
        """Get PostgreSQL connection"""
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    
    def _get_connection(self):
        """Get database connection (PostgreSQL or SQLite)"""
        if self.use_postgres:
            return self._get_postgres_connection()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
    
    def _execute(self, conn, query: str, params: tuple = ()):
        """Execute query with proper placeholder substitution for PostgreSQL"""
        if self.use_postgres:
            # PostgreSQL uses %s placeholders
            query = query.replace("?", "%s")
            cur = conn.cursor()
            cur.execute(query, params)
            return cur
        else:
            return conn.execute(query, params)
    
    def _fetchone(self, cur):
        """Fetch one row as dict"""
        if self.use_postgres:
            row = cur.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))
        else:
            row = cur.fetchone()
            return dict(row) if row else None
    
    def _fetchall(self, cur) -> list:
        """Fetch all rows as list of dicts"""
        if self.use_postgres:
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in rows]
        else:
            return [dict(row) for row in cur.fetchall()]
    
    def _ensure_table(self):
        """Create paper_trades table if not exists"""
        try:
            if self.use_postgres:
                conn = self._get_postgres_connection()
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trades (
                        paper_trade_id TEXT PRIMARY KEY,
                        cascade_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_direction TEXT NOT NULL,
                        signal_confidence REAL NOT NULL,
                        signal_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        target_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        target_price REAL,
                        position_size REAL DEFAULT 1000.0,
                        stop_loss_pct REAL DEFAULT 0.05,
                        take_profit_pct REAL DEFAULT 0.10,
                        actual_direction TEXT,
                        outcome TEXT,
                        profit_loss REAL,
                        profit_loss_pct REAL,
                        checked_at TIMESTAMP WITH TIME ZONE,
                        notes TEXT,
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_outcome ON paper_trades(outcome)")
                conn.commit()
                conn.close()
                LOGGER.info("✅ paper_trades table ready (PostgreSQL)")
            else:
                conn = sqlite3.connect(self.db_path)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trades (
                        paper_trade_id TEXT PRIMARY KEY,
                        cascade_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_direction TEXT NOT NULL,
                        signal_confidence REAL NOT NULL,
                        signal_time TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        entry_time TEXT NOT NULL,
                        target_time TEXT NOT NULL,
                        target_price REAL,
                        position_size REAL DEFAULT 1000.0,
                        stop_loss_pct REAL DEFAULT 0.05,
                        take_profit_pct REAL DEFAULT 0.10,
                        actual_direction TEXT,
                        outcome TEXT,
                        profit_loss REAL,
                        profit_loss_pct REAL,
                        checked_at TEXT,
                        notes TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_outcome ON paper_trades(outcome)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_target_time ON paper_trades(target_time)")
                conn.commit()
                conn.close()
                LOGGER.info("✅ paper_trades table ready (SQLite)")
        except Exception as e:
            LOGGER.error(f"Failed to create paper_trades table: {e}")
            # Don't raise - table might already exist
    
    def log_signal(
        self,
        cascade_id: str,
        symbol: str,
        signal_direction: str,
        signal_confidence: float,
        entry_price: float,
        entry_time: str,
        position_size: float = 1000.0,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> str:
        """
        Log a Ghost signal as a paper trade.
        
        Called when cascade reaches 6h final call.
        Supports both PostgreSQL (production) and SQLite (local).
        """
        # =====================================================================
        # TRADING CONTROLS: Check blacklist/whitelist BEFORE logging trade
        # This prevents paper trades on assets with 0% historical win rate
        # =====================================================================
        try:
            from core.trading_controls import should_trade
            
            can_trade, reason = should_trade(symbol, signal_confidence)
            if not can_trade:
                LOGGER.info(
                    f"[{symbol}] ❌ Paper trade BLOCKED: {reason} "
                    f"(direction={signal_direction}, confidence={signal_confidence:.1%})"
                )
                return None  # Don't log blacklisted trades
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Trading controls check failed: {e} - Proceeding with paper trade")
        
        import uuid
        
        paper_trade_id = str(uuid.uuid4())
        signal_time = datetime.utcnow().isoformat()
        
        # =====================================================================
        # TRUST LADDER: Get dynamic prediction window based on symbol trust level
        # Level 1 (default): 48hr | Level 2: 120hr | Level 3: 168hr
        # =====================================================================
        try:
            from core.trust_ladder import get_symbol_prediction_window
            trust_config = get_symbol_prediction_window(symbol)
            prediction_hours = trust_config["prediction_hours"]
            trust_level = trust_config["trust_level"]
            LOGGER.info(f"[{symbol}] Trust Level {trust_level}: {prediction_hours}hr prediction window")
        except Exception as e:
            LOGGER.debug(f"[{symbol}] Trust ladder unavailable: {e} - using default 48hr")
            prediction_hours = 48
            trust_level = 1
        
        # Calculate target time based on trust level
        try:
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        except ValueError:
            # Handle timezone-naive format
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', ''))
        target_dt = entry_dt + timedelta(hours=prediction_hours)
        target_time = target_dt.isoformat()
        
        params = (
            paper_trade_id, cascade_id, symbol,
            signal_direction.upper(), signal_confidence, signal_time,
            entry_price, entry_time,
            target_time,
            position_size, stop_loss_pct, take_profit_pct,
            "PENDING",
            signal_time
        )
        
        try:
            if self.use_postgres:
                conn = self._get_postgres_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO paper_trades (
                        paper_trade_id, cascade_id, symbol,
                        signal_direction, signal_confidence, signal_time,
                        entry_price, entry_time, target_time,
                        position_size, stop_loss_pct, take_profit_pct,
                        outcome, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, params)
                conn.commit()
                conn.close()
            else:
                conn = sqlite3.connect(self.db_path)
                conn.execute("""
                    INSERT INTO paper_trades (
                        paper_trade_id, cascade_id, symbol,
                        signal_direction, signal_confidence, signal_time,
                        entry_price, entry_time, target_time,
                        position_size, stop_loss_pct, take_profit_pct,
                        outcome, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, params)
                conn.commit()
                conn.close()
            
            LOGGER.info(
                f"📝 Paper trade logged: {symbol} {signal_direction} "
                f"@ ${entry_price:,.2f} (conf={signal_confidence:.1%})"
            )
            
            return paper_trade_id
        
        except Exception as e:
            LOGGER.error(f"Failed to log paper trade: {e}")
            raise
    
    def check_outcome(self, paper_trade_id: str, current_price: float) -> dict:
        """
        Check if paper trade has reached target time and calculate outcome.
        
        Args:
            paper_trade_id: ID of paper trade to check
            current_price: Current market price
        
        Returns:
            {
                "resolved": bool,
                "outcome": "WIN|LOSS|STOPPED|BREAK_EVEN|PENDING",
                "profit_loss": float,
                "profit_loss_pct": float
            }
        """
        conn = self._get_connection()
        
        try:
            cur = self._execute(conn, """
                SELECT * FROM paper_trades
                WHERE paper_trade_id = ?
            """, (paper_trade_id,))
            
            trade = self._fetchone(cur)
            
            if not trade:
                return {"resolved": False, "error": "Trade not found"}
            
            # Already resolved?
            if trade["outcome"] != "PENDING":
                return {
                    "resolved": True,
                    "outcome": trade["outcome"],
                    "profit_loss": trade["profit_loss"],
                    "profit_loss_pct": trade["profit_loss_pct"]
                }
            
            # Target time reached?
            # Parse target_time and strip timezone to compare with naive utcnow
            target_time = datetime.fromisoformat(trade["target_time"].replace("Z", "+00:00"))
            if target_time.tzinfo is not None:
                target_time = target_time.replace(tzinfo=None)
            now = datetime.utcnow()
            
            if now < target_time:
                return {
                    "resolved": False,
                    "outcome": "PENDING",
                    "time_remaining": str(target_time - now)
                }
            
            # Calculate outcome - FIXED: Only evaluate at target time
            # Stop losses make sense for live trading but NOT for prediction accuracy evaluation
            # A prediction "BTC DOWN in 48h" should be evaluated AT 48 hours, not stopped early
            entry_price = trade["entry_price"]
            signal_direction = trade["signal_direction"]
            position_size = trade["position_size"]
            
            price_change_pct = (current_price - entry_price) / entry_price
            
            # Determine actual direction at target time
            if abs(price_change_pct) < 0.01:  # Within 1%
                actual_direction = "FLAT"
            elif price_change_pct > 0:
                actual_direction = "UP"
            else:
                actual_direction = "DOWN"
            
            # Simple win/loss evaluation based on prediction direction
            # Signal direction can be "LONG"/"SHORT" or "UP"/"DOWN"
            is_up_prediction = signal_direction in ("LONG", "UP")
            is_down_prediction = signal_direction in ("SHORT", "DOWN")
            
            if is_up_prediction:
                # UP prediction: WIN if price went up at target time
                if price_change_pct > 0.01:  # Up more than 1%
                    outcome = "WIN"
                    pnl_pct = price_change_pct
                elif price_change_pct < -0.01:  # Down more than 1%
                    outcome = "LOSS"
                    pnl_pct = price_change_pct
                else:
                    outcome = "BREAK_EVEN"
                    pnl_pct = 0.0
            
            elif is_down_prediction:
                # DOWN prediction: WIN if price went down at target time
                if price_change_pct < -0.01:  # Down more than 1%
                    outcome = "WIN"
                    pnl_pct = abs(price_change_pct)  # Positive P&L for correct DOWN prediction
                elif price_change_pct > 0.01:  # Up more than 1%
                    outcome = "LOSS"
                    pnl_pct = -abs(price_change_pct)  # Negative P&L for wrong DOWN prediction
                else:
                    outcome = "BREAK_EVEN"
                    pnl_pct = 0.0
            
            else:
                # Unknown direction
                LOGGER.warning(f"Unknown signal_direction: {signal_direction}, treating as BREAK_EVEN")
                outcome = "BREAK_EVEN"
                pnl_pct = 0.0
            
            pnl = position_size * pnl_pct
            
            # Update trade
            checked_at = now.isoformat()
            
            self._execute(conn, """
                UPDATE paper_trades
                SET target_price = ?,
                    actual_direction = ?,
                    outcome = ?,
                    profit_loss = ?,
                    profit_loss_pct = ?,
                    checked_at = ?
                WHERE paper_trade_id = ?
            """, (
                current_price,
                actual_direction,
                outcome,
                pnl,
                pnl_pct,
                checked_at,
                paper_trade_id
            ))
            conn.commit()
            
            LOGGER.info(
                f"✅ Paper trade resolved: {trade['symbol']} {outcome} "
                f"(${pnl:+,.2f}, {pnl_pct:+.2%})"
            )
            
            # =====================================================================
            # TRUST LADDER: Record outcome for promotion/demotion
            # WIN = move up ladder (longer prediction windows)
            # LOSS = move down ladder (back to 48hr)
            # =====================================================================
            trust_result = None
            try:
                from core.trust_ladder import record_prediction_outcome
                is_win = outcome == "WIN"
                trust_result = record_prediction_outcome(trade['symbol'], is_win)
                
                if trust_result.get("promoted"):
                    LOGGER.info(
                        f"🚀 {trade['symbol']} PROMOTED to Level {trust_result['new_level']} "
                        f"({trust_result['consecutive_wins']} consecutive wins)"
                    )
                elif trust_result.get("demoted"):
                    LOGGER.info(
                        f"📉 {trade['symbol']} DEMOTED to Level 1 "
                        f"({trust_result['consecutive_losses']} consecutive losses)"
                    )
            except Exception as e:
                LOGGER.debug(f"Trust ladder update failed: {e}")
            
            return {
                "resolved": True,
                "outcome": outcome,
                "profit_loss": pnl,
                "profit_loss_pct": pnl_pct,
                "actual_direction": actual_direction,
                "target_price": current_price,
                "trust_update": trust_result
            }
        
        except Exception as e:
            LOGGER.error(f"Failed to check paper trade outcome: {e}")
            return {"resolved": False, "error": str(e)}
        finally:
            conn.close()
    
    def check_all_pending(self, price_data: dict) -> list:
        """
        Check all pending paper trades and resolve if target time reached.
        
        Args:
            price_data: {"BTC": 87500.0, "ETH": 3200.0, ...}
        
        Returns:
            List of resolved trades
        """
        conn = self._get_connection()
        
        try:
            cur = self._execute(conn, """
                SELECT paper_trade_id, symbol, target_time
                FROM paper_trades
                WHERE outcome = 'PENDING'
                ORDER BY target_time ASC
            """)
            
            rows = self._fetchall(cur)
            
            resolved = []
            now = datetime.utcnow()
            
            for trade in rows:
                # Parse target_time and strip timezone to compare with naive utcnow
                target_time = datetime.fromisoformat(trade["target_time"].replace("Z", "+00:00"))
                if target_time.tzinfo is not None:
                    target_time = target_time.replace(tzinfo=None)
                
                # Target time reached?
                if now >= target_time:
                    symbol = trade["symbol"]
                    
                    # Have current price?
                    if symbol in price_data:
                        current_price = price_data[symbol]
                        result = self.check_outcome(trade["paper_trade_id"], current_price)
                        
                        if result.get("resolved"):
                            resolved.append({
                                "paper_trade_id": trade["paper_trade_id"],
                                "symbol": symbol,
                                **result
                            })
            
            if resolved:
                LOGGER.info(f"✅ Resolved {len(resolved)} paper trades")
            
            return resolved
        
        except Exception as e:
            LOGGER.error(f"Failed to check pending trades: {e}")
            return []
        finally:
            conn.close()
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None,
        outcome: Optional[str] = None,
        limit: int = 50
    ) -> list:
        """Get paper trades with filters"""
        conn = self._get_connection()
        
        try:
            query = "SELECT * FROM paper_trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.upper())
            
            if days:
                cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
                query += " AND created_at >= ?"
                params.append(cutoff)
            
            if outcome:
                query += " AND outcome = ?"
                params.append(outcome.upper())
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cur = self._execute(conn, query, tuple(params))
            return self._fetchall(cur)
        
        finally:
            conn.close()
    
    def get_stats(self, days: int = 30, since: str = None, v2_only: bool = False) -> dict:
        """
        Calculate paper trading statistics.
        
        Args:
            days: Number of days to look back (default: 30)
            since: Optional date string (e.g., "2026-01-14") to filter from specific date.
                   Overrides 'days' parameter when provided.
            v2_only: DEPRECATED - ignored. All symbols compete in Money Game now.
        
        Returns:
            {
                "total_trades": int,
                "resolved_trades": int,
                "pending_trades": int,
                "wins": int,
                "losses": int,
                "stopped": int,
                "win_rate": float,
                "total_pnl": float,
                "avg_win": float,
                "avg_loss": float,
                "best_trade": {...},
                "worst_trade": {...},
                "accuracy_by_symbol": {...}
            }
        """
        conn = self._get_connection()
        
        try:
            # Use 'since' parameter if provided, otherwise calculate from 'days'
            if since:
                cutoff = since
            else:
                cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            # MONEY GAME: No more V2 whitelist filtering - all symbols compete!
            # Removed Jan 29, 2026 - the Money Game system handles ranking
            symbol_filter = ""
            symbol_params = []
            
            # Overall stats
            cur = self._execute(conn, f"""
                SELECT COUNT(*) as count FROM paper_trades
                WHERE created_at >= ?{symbol_filter}
            """, (cutoff, *symbol_params))
            total = self._fetchone(cur)["count"]
            
            cur = self._execute(conn, f"""
                SELECT COUNT(*) as count FROM paper_trades
                WHERE created_at >= ? AND outcome != 'PENDING'{symbol_filter}
            """, (cutoff, *symbol_params))
            resolved = self._fetchone(cur)["count"]
            
            pending = total - resolved
            
            # Outcome counts
            cur = self._execute(conn, f"""
                SELECT COUNT(*) as count FROM paper_trades
                WHERE created_at >= ? AND outcome = 'WIN'{symbol_filter}
            """, (cutoff, *symbol_params))
            wins = self._fetchone(cur)["count"]
            
            cur = self._execute(conn, f"""
                SELECT COUNT(*) as count FROM paper_trades
                WHERE created_at >= ? AND outcome IN ('LOSS', 'STOPPED'){symbol_filter}
            """, (cutoff, *symbol_params))
            losses = self._fetchone(cur)["count"]
            
            cur = self._execute(conn, f"""
                SELECT COUNT(*) as count FROM paper_trades
                WHERE created_at >= ? AND outcome = 'STOPPED'{symbol_filter}
            """, (cutoff, *symbol_params))
            stopped = self._fetchone(cur)["count"]
            
            win_rate = wins / resolved if resolved > 0 else 0.0
            
            # P&L stats
            cur = self._execute(conn, f"""
                SELECT profit_loss FROM paper_trades
                WHERE created_at >= ? AND profit_loss IS NOT NULL{symbol_filter}
            """, (cutoff, *symbol_params))
            pnl_rows = self._fetchall(cur)
            
            total_pnl = sum(row["profit_loss"] for row in pnl_rows)
            
            cur = self._execute(conn, f"""
                SELECT profit_loss FROM paper_trades
                WHERE created_at >= ? AND outcome = 'WIN'{symbol_filter}
            """, (cutoff, *symbol_params))
            win_trades = self._fetchall(cur)
            
            cur = self._execute(conn, f"""
                SELECT profit_loss FROM paper_trades
                WHERE created_at >= ? AND outcome IN ('LOSS', 'STOPPED'){symbol_filter}
            """, (cutoff, *symbol_params))
            loss_trades = self._fetchall(cur)
            
            avg_win = sum(t["profit_loss"] for t in win_trades) / len(win_trades) if win_trades else 0.0
            avg_loss = sum(t["profit_loss"] for t in loss_trades) / len(loss_trades) if loss_trades else 0.0
            
            # Best/worst trades
            cur = self._execute(conn, f"""
                SELECT * FROM paper_trades
                WHERE created_at >= ? AND profit_loss IS NOT NULL{symbol_filter}
                ORDER BY profit_loss DESC LIMIT 1
            """, (cutoff, *symbol_params))
            best = self._fetchone(cur)
            
            cur = self._execute(conn, f"""
                SELECT * FROM paper_trades
                WHERE created_at >= ? AND profit_loss IS NOT NULL{symbol_filter}
                ORDER BY profit_loss ASC LIMIT 1
            """, (cutoff, *symbol_params))
            worst = self._fetchone(cur)
            
            # Accuracy by symbol - ALL symbols that have trades (Money Game)
            cur = self._execute(conn, f"""
                SELECT DISTINCT symbol FROM paper_trades
                WHERE created_at >= ?{symbol_filter}
            """, (cutoff, *symbol_params))
            symbols_to_check = self._fetchall(cur)
            
            accuracy_by_symbol = {}
            
            for sym_row in symbols_to_check:
                symbol = sym_row["symbol"]
                
                cur = self._execute(conn, """
                    SELECT COUNT(*) as count FROM paper_trades
                    WHERE created_at >= ? AND symbol = ? AND outcome != 'PENDING'
                """, (cutoff, symbol))
                sym_resolved = self._fetchone(cur)["count"]
                
                cur = self._execute(conn, """
                    SELECT COUNT(*) as count FROM paper_trades
                    WHERE created_at >= ? AND symbol = ? AND outcome = 'WIN'
                """, (cutoff, symbol))
                sym_wins = self._fetchone(cur)["count"]
                
                sym_win_rate = sym_wins / sym_resolved if sym_resolved > 0 else 0.0
                
                # Include all symbols with trades
                if sym_resolved > 0:
                    accuracy_by_symbol[symbol] = {
                        "trades": sym_resolved,
                        "wins": sym_wins,
                        "win_rate": sym_win_rate
                    }
            
            return {
                "total_trades": total,
                "resolved_trades": resolved,
                "pending_trades": pending,
                "wins": wins,
                "losses": losses,
                "stopped": stopped,
                "win_rate": win_rate,
                "win_rate_pct": round(win_rate * 100, 1),
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "best_trade": best,
                "worst_trade": worst,
                "accuracy_by_symbol": accuracy_by_symbol,
                "money_game_mode": True  # Using Money Game, not V2 whitelist
            }
        
        except Exception as e:
            LOGGER.error(f"Failed to calculate stats: {e}")
            return {}
        finally:
            conn.close()


# Singleton
_PAPER_TRACKER: Optional[PaperTracker] = None

def get_paper_tracker() -> PaperTracker:
    """Get singleton paper tracker instance"""
    global _PAPER_TRACKER
    if _PAPER_TRACKER is None:
        _PAPER_TRACKER = PaperTracker()
    return _PAPER_TRACKER