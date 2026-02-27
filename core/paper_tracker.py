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
from config.symbols import DEFAULT_EDGE_SYMBOLS

LOGGER = logging.getLogger("paper_tracker")

# PostgreSQL support for production
DATABASE_URL = os.getenv("DATABASE_URL")

# Only use PostgreSQL if DATABASE_URL is actually a PostgreSQL URL
# (not if it's a sqlite:/// URL which would crash psycopg2)
_IS_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith(("postgres://", "postgresql://")))


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
        self.use_postgres = _IS_POSTGRES
        self._ensure_table()
    
    def _get_postgres_connection(self):
        """Get PostgreSQL connection from shared pool.
        
        Returns a connection whose .close() returns it to the pool
        instead of destroying the underlying TCP socket.
        """
        from core.db_pool import get_sync_connection_raw
        return get_sync_connection_raw()
    
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
                        created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        trust_level INTEGER DEFAULT 1,
                        checkpoint_times JSONB DEFAULT '[]',
                        checkpoint_results JSONB DEFAULT '[]',
                        checkpoint_evaluated JSONB DEFAULT '[]',
                        checkpoint_prices JSONB DEFAULT '[]',
                        v3_validated BOOLEAN DEFAULT FALSE,
                        v3_strategy TEXT,
                        v3_is_inverse BOOLEAN DEFAULT FALSE,
                        v3_original_direction TEXT,
                        v3_hold_hours INTEGER,
                        v3_backtest_win_rate REAL
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
                        created_at TEXT NOT NULL,
                        trust_level INTEGER DEFAULT 1,
                        checkpoint_times TEXT,
                        checkpoint_results TEXT,
                        checkpoint_evaluated TEXT,
                        checkpoint_prices TEXT,
                        v3_validated INTEGER DEFAULT 0,
                        v3_strategy TEXT,
                        v3_is_inverse INTEGER DEFAULT 0,
                        v3_original_direction TEXT,
                        v3_hold_hours INTEGER,
                        v3_backtest_win_rate REAL
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
        
        # Run migrations to add columns that might be missing on existing tables
        self._run_migrations()
    
    def _run_migrations(self):
        """Add missing columns to existing tables (idempotent migrations)"""
        # Columns that may need to be added to existing tables
        migrations = [
            ("trust_level", "INTEGER DEFAULT 1"),
            ("checkpoint_times", "JSONB DEFAULT '[]'" if self.use_postgres else "TEXT"),
            ("checkpoint_results", "JSONB DEFAULT '[]'" if self.use_postgres else "TEXT"),
            ("checkpoint_evaluated", "JSONB DEFAULT '[]'" if self.use_postgres else "TEXT"),
            ("checkpoint_prices", "JSONB DEFAULT '[]'" if self.use_postgres else "TEXT"),
            ("v3_validated", "BOOLEAN DEFAULT FALSE" if self.use_postgres else "INTEGER DEFAULT 0"),
            ("v3_strategy", "TEXT"),
            ("v3_is_inverse", "BOOLEAN DEFAULT FALSE" if self.use_postgres else "INTEGER DEFAULT 0"),
            ("v3_original_direction", "TEXT"),
            ("v3_hold_hours", "INTEGER"),
            ("v3_backtest_win_rate", "REAL"),
            ("expected_move_pct", "REAL"),
            ("actual_move_pct", "REAL"),
            ("magnitude_error_pct", "REAL"),
        ]
        
        try:
            conn = self._get_connection()
            
            for col_name, col_type in migrations:
                try:
                    if self.use_postgres:
                        cur = conn.cursor()
                        # PostgreSQL: Check if column exists
                        cur.execute("""
                            SELECT column_name FROM information_schema.columns 
                            WHERE table_name = 'paper_trades' AND column_name = %s
                        """, (col_name,))
                        if not cur.fetchone():
                            cur.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                            conn.commit()
                            LOGGER.info(f"🔧 Migration: Added column {col_name} to paper_trades")
                    else:
                        # SQLite: Try to add, ignore if exists
                        try:
                            conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                            conn.commit()
                            LOGGER.info(f"🔧 Migration: Added column {col_name} to paper_trades")
                        except sqlite3.OperationalError as e:
                            if "duplicate column" not in str(e).lower():
                                raise
                except Exception as col_err:
                    LOGGER.debug(f"Column {col_name} migration skipped: {col_err}")
            
            conn.close()
        except Exception as e:
            LOGGER.warning(f"Migration check failed (non-fatal): {e}")
    
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
        take_profit_pct: float = 0.10,
        # V3 metadata (optional)
        v3_validated: bool = False,
        v3_strategy: str = None,
        v3_is_inverse: bool = False,
        v3_original_direction: str = None,
        v3_hold_hours: int = None,
        v3_backtest_win_rate: float = None,
        expected_move_pct: float = None
    ) -> str:
        """
        Log a Ghost signal as a paper trade.
        
        Called when cascade reaches 6h final call.
        Supports both PostgreSQL (production) and SQLite (local).
        """
        # =====================================================================
        # EDGE SYMBOL WHITELIST (Feb 9, 2026): Only trade proven symbols
        # 30-day data: 24 edge symbols = 74.7% WR, 76 non-edge = 21.5% WR
        # Catches ALL callers: run_prediction, ghost_notifications, cascade
        # =====================================================================
        _edge_whitelist_enabled = os.environ.get("EDGE_WHITELIST_ENABLED", "1") == "1"
        if _edge_whitelist_enabled:
            _edge_csv = os.environ.get("EDGE_SYMBOLS", DEFAULT_EDGE_SYMBOLS)
            _edge_set = set(s.strip().upper() for s in _edge_csv.split(",") if s.strip())
            if symbol.upper() not in _edge_set:
                LOGGER.info(
                    f"[{symbol}] 🚫 EDGE WHITELIST (centralized): Not in {len(_edge_set)} proven symbols — blocking paper trade"
                )
                return None
        
        # =====================================================================
        # PRICE SANITY: Reject $0.00 or near-zero entry prices (Feb 11, 2026)
        # GIGA was logging paper trades at $0.00 — corrupts P&L tracking
        # =====================================================================
        if entry_price is None or entry_price <= 0:
            LOGGER.warning(f"[{symbol}] 🚫 PRICE SANITY (centralized): entry_price is {entry_price} — blocking paper trade")
            return None
        if entry_price < 0.00001:
            LOGGER.warning(f"[{symbol}] 🚫 PRICE SANITY (centralized): entry_price ${entry_price} suspiciously low — blocking paper trade")
            return None
        if entry_price > 1_000_000:
            LOGGER.warning(f"[{symbol}] 🚫 PRICE SANITY (centralized): entry_price ${entry_price:,.2f} suspiciously high — blocking paper trade")
            return None

        # =====================================================================
        # HOLD ZONE: Don't log paper trades for HOLD signals
        # =====================================================================
        if signal_direction and signal_direction.upper() == "HOLD":
            LOGGER.info(f"[{symbol}] 🛑 HOLD ZONE (centralized): Model has no conviction — skipping paper trade")
            return None
        
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
        
        # =====================================================================
        # CENTRALIZED DEDUP: Prevent duplicate trades for same symbol within window
        # All callers (run_prediction, stock_engine, ghost_notifications, cascade)
        # go through log_signal(), so dedup here catches ALL paths
        # =====================================================================
        _dedup_minutes = int(os.environ.get("PAPER_TRADE_DEDUP_MINUTES", "90"))
        try:
            dedup_conn = self._get_connection()
            dedup_cutoff = (datetime.utcnow() - timedelta(minutes=_dedup_minutes)).isoformat()
            dedup_cur = self._execute(dedup_conn,
                "SELECT COUNT(*) as cnt FROM paper_trades WHERE symbol = ? AND entry_time > ?",
                (symbol.upper(), dedup_cutoff)
            )
            dedup_row = self._fetchall(dedup_cur)
            dedup_conn.close()
            recent_count = dedup_row[0]["cnt"] if dedup_row and dedup_row[0] else 0
            if recent_count > 0:
                LOGGER.info(f"[{symbol}] ⏭️ DEDUP (centralized): Already {recent_count} trade(s) in last {_dedup_minutes}min — skipping")
                return None
        except Exception as dedup_err:
            LOGGER.debug(f"[{symbol}] Centralized dedup check failed (continuing): {dedup_err}")
        
        import uuid
        
        paper_trade_id = str(uuid.uuid4())
        signal_time = datetime.utcnow().isoformat()
        
        # =====================================================================
        # TRUST LADDER: Get dynamic prediction window based on symbol trust level
        # Level 1 (default): 48hr | Level 2: 120hr | Level 3: 168hr
        # V3 OVERRIDE: If v3_hold_hours is specified, use that instead
        # =====================================================================
        try:
            from core.trust_ladder import get_symbol_prediction_window, TRUST_LEVELS
            trust_config = get_symbol_prediction_window(symbol)
            prediction_hours = trust_config["prediction_hours"]
            trust_level = trust_config["trust_level"]
            checkpoints = TRUST_LEVELS[trust_level]["checkpoints"]
            LOGGER.info(f"[{symbol}] Trust Level {trust_level}: {prediction_hours}hr prediction window, checkpoints={checkpoints}")
        except Exception as e:
            LOGGER.debug(f"[{symbol}] Trust ladder unavailable: {e} - using default 72hr")
            prediction_hours = 72  # Changed from 48 (backtest validated)
            trust_level = 1
            checkpoints = [72]
        
        # V3 OVERRIDE: Use V3-specific hold hours if this is a V3 validated signal
        if v3_validated and v3_hold_hours:
            prediction_hours = v3_hold_hours
            checkpoints = [v3_hold_hours]  # Single checkpoint at hold time
            LOGGER.info(f"[{symbol}] V3 OVERRIDE: Using {v3_hold_hours}hr hold period (strategy={v3_strategy})")
        
        # Calculate target time based on trust level
        try:
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        except ValueError:
            # Handle timezone-naive format
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', ''))
        target_dt = entry_dt + timedelta(hours=prediction_hours)
        target_time = target_dt.isoformat()
        
        # =====================================================================
        # MULTI-CHECKPOINT: Calculate checkpoint times for progressive evaluation
        # Level 2: checkpoints at [60, 120] hours
        # Level 3: checkpoints at [72, 168] hours
        # =====================================================================
        checkpoint_times = []
        checkpoint_results = []
        checkpoint_evaluated = []
        for cp_hours in checkpoints:
            cp_dt = entry_dt + timedelta(hours=cp_hours)
            checkpoint_times.append(cp_dt.isoformat())
            checkpoint_results.append(None)  # Will be WIN/LOSS when evaluated
            checkpoint_evaluated.append(False)
        
        LOGGER.info(
            f"[{symbol}] Trust Level {trust_level}: Checkpoints scheduled at "
            f"{', '.join([f'{cp}hr' for cp in checkpoints])}"
        )
        
        params = (
            paper_trade_id, cascade_id, symbol,
            signal_direction.upper(), signal_confidence, signal_time,
            entry_price, entry_time,
            target_time,
            position_size, stop_loss_pct, take_profit_pct,
            "PENDING",
            signal_time,
            trust_level,
            json.dumps(checkpoint_times),
            json.dumps(checkpoint_results),
            json.dumps(checkpoint_evaluated),
            json.dumps([]),  # checkpoint_prices starts empty
            # V3 tracking fields
            v3_validated,
            v3_strategy,
            v3_is_inverse,
            v3_original_direction,
            v3_hold_hours,
            v3_backtest_win_rate,
            expected_move_pct
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
                        outcome, created_at,
                        trust_level, checkpoint_times, checkpoint_results,
                        checkpoint_evaluated, checkpoint_prices,
                        v3_validated, v3_strategy, v3_is_inverse,
                        v3_original_direction, v3_hold_hours, v3_backtest_win_rate,
                        expected_move_pct
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        outcome, created_at,
                        trust_level, checkpoint_times, checkpoint_results,
                        checkpoint_evaluated, checkpoint_prices,
                        v3_validated, v3_strategy, v3_is_inverse,
                        v3_original_direction, v3_hold_hours, v3_backtest_win_rate,
                        expected_move_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, params)
                conn.commit()
                conn.close()
            
            # Enhanced logging for V3 trades
            v3_info = ""
            if v3_validated:
                v3_info = f", V3={v3_strategy}, hold={v3_hold_hours}hr, backtest={v3_backtest_win_rate:.1%}" if v3_backtest_win_rate else f", V3={v3_strategy}, hold={v3_hold_hours}hr"
            
            mag_info = f", expected_move={expected_move_pct:+.1f}%" if expected_move_pct else ""
            
            LOGGER.info(
                f"📝 Paper trade logged: {symbol} {signal_direction} "
                f"@ ${entry_price:,.2f} (conf={signal_confidence:.1%}, trust_level={trust_level}{v3_info}{mag_info})"
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
            
            # Magnitude tracking: compare predicted vs actual move
            actual_move_pct_val = abs(price_change_pct * 100)  # Always positive
            expected_move = trade.get("expected_move_pct") or 0
            magnitude_error = abs(actual_move_pct_val - expected_move) if expected_move else None
            
            self._execute(conn, """
                UPDATE paper_trades
                SET target_price = ?,
                    actual_direction = ?,
                    outcome = ?,
                    profit_loss = ?,
                    profit_loss_pct = ?,
                    checked_at = ?,
                    actual_move_pct = ?,
                    magnitude_error_pct = ?
                WHERE paper_trade_id = ?
            """, (
                current_price,
                actual_direction,
                outcome,
                pnl,
                pnl_pct,
                checked_at,
                round(actual_move_pct_val, 2),
                round(magnitude_error, 2) if magnitude_error is not None else None,
                paper_trade_id
            ))
            conn.commit()
            
            mag_info = f", actual_move={actual_move_pct_val:.1f}%, predicted={expected_move:.1f}%" if expected_move else ""
            LOGGER.info(
                f"✅ Paper trade resolved: {trade['symbol']} {outcome} "
                f"(${pnl:+,.2f}, {pnl_pct:+.2%}{mag_info})"
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
            
            # =====================================================================
            # ONLINE CALIBRATOR: Log forecast result for weight recalibration
            # Feeds the calibration engine with real outcome data so it can
            # adjust horizon weights and strategy weights over time.
            # =====================================================================
            try:
                from core.online_calibrator import get_online_calibrator
                
                calibrator = get_online_calibrator()
                # Derive horizon from actual trade metadata instead of hardcoding
                _hold_hours = trade.get('v3_hold_hours') or 0
                if _hold_hours <= 12:
                    _cal_horizon = "nowcast"
                elif _hold_hours <= 72:
                    _cal_horizon = "swing"
                else:
                    _cal_horizon = "position"
                calibrator.log_forecast_result(
                    horizon=_cal_horizon,
                    symbol=trade['symbol'],
                    predicted_price=float(trade.get('target_price') or entry_price),
                    actual_price=float(current_price),
                    confidence=float(trade.get('signal_confidence') or 0.5),
                )
                
                # Also log as strategy result for strategy weight calibration
                calibrator.log_strategy_result(
                    strategy_name=trade.get('v3_strategy') or "ensemble_v3",
                    symbol=trade['symbol'],
                    action=trade.get('signal_direction', 'UP'),
                    confidence=float(trade.get('signal_confidence') or 0.5),
                    entry_price=float(entry_price),
                    exit_price=float(current_price),
                )
                LOGGER.debug(f"[{trade['symbol']}] Online calibrator: logged forecast+strategy result")
            except Exception as e:
                LOGGER.debug(f"Online calibrator logging failed: {e}")
            
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
        Check all pending paper trades and evaluate checkpoints + final resolution.
        
        MULTI-CHECKPOINT SYSTEM:
        - Level 1: Single checkpoint at 48hr
        - Level 2: Checkpoints at 60hr and 120hr (both must pass)
        - Level 3: Checkpoints at 72hr and 168hr (both must pass)
        
        Args:
            price_data: {"BTC": 87500.0, "ETH": 3200.0, ...}
        
        Returns:
            List of resolved trades (final outcomes only)
        """
        conn = self._get_connection()
        
        try:
            cur = self._execute(conn, """
                SELECT paper_trade_id, symbol, target_time, signal_direction, entry_price,
                       trust_level, checkpoint_times, checkpoint_results, checkpoint_evaluated, checkpoint_prices
                FROM paper_trades
                WHERE outcome = 'PENDING'
                ORDER BY target_time ASC
            """)
            
            rows = self._fetchall(cur)
            
            resolved = []
            checkpoint_updates = []
            now = datetime.utcnow()
            
            for trade in rows:
                symbol = trade["symbol"]
                paper_trade_id = trade["paper_trade_id"]
                
                # Skip if no current price available
                if symbol not in price_data:
                    continue
                
                current_price = price_data[symbol]
                
                # =====================================================================
                # MULTI-CHECKPOINT EVALUATION
                # Check each checkpoint that hasn't been evaluated yet
                # =====================================================================
                trust_level = trade.get("trust_level", 1) or 1
                
                # Parse checkpoint arrays (handle both string JSON and native JSONB)
                checkpoint_times = trade.get("checkpoint_times") or []
                checkpoint_results = trade.get("checkpoint_results") or []
                checkpoint_evaluated = trade.get("checkpoint_evaluated") or []
                checkpoint_prices = trade.get("checkpoint_prices") or []
                
                if isinstance(checkpoint_times, str):
                    checkpoint_times = json.loads(checkpoint_times)
                if isinstance(checkpoint_results, str):
                    checkpoint_results = json.loads(checkpoint_results)
                if isinstance(checkpoint_evaluated, str):
                    checkpoint_evaluated = json.loads(checkpoint_evaluated)
                if isinstance(checkpoint_prices, str):
                    checkpoint_prices = json.loads(checkpoint_prices)
                
                # Ensure lists are proper length
                if not checkpoint_times:
                    # Legacy trade without checkpoints - treat as single checkpoint at target_time
                    checkpoint_times = [trade["target_time"]]
                    checkpoint_results = [None]
                    checkpoint_evaluated = [False]
                    checkpoint_prices = []
                
                entry_price = trade["entry_price"]
                signal_direction = trade["signal_direction"]
                checkpoints_updated = False
                
                # Check each checkpoint
                for i, cp_time_str in enumerate(checkpoint_times):
                    # Skip if already evaluated
                    if i < len(checkpoint_evaluated) and checkpoint_evaluated[i]:
                        continue
                    
                    # Parse checkpoint time
                    cp_time = datetime.fromisoformat(cp_time_str.replace("Z", "+00:00"))
                    if cp_time.tzinfo is not None:
                        cp_time = cp_time.replace(tzinfo=None)
                    
                    # Checkpoint time reached?
                    if now >= cp_time:
                        # Calculate checkpoint outcome
                        price_change = current_price - entry_price
                        price_change_pct = price_change / entry_price if entry_price > 0 else 0
                        actual_direction = "UP" if price_change > 0 else "DOWN"
                        
                        # Determine if this checkpoint is a win
                        # Signal directions are stored as UP/DOWN (not BULLISH/BEARISH)
                        is_up_pred = signal_direction in ("UP", "LONG", "BULLISH")
                        is_down_pred = signal_direction in ("DOWN", "SHORT", "BEARISH")
                        
                        # Dead zone: less than 1% move = BREAK_EVEN at checkpoint
                        if abs(price_change_pct) < 0.01:
                            cp_result = "BREAK_EVEN"
                        elif is_up_pred:
                            cp_result = "WIN" if actual_direction == "UP" else "LOSS"
                        elif is_down_pred:
                            cp_result = "WIN" if actual_direction == "DOWN" else "LOSS"
                        else:
                            LOGGER.warning(f"[{symbol}] Unknown signal_direction at checkpoint: {signal_direction}")
                            cp_result = "BREAK_EVEN"
                        
                        # Update checkpoint arrays
                        while len(checkpoint_results) <= i:
                            checkpoint_results.append(None)
                        while len(checkpoint_evaluated) <= i:
                            checkpoint_evaluated.append(False)
                        while len(checkpoint_prices) <= i:
                            checkpoint_prices.append(None)
                        
                        checkpoint_results[i] = cp_result
                        checkpoint_evaluated[i] = True
                        checkpoint_prices[i] = current_price
                        checkpoints_updated = True
                        
                        # Calculate which checkpoint this is (1/2, 2/2, etc.)
                        cp_num = i + 1
                        total_cps = len(checkpoint_times)
                        
                        # Log checkpoint evaluation
                        cp_emoji = "✓" if cp_result == "WIN" else "✗"
                        LOGGER.info(
                            f"[{symbol}] Trust Level {trust_level}: Checkpoint {cp_num}/{total_cps} "
                            f"- {cp_result} {cp_emoji} (entry=${entry_price:.2f}, now=${current_price:.2f})"
                        )
                        
                        # =====================================================================
                        # TRUST LADDER: Record checkpoint outcome
                        # This is what enables multi-checkpoint tracking for promotion
                        # =====================================================================
                        try:
                            from core.trust_ladder import record_prediction_outcome
                            is_win = cp_result == "WIN"
                            is_final = (cp_num == total_cps)  # Is this the final checkpoint?
                            
                            trust_result = record_prediction_outcome(
                                symbol, 
                                is_win, 
                                is_checkpoint=not is_final  # True for intermediate, False for final
                            )
                            
                            if trust_result.get("promoted"):
                                LOGGER.info(
                                    f"🚀 {symbol} PROMOTED to Level {trust_result['new_level']} "
                                    f"(all checkpoints passed)"
                                )
                            elif trust_result.get("demoted"):
                                LOGGER.info(
                                    f"📉 {symbol} DEMOTED to Level 1 "
                                    f"(checkpoint {cp_num}/{total_cps} failed)"
                                )
                        except Exception as e:
                            LOGGER.debug(f"Trust ladder checkpoint update failed: {e}")
                
                # Save checkpoint updates to database
                if checkpoints_updated:
                    try:
                        if self.use_postgres:
                            update_conn = self._get_postgres_connection()
                            update_cur = update_conn.cursor()
                            update_cur.execute("""
                                UPDATE paper_trades
                                SET checkpoint_results = %s,
                                    checkpoint_evaluated = %s,
                                    checkpoint_prices = %s
                                WHERE paper_trade_id = %s
                            """, (
                                json.dumps(checkpoint_results),
                                json.dumps(checkpoint_evaluated),
                                json.dumps(checkpoint_prices),
                                paper_trade_id
                            ))
                            update_conn.commit()
                            update_conn.close()
                        else:
                            update_conn = sqlite3.connect(self.db_path)
                            update_conn.execute("""
                                UPDATE paper_trades
                                SET checkpoint_results = ?,
                                    checkpoint_evaluated = ?,
                                    checkpoint_prices = ?
                                WHERE paper_trade_id = ?
                            """, (
                                json.dumps(checkpoint_results),
                                json.dumps(checkpoint_evaluated),
                                json.dumps(checkpoint_prices),
                                paper_trade_id
                            ))
                            update_conn.commit()
                            update_conn.close()
                            
                        LOGGER.info(
                            f"[{symbol}] Checkpoint data saved: {checkpoint_results}"
                        )
                    except Exception as e:
                        LOGGER.error(f"Failed to save checkpoint data: {e}")
                
                # =====================================================================
                # FINAL RESOLUTION: Check if target time reached for final outcome
                # =====================================================================
                target_time = datetime.fromisoformat(trade["target_time"].replace("Z", "+00:00"))
                if target_time.tzinfo is not None:
                    target_time = target_time.replace(tzinfo=None)
                
                if now >= target_time:
                    result = self.check_outcome(paper_trade_id, current_price)
                    
                    if result.get("resolved"):
                        resolved.append({
                            "paper_trade_id": paper_trade_id,
                            "symbol": symbol,
                            "checkpoint_results": checkpoint_results,
                            **result
                        })
            
            if resolved:
                LOGGER.info(f"✅ Resolved {len(resolved)} paper trades (final outcomes)")
            
            return resolved
        
        except Exception as e:
            LOGGER.error(f"Failed to check pending trades: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
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
            
            # OPTIMIZED: Single aggregate query instead of 8+ separate ones
            cur = self._execute(conn, """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome != 'PENDING' THEN 1 ELSE 0 END) as resolved,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome IN ('LOSS', 'STOPPED') THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome = 'STOPPED' THEN 1 ELSE 0 END) as stopped,
                    SUM(CASE WHEN outcome = 'BREAK_EVEN' THEN 1 ELSE 0 END) as break_even,
                    SUM(CASE WHEN outcome = 'EXPIRED' THEN 1 ELSE 0 END) as expired,
                    SUM(CASE WHEN profit_loss IS NOT NULL THEN profit_loss ELSE 0 END) as total_pnl
                FROM paper_trades
                WHERE created_at >= ?
            """, (cutoff,))
            agg = self._fetchone(cur)
            
            total = agg["total"] or 0
            resolved = agg["resolved"] or 0
            wins = agg["wins"] or 0
            losses = agg["losses"] or 0
            stopped = agg["stopped"] or 0
            break_even = agg["break_even"] or 0
            expired = agg["expired"] or 0
            total_pnl = agg["total_pnl"] or 0
            pending = total - resolved
            
            decided = wins + losses
            win_rate = wins / decided if decided > 0 else 0.0
            
            # OPTIMIZED: Single query for avg win/loss
            cur = self._execute(conn, """
                SELECT 
                    AVG(CASE WHEN outcome = 'WIN' THEN profit_loss END) as avg_win,
                    AVG(CASE WHEN outcome IN ('LOSS', 'STOPPED') THEN profit_loss END) as avg_loss
                FROM paper_trades
                WHERE created_at >= ? AND profit_loss IS NOT NULL
            """, (cutoff,))
            avgs = self._fetchone(cur)
            avg_win = avgs["avg_win"] or 0.0
            avg_loss = avgs["avg_loss"] or 0.0
            
            # Best/worst trades
            cur = self._execute(conn, """
                SELECT * FROM paper_trades
                WHERE created_at >= ? AND profit_loss IS NOT NULL
                ORDER BY profit_loss DESC LIMIT 1
            """, (cutoff,))
            best = self._fetchone(cur)
            
            cur = self._execute(conn, """
                SELECT * FROM paper_trades
                WHERE created_at >= ? AND profit_loss IS NOT NULL
                ORDER BY profit_loss ASC LIMIT 1
            """, (cutoff,))
            worst = self._fetchone(cur)
            
            # OPTIMIZED: Single query for per-symbol accuracy (was N*3 queries for N symbols)
            cur = self._execute(conn, """
                SELECT 
                    symbol,
                    SUM(CASE WHEN outcome != 'PENDING' THEN 1 ELSE 0 END) as resolved,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome IN ('LOSS', 'STOPPED') THEN 1 ELSE 0 END) as losses
                FROM paper_trades
                WHERE created_at >= ?
                GROUP BY symbol
                HAVING SUM(CASE WHEN outcome IN ('WIN', 'LOSS', 'STOPPED') THEN 1 ELSE 0 END) > 0
            """, (cutoff,))
            sym_rows = self._fetchall(cur)
            
            accuracy_by_symbol = {}
            for row in sym_rows:
                sym_wins = row["wins"] or 0
                sym_losses = row["losses"] or 0
                sym_decided = sym_wins + sym_losses
                if sym_decided > 0:
                    accuracy_by_symbol[row["symbol"]] = {
                        "trades": row["resolved"] or 0,
                        "wins": sym_wins,
                        "losses": sym_losses,
                        "win_rate": sym_wins / sym_decided
                    }
            
            return {
                "total_trades": total,
                "resolved_trades": resolved,
                "pending_trades": pending,
                "wins": wins,
                "losses": losses,
                "stopped": stopped,
                "break_even": break_even,
                "expired": expired,
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