"""
Ghost Position Manager
======================

Tracks positions across days to ensure consistency.
Locks entry prices when predictions are made.
Prevents positions from "drifting" with market prices.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

LOGGER = logging.getLogger("position_manager")


class PositionManager:
    """
    Manages open positions with locked entry prices.
    
    Key features:
    - Locks entry price when position opened
    - Tracks positions across multiple days
    - Prevents entry price drift
    - Validates if position should continue or exit
    """
    
    def __init__(self, db_path: str = "data/ghost_predictions.db"):
        self.db_path = db_path
        self._ensure_table()
    
    def _ensure_table(self):
        """Create positions table if not exists"""
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL UNIQUE,
                    asset_type TEXT NOT NULL,
                    
                    -- Locked at entry
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    direction TEXT NOT NULL,
                    
                    -- Original prediction
                    predicted_gain_pct REAL NOT NULL,
                    original_confidence REAL NOT NULL,
                    reasoning TEXT,
                    
                    -- Current state
                    current_price REAL,
                    current_confidence REAL,
                    current_pnl_pct REAL,
                    last_update_time TEXT,
                    
                    -- Exit tracking
                    exit_price REAL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    realized_pnl_pct REAL,
                    
                    -- Status
                    status TEXT DEFAULT 'active',  -- active, exited, stopped
                    days_held INTEGER DEFAULT 0,
                    
                    -- Metadata
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_status 
                ON active_positions(status, symbol)
            """)
            
            conn.commit()
            LOGGER.info("✅ Position manager table ready")
        
        except Exception as e:
            LOGGER.error(f"Failed to create positions table: {e}")
            raise
        finally:
            conn.close()
    
    def open_position(
        self,
        symbol: str,
        asset_type: str,
        entry_price: float,
        target_price: float,
        direction: str,
        predicted_gain_pct: float,
        confidence: float,
        reasoning: str = ""
    ) -> str:
        """
        Open new position with LOCKED entry price.
        
        Returns position_id or None if position already exists.
        """
        import uuid
        
        # Check if position already exists
        existing = self.get_position(symbol)
        if existing and existing['status'] == 'active':
            LOGGER.info(f"📍 Position {symbol} already active, keeping original entry ${existing['entry_price']:.2f}")
            return existing['position_id']
        
        position_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        # Use AssetClassifier for proper stop sizing
        try:
            from core.asset_classifier import AssetClassifier
            targets = AssetClassifier.get_target_stop(symbol, horizon_hours=48)
            stop_pct = targets["stop_pct"]
        except:
            stop_pct = 4.0  # Fallback
        
        if direction == "UP":
            stop_loss = entry_price * (1 - stop_pct / 100)
        else:
            stop_loss = entry_price * (1 + stop_pct / 100)
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO active_positions (
                    position_id, symbol, asset_type,
                    entry_price, entry_time, target_price, stop_loss, direction,
                    predicted_gain_pct, original_confidence, reasoning,
                    current_price, current_confidence, current_pnl_pct,
                    last_update_time, status, days_held
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 0)
            """, (
                position_id, symbol, asset_type,
                entry_price, now, target_price, stop_loss, direction,
                predicted_gain_pct, confidence, reasoning,
                entry_price, confidence, 0.0, now
            ))
            
            conn.commit()
            LOGGER.info(f"🔒 Opened position: {symbol} @ ${entry_price:.2f} (target: ${target_price:.2f})")
            return position_id
        
        except sqlite3.IntegrityError:
            LOGGER.warning(f"Position {symbol} already exists, keeping original")
            return existing['position_id'] if existing else None
        finally:
            conn.close()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position by symbol"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            row = conn.execute("""
                SELECT * FROM active_positions
                WHERE symbol = ? AND status = 'active'
            """, (symbol,)).fetchone()
            
            return dict(row) if row else None
        finally:
            conn.close()
    
    def get_all_active(self) -> List[Dict]:
        """Get all active positions"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            rows = conn.execute("""
                SELECT * FROM active_positions
                WHERE status = 'active'
                ORDER BY entry_time DESC
            """).fetchall()
            
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_confidence: Optional[float] = None
    ) -> Dict:
        """
        Update position with current price.
        
        Entry price remains LOCKED.
        Returns updated position with PnL calculated from locked entry.
        """
        position = self.get_position(symbol)
        if not position:
            LOGGER.warning(f"Position {symbol} not found")
            return None
        
        # Calculate PnL from LOCKED entry price
        entry_price = position['entry_price']
        direction = position['direction']
        
        if direction == "UP":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        now = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            if current_confidence is not None:
                conn.execute("""
                    UPDATE active_positions
                    SET current_price = ?,
                        current_confidence = ?,
                        current_pnl_pct = ?,
                        last_update_time = ?
                    WHERE symbol = ? AND status = 'active'
                """, (current_price, current_confidence, pnl_pct, now, symbol))
            else:
                conn.execute("""
                    UPDATE active_positions
                    SET current_price = ?,
                        current_pnl_pct = ?,
                        last_update_time = ?
                    WHERE symbol = ? AND status = 'active'
                """, (current_price, pnl_pct, now, symbol))
            
            conn.commit()
            
            # Return updated position
            position['current_price'] = current_price
            position['current_pnl_pct'] = pnl_pct
            if current_confidence is not None:
                position['current_confidence'] = current_confidence
            
            return position
        
        finally:
            conn.close()
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str
    ) -> Dict:
        """
        Close position and calculate realized PnL.
        
        Uses LOCKED entry price for PnL calculation.
        """
        position = self.get_position(symbol)
        if not position:
            LOGGER.warning(f"Position {symbol} not found")
            return None
        
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate realized PnL from LOCKED entry
        if direction == "UP":
            realized_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            realized_pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        now = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE active_positions
                SET status = 'exited',
                    exit_price = ?,
                    exit_time = ?,
                    exit_reason = ?,
                    realized_pnl_pct = ?
                WHERE symbol = ? AND status = 'active'
            """, (exit_price, now, exit_reason, realized_pnl_pct, symbol))
            
            conn.commit()
            
            LOGGER.info(f"✅ Closed {symbol}: Entry ${entry_price:.2f} → Exit ${exit_price:.2f} = {realized_pnl_pct:+.1f}% ({exit_reason})")
            
            return {
                **position,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'realized_pnl_pct': realized_pnl_pct
            }
        
        finally:
            conn.close()
    
    def check_stop_losses(self) -> List[Dict]:
        """
        Check all active positions for stop loss hits.
        
        Returns list of positions that hit stop loss.
        """
        stopped = []
        active_positions = self.get_all_active()
        
        for position in active_positions:
            current_price = position.get('current_price')
            stop_loss = position['stop_loss']
            direction = position['direction']
            
            if not current_price:
                continue
            
            # Check if stop loss hit
            if direction == "UP" and current_price <= stop_loss:
                stopped.append(position)
                self.close_position(
                    position['symbol'],
                    current_price,
                    f"Stop loss hit (${current_price:.2f} <= ${stop_loss:.2f})"
                )
            
            elif direction == "DOWN" and current_price >= stop_loss:
                stopped.append(position)
                self.close_position(
                    position['symbol'],
                    current_price,
                    f"Stop loss hit (${current_price:.2f} >= ${stop_loss:.2f})"
                )
        
        return stopped
    
    def increment_days_held(self):
        """Increment days_held for all active positions (call daily)"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                UPDATE active_positions
                SET days_held = days_held + 1
                WHERE status = 'active'
            """)
            conn.commit()
        finally:
            conn.close()
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Count active and closed positions
            active_count = conn.execute("""
                SELECT COUNT(*) FROM active_positions WHERE status = 'active'
            """).fetchone()[0]
            
            closed_count = conn.execute("""
                SELECT COUNT(*) FROM active_positions WHERE status = 'exited'
            """).fetchone()[0]
            
            # Get win/loss stats
            winners = conn.execute("""
                SELECT COUNT(*), AVG(realized_pnl_pct)
                FROM active_positions
                WHERE status = 'exited' AND realized_pnl_pct > 0
            """).fetchone()
            
            losers = conn.execute("""
                SELECT COUNT(*), AVG(realized_pnl_pct)
                FROM active_positions
                WHERE status = 'exited' AND realized_pnl_pct <= 0
            """).fetchone()
            
            win_count, avg_win = winners if winners[0] else (0, 0)
            loss_count, avg_loss = losers if losers[0] else (0, 0)
            
            win_rate = (win_count / closed_count * 100) if closed_count > 0 else 0
            
            return {
                'active_positions': active_count,
                'closed_positions': closed_count,
                'wins': win_count,
                'losses': loss_count,
                'win_rate': win_rate,
                'avg_win_pct': avg_win or 0,
                'avg_loss_pct': avg_loss or 0
            }
        
        finally:
            conn.close()


# Singleton instance
_position_manager = None

def get_position_manager() -> PositionManager:
    """Get singleton position manager instance"""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager
