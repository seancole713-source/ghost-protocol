"""
Goal Tracker Module
Tracks YTD P&L, win rate, and trading statistics
"""

import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Database path
_raw_db_path = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
if _raw_db_path.startswith("/data") and not Path("/data").exists():
    DB_PATH = "data" + _raw_db_path[5:]
else:
    DB_PATH = _raw_db_path


def get_ytd_stats() -> dict[str, Any]:
    """
    Get year-to-date trading statistics
    
    Returns:
        {
            'ytd_pnl': float,
            'ytd_target': float,
            'win_rate': float,
            'total_trades': int,
            'avg_gain': float,
            'avg_loss': float
        }
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get current portfolio PnL
        cursor.execute("""
            SELECT 
                COALESCE(SUM((last_known_price - avg_cost) * quantity), 0) as unrealized_pnl
            FROM portfolio_positions
            WHERE quantity > 0
        """)
        unrealized_pnl = cursor.fetchone()[0] or 0.0
        
        # Get realized PnL from closed trades
        cursor.execute("""
            SELECT 
                COALESCE(SUM(pnl), 0) as realized_pnl,
                COUNT(*) as trade_count,
                COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN pnl < 0 THEN pnl END), 0) as avg_loss,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0) as win_rate
            FROM orders
            WHERE strftime('%Y', order_date) = strftime('%Y', 'now')
            AND status = 'filled'
            AND side = 'sell'
        """)
        row = cursor.fetchone()
        
        realized_pnl = row[0] or 0.0
        trade_count = row[1] or 0
        avg_win = row[2] or 0.0
        avg_loss = row[3] or 0.0
        win_rate = row[4] or 0.0
        
        # Total YTD P&L
        ytd_pnl = realized_pnl + unrealized_pnl
        
        # YTD target (configurable)
        ytd_target = float(os.getenv("YTD_TARGET", "50000"))
        
        conn.close()
        
        return {
            'ytd_pnl': round(ytd_pnl, 2),
            'ytd_target': ytd_target,
            'win_rate': round(win_rate, 1),
            'total_trades': trade_count,
            'avg_gain': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2)
        }
    
    except Exception as e:
        LOGGER.error(f"Failed to get YTD stats: {e}")
        return {
            'ytd_pnl': 0.0,
            'ytd_target': 50000.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'avg_gain': 0.0,
            'avg_loss': 0.0
        }
