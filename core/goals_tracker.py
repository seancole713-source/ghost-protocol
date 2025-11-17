"""
Goals Tracker Module
Tracks daily, weekly, monthly, and yearly trading goals with progress.
"""

import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "goals.db"


class GoalsTracker:
    """Tracks user-defined financial goals and progress."""
    
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()
    
    def _init_db(self):
        """Initialize goals database."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period TEXT NOT NULL,
                    target_amount REAL NOT NULL,
                    current_amount REAL DEFAULT 0,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goals_period
                ON goals(period, status)
            """)
    
    def set_goal(self, period: str, target_amount: float) -> dict[str, Any]:
        """
        Set or update a goal for a specific period.
        
        Args:
            period: 'daily', 'weekly', 'monthly', or 'yearly'
            target_amount: Target profit amount in USD
        
        Returns:
            Goal data with ID
        """
        now = datetime.now(UTC)
        
        # Calculate end date based on period
        if period == "daily":
            end_date = now.replace(hour=23, minute=59, second=59)
        elif period == "weekly":
            days_until_sunday = (6 - now.weekday()) % 7
            end_date = now.replace(hour=23, minute=59, second=59)
            end_date = end_date.replace(day=end_date.day + days_until_sunday)
        elif period == "monthly":
            # Last day of current month
            if now.month == 12:
                end_date = now.replace(year=now.year + 1, month=1, day=1)
            else:
                end_date = now.replace(month=now.month + 1, day=1)
            end_date = end_date.replace(hour=23, minute=59, second=59)
        elif period == "yearly":
            end_date = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0)
        else:
            raise ValueError(f"Invalid period: {period}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO goals (period, target_amount, start_date, end_date)
                VALUES (?, ?, ?, ?)
            """, (period, target_amount, now.isoformat(), end_date.isoformat()))
            
            goal_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        return {
            "id": goal_id,
            "period": period,
            "target_amount": target_amount,
            "start_date": now.isoformat(),
            "end_date": end_date.isoformat()
        }
    
    def update_progress(self, period: str, amount: float):
        """Update current progress for a goal period."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE goals
                SET current_amount = ?, updated_at = CURRENT_TIMESTAMP
                WHERE period = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            """, (amount, period))
    
    def get_all_goals(self) -> dict[str, Any]:
        """
        Get all active goals with progress.
        
        Returns:
            {
                "daily": {"target": float, "current": float, "progress_pct": float},
                "weekly": {...},
                "monthly": {...},
                "yearly": {...}
            }
        """
        result = {}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            for period in ["daily", "weekly", "monthly", "yearly"]:
                row = cursor.execute("""
                    SELECT target_amount, current_amount
                    FROM goals
                    WHERE period = ? AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (period,)).fetchone()
                
                if row:
                    target = row["target_amount"]
                    current = row["current_amount"]
                    progress_pct = (current / target * 100) if target > 0 else 0
                    
                    result[period] = {
                        "target": target,
                        "current": current,
                        "progress_pct": round(progress_pct, 1),
                        "remaining": target - current
                    }
                else:
                    result[period] = {
                        "target": 0,
                        "current": 0,
                        "progress_pct": 0,
                        "remaining": 0
                    }
        
        return result
