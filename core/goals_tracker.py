"""
Goals Tracker Module
Tracks daily, weekly, monthly, and yearly trading goals with progress.
"""

import json
import logging
import sqlite3
import time
from datetime import UTC, datetime, timedelta
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
        """Initialize goals database with both dollar and percentage targets."""
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
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    target_pct REAL DEFAULT NULL,
                    realized_pct REAL DEFAULT 0,
                    model_edge_pct REAL DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_goals_period
                ON goals(period, status)
            """)
            
            # Add new columns to existing tables if they don't exist
            try:
                conn.execute("ALTER TABLE goals ADD COLUMN target_pct REAL DEFAULT NULL")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                conn.execute("ALTER TABLE goals ADD COLUMN realized_pct REAL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            
            try:
                conn.execute("ALTER TABLE goals ADD COLUMN model_edge_pct REAL DEFAULT 0")
            except sqlite3.OperationalError:
                pass
    
    def set_goal(self, period: str, target_amount: float | None = None, target_pct: float | None = None) -> dict[str, Any]:
        """
        Set or update a goal for a specific period.
        
        Args:
            period: 'daily', 'weekly', 'monthly', or 'yearly'
            target_amount: Target profit amount in USD (optional)
            target_pct: Target percentage return (optional)
        
        Returns:
            Goal data with ID
        
        Note:
            At least one of target_amount or target_pct must be provided.
            If both provided, both are stored.
            If only % provided, dollar amount defaults to 0.
        """
        if target_amount is None and target_pct is None:
            raise ValueError("Must provide at least one of target_amount or target_pct")
        
        # Default values
        if target_amount is None:
            target_amount = 0.0
        if target_pct is None:
            target_pct = 0.0
        
        now = datetime.now(UTC)
        
        # Calculate end date based on period
        if period == "daily":
            end_date = now.replace(hour=23, minute=59, second=59)
        elif period == "weekly":
            days_until_sunday = (6 - now.weekday()) % 7
            # Add a delta rather than bumping the day to avoid month-end rollover errors
            end_date = (now + timedelta(days=days_until_sunday)).replace(
                hour=23, minute=59, second=59
            )
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
                INSERT INTO goals (period, target_amount, target_pct, start_date, end_date)
                VALUES (?, ?, ?, ?, ?)
            """, (period, target_amount, target_pct, now.isoformat(), end_date.isoformat()))
            
            goal_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        return {
            "id": goal_id,
            "period": period,
            "target_amount": target_amount,
            "target_pct": target_pct,
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
        Get all active goals with progress (both dollar and percentage).
        
        Returns:
            {
                "daily": {
                    "target": float,
                    "target_pct": float,
                    "current": float,
                    "realized_pct": float,
                    "model_edge_pct": float,
                    "progress_pct": float,
                    "progress_vs_pct_goal": float
                },
                ...
            }
        """
        result = {}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            for period in ["daily", "weekly", "monthly", "yearly"]:
                row = cursor.execute("""
                    SELECT target_amount, current_amount, target_pct, 
                           realized_pct, model_edge_pct
                    FROM goals
                    WHERE period = ? AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (period,)).fetchone()
                
                if row:
                    target = row["target_amount"]
                    current = row["current_amount"]
                    target_pct = row["target_pct"] or 0.0
                    realized_pct = row["realized_pct"] or 0.0
                    model_edge_pct = row["model_edge_pct"] or 0.0
                    
                    # Dollar progress
                    progress_pct = (current / target * 100) if target > 0 else 0
                    
                    # Percentage progress (if % goal set)
                    progress_vs_pct_goal = (realized_pct / target_pct * 100) if target_pct > 0 else 0
                    
                    result[period] = {
                        "target": target,
                        "target_pct": target_pct,
                        "current": current,
                        "realized_pct": round(realized_pct, 2),
                        "model_edge_pct": round(model_edge_pct, 2),
                        "progress_pct": round(progress_pct, 1),
                        "progress_vs_pct_goal": round(progress_vs_pct_goal, 1),
                        "remaining": target - current,
                        "remaining_pct": target_pct - realized_pct
                    }
                else:
                    result[period] = {
                        "target": 0,
                        "target_pct": 0,
                        "current": 0,
                        "realized_pct": 0,
                        "model_edge_pct": 0,
                        "progress_pct": 0,
                        "progress_vs_pct_goal": 0,
                        "remaining": 0,
                        "remaining_pct": 0
                    }
        
        return result
    
    def compute_model_performance(self, period: str) -> float:
        """
        Compute model-implied % performance from Ghost's predictions.
        
        This approximates the theoretical edge Ghost provides by analyzing:
        - Win rate (% of correct predictions)
        - Average predicted move for winners
        - Model edge = avg_winner_move * win_rate
        
        Args:
            period: 'daily', 'weekly', 'monthly', 'yearly'
        
        Returns:
            Model-implied percentage return
        """
        try:
            from core.prediction_tracker import calculate_accuracy
        except ImportError:
            logger.warning("Cannot import prediction_tracker, using 0% model edge")
            return 0.0
        
        # Map periods to days
        period_days = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "yearly": 365
        }
        
        days = period_days.get(period, 30)
        
        # Get accuracy for this timeframe
        if days == 1:
            stats = calculate_accuracy("24h")
        elif days == 7:
            stats = calculate_accuracy("7d")
        elif days == 30:
            stats = calculate_accuracy("30d")
        else:
            stats = calculate_accuracy("all")
        
        total = stats.get("total_predictions", 0)
        correct = stats.get("correct_predictions", 0)
        
        if total == 0:
            return 0.0
        
        # Win rate
        win_rate = correct / total
        
        # Get winning predictions and calculate average predicted move
        predictions = stats.get("predictions", [])
        winning_moves = [
            abs(p.get("predicted_pct", 0))
            for p in predictions
            if p.get("correct", False)
        ]
        
        if not winning_moves:
            return 0.0
        
        avg_winner_move = sum(winning_moves) / len(winning_moves)
        
        # Model edge = win_rate * avg_winner_move
        # This is a conservative estimate since it doesn't account for position sizing
        model_edge = win_rate * avg_winner_move
        
        logger.info(
            f"Model performance ({period}): {model_edge:.2f}% "
            f"(win_rate={win_rate:.2%}, avg_move={avg_winner_move:.2f}%)"
        )
        
        return model_edge
    
    def update_model_performance(self, period: str):
        """Update the model_edge_pct for a period's active goal."""
        edge_pct = self.compute_model_performance(period)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE goals
                SET model_edge_pct = ?, updated_at = CURRENT_TIMESTAMP
                WHERE period = ? AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            """, (edge_pct, period))
        
        logger.info(f"Updated {period} goal model_edge_pct = {edge_pct:.2f}%")

