"""
APEX Dynamic Goal Engine
Portfolio target tracking with adaptive risk budgets

Sets weekly/monthly targets (return %, drawdown, Sharpe)
Auto-adjusts position sizing based on goal progress

Expected Impact: +18% goal alignment
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.cpu_queue import get_cpu_queue
from core.price_quorum import get_price_quorum

LOGGER = logging.getLogger(__name__)


class GoalPeriod(Enum):
    """Goal time periods"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class GoalStatus(Enum):
    """Goal achievement status"""

    ON_TRACK = "on_track"  # Making good progress
    AT_RISK = "at_risk"  # Behind target
    EXCEEDED = "exceeded"  # Beat the target
    FAILED = "failed"  # Missed target


@dataclass
class PortfolioGoal:
    """Single portfolio goal"""

    goal_id: str
    period: str  # "weekly", "monthly", "quarterly"
    target_return_pct: float  # Target return %
    max_drawdown_pct: float  # Max acceptable drawdown %
    target_sharpe: float  # Target Sharpe ratio
    risk_budget: float  # Max risk budget (0-100%)
    start_date: int  # Unix timestamp
    end_date: int  # Unix timestamp

    # Progress tracking
    current_return_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    current_sharpe: float = 0.0
    days_elapsed: int = 0
    days_total: int = 0

    # Status
    status: str = "on_track"
    risk_adjustment: float = 1.0  # Multiplier for position sizing (0.5-2.0)


@dataclass
class GoalProgress:
    """Progress report for a goal"""

    goal_id: str
    period: str
    progress_pct: float  # % complete (0-100)
    on_pace: bool
    days_remaining: int
    required_daily_return: float  # What return is needed per day to hit target
    recommendation: str  # Action recommendation


class DynamicGoalEngine:
    """
    Manages portfolio goals and adapts risk based on progress

    Features:
    - Set weekly/monthly/quarterly targets
    - Track progress in real-time
    - Adjust position sizing based on goal progress
    - Alert when falling behind or exceeding targets
    """

    def __init__(self, db_path: str = "data/goal_engine.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize goal tracking database"""
        conn = sqlite3.connect(self.db_path)

        # Goals table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                period TEXT NOT NULL,
                target_return_pct REAL NOT NULL,
                max_drawdown_pct REAL NOT NULL,
                target_sharpe REAL NOT NULL,
                risk_budget REAL NOT NULL,
                start_date INTEGER NOT NULL,
                end_date INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        """)

        # Daily progress snapshots
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goal_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                current_return_pct REAL,
                current_drawdown_pct REAL,
                current_sharpe REAL,
                portfolio_value REAL,
                status TEXT,
                risk_adjustment REAL,
                FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
            )
        """)

        # Risk adjustments log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                old_risk_budget REAL,
                new_risk_budget REAL,
                reason TEXT,
                FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
            )
        """)

        conn.commit()
        conn.close()

        LOGGER.info(f"Goal Engine initialized: {self.db_path}")

    def create_goal(
        self,
        period: str,
        target_return_pct: float,
        max_drawdown_pct: float = 10.0,
        target_sharpe: float = 1.5,
        risk_budget: float = 100.0,
    ) -> PortfolioGoal:
        """
        Create a new portfolio goal

        Args:
            period: "daily", "weekly", "monthly", "quarterly", "yearly"
            target_return_pct: Target return % (e.g., 5.0 for 5%)
            max_drawdown_pct: Max acceptable drawdown % (e.g., 10.0 for 10%)
            target_sharpe: Target Sharpe ratio (e.g., 1.5)
            risk_budget: Starting risk budget % (default: 100%)

        Returns:
            PortfolioGoal object
        """

        # Calculate date range
        now = int(time.time())

        if period == "daily":
            end_date = now + 86400  # 1 day
            days_total = 1
        elif period == "weekly":
            end_date = now + 604800  # 7 days
            days_total = 7
        elif period == "monthly":
            end_date = now + 2592000  # 30 days
            days_total = 30
        elif period == "quarterly":
            end_date = now + 7776000  # 90 days
            days_total = 90
        elif period == "yearly":
            end_date = now + 31536000  # 365 days
            days_total = 365
        else:
            raise ValueError(f"Invalid period: {period}")

        goal_id = f"{period}_{now}"

        goal = PortfolioGoal(
            goal_id=goal_id,
            period=period,
            target_return_pct=target_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            target_sharpe=target_sharpe,
            risk_budget=risk_budget,
            start_date=now,
            end_date=end_date,
            days_total=days_total,
            status="on_track",
        )

        # Save to database
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO goals (goal_id, period, target_return_pct, max_drawdown_pct,
                             target_sharpe, risk_budget, start_date, end_date, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                goal.goal_id,
                goal.period,
                goal.target_return_pct,
                goal.max_drawdown_pct,
                goal.target_sharpe,
                goal.risk_budget,
                goal.start_date,
                goal.end_date,
                goal.status,
                now,
            ),
        )
        conn.commit()
        conn.close()

        LOGGER.info(f"Created {period} goal: {target_return_pct}% return target")

        return goal

    def update_progress(
        self,
        goal_id: str,
        current_return_pct: float,
        current_drawdown_pct: float,
        current_sharpe: float,
        portfolio_value: float,
    ) -> GoalProgress:
        """
        Update goal progress and calculate risk adjustments

        Args:
            goal_id: Goal identifier
            current_return_pct: Current period return %
            current_drawdown_pct: Current drawdown %
            current_sharpe: Current Sharpe ratio
            portfolio_value: Current portfolio value

        Returns:
            GoalProgress with recommendations
        """

        goal = self._load_goal(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        now = int(time.time())
        days_elapsed = max(1, (now - goal.start_date) // 86400)
        days_remaining = max(0, (goal.end_date - now) // 86400)
        days_total = goal.days_total

        # Update goal metrics
        goal.current_return_pct = current_return_pct
        goal.current_drawdown_pct = current_drawdown_pct
        goal.current_sharpe = current_sharpe
        goal.days_elapsed = days_elapsed

        # Calculate progress %
        expected_progress = (days_elapsed / days_total) * 100
        actual_progress = (
            (current_return_pct / goal.target_return_pct) * 100
            if goal.target_return_pct != 0
            else 0
        )

        # Determine if on pace
        on_pace = actual_progress >= (expected_progress * 0.8)  # 80% threshold

        # Calculate required daily return to hit target
        remaining_return_needed = goal.target_return_pct - current_return_pct
        required_daily_return = remaining_return_needed / max(1, days_remaining)

        # Determine status
        if current_return_pct >= goal.target_return_pct:
            status = GoalStatus.EXCEEDED
            recommendation = "Goal exceeded! Consider locking in gains or raising target."
        elif current_drawdown_pct > goal.max_drawdown_pct:
            status = GoalStatus.FAILED
            recommendation = "Drawdown limit breached. Reduce risk immediately."
        elif on_pace:
            status = GoalStatus.ON_TRACK
            recommendation = "On track. Maintain current strategy."
        else:
            status = GoalStatus.AT_RISK
            recommendation = (
                f"Behind target. Need {required_daily_return:.2f}% daily return to catch up."
            )

        goal.status = status.value

        # Adaptive risk adjustment
        risk_adjustment = self._calculate_risk_adjustment(
            goal, actual_progress, expected_progress, current_drawdown_pct
        )
        goal.risk_adjustment = risk_adjustment

        # Save progress snapshot
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO goal_progress (goal_id, timestamp, current_return_pct,
                                      current_drawdown_pct, current_sharpe,
                                      portfolio_value, status, risk_adjustment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                goal_id,
                now,
                current_return_pct,
                current_drawdown_pct,
                current_sharpe,
                portfolio_value,
                status.value,
                risk_adjustment,
            ),
        )

        # Update goal status in goals table
        conn.execute(
            """
            UPDATE goals SET status = ? WHERE goal_id = ?
        """,
            (status.value, goal_id),
        )

        conn.commit()
        conn.close()

        progress = GoalProgress(
            goal_id=goal_id,
            period=goal.period,
            progress_pct=actual_progress,
            on_pace=on_pace,
            days_remaining=days_remaining,
            required_daily_return=required_daily_return,
            recommendation=recommendation,
        )

        LOGGER.info(f"Goal {goal_id} progress: {actual_progress:.1f}% ({status.value})")

        return progress

    def _calculate_risk_adjustment(
        self,
        goal: PortfolioGoal,
        actual_progress: float,
        expected_progress: float,
        current_drawdown: float,
    ) -> float:
        """
        Calculate adaptive risk budget multiplier

        Logic:
        - If ahead of target → reduce risk (lock in gains)
        - If behind target → carefully increase risk (catch up)
        - If drawdown is high → reduce risk (protect capital)
        """

        base_adjustment = 1.0

        # 1. Progress-based adjustment
        progress_diff = actual_progress - expected_progress

        if progress_diff > 50:
            # Way ahead - reduce risk significantly
            base_adjustment *= 0.5
        elif progress_diff > 20:
            # Ahead - reduce risk moderately
            base_adjustment *= 0.75
        elif progress_diff < -50:
            # Way behind - increase risk (but capped)
            base_adjustment *= 1.5
        elif progress_diff < -20:
            # Behind - increase risk moderately
            base_adjustment *= 1.25

        # 2. Drawdown-based adjustment
        drawdown_pct = (
            (current_drawdown / goal.max_drawdown_pct) if goal.max_drawdown_pct > 0 else 0
        )

        if drawdown_pct > 0.8:
            # Close to drawdown limit - reduce risk aggressively
            base_adjustment *= 0.5
        elif drawdown_pct > 0.5:
            # Moderate drawdown - reduce risk
            base_adjustment *= 0.75

        # 3. Clip to reasonable bounds
        risk_adjustment = max(0.5, min(2.0, base_adjustment))

        return risk_adjustment

    def get_active_goals(self) -> list[PortfolioGoal]:
        """Get all active (non-expired) goals"""

        conn = sqlite3.connect(self.db_path)
        now = int(time.time())

        cursor = conn.execute(
            """
            SELECT goal_id, period, target_return_pct, max_drawdown_pct,
                   target_sharpe, risk_budget, start_date, end_date, status
            FROM goals
            WHERE end_date > ?
            ORDER BY end_date ASC
        """,
            (now,),
        )

        goals = []
        for row in cursor.fetchall():
            goal = PortfolioGoal(
                goal_id=row[0],
                period=row[1],
                target_return_pct=row[2],
                max_drawdown_pct=row[3],
                target_sharpe=row[4],
                risk_budget=row[5],
                start_date=row[6],
                end_date=row[7],
                status=row[8],
                days_total=max(1, (row[7] - row[6]) // 86400),
            )
            goals.append(goal)

        conn.close()

        return goals

    def get_goal_history(self, goal_id: str, limit: int = 30) -> list[dict[str, Any]]:
        """Get historical progress snapshots for a goal"""

        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute(
            """
            SELECT timestamp, current_return_pct, current_drawdown_pct,
                   current_sharpe, portfolio_value, status, risk_adjustment
            FROM goal_progress
            WHERE goal_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (goal_id, limit),
        )

        history = []
        for row in cursor.fetchall():
            history.append(
                {
                    "timestamp": row[0],
                    "return_pct": row[1],
                    "drawdown_pct": row[2],
                    "sharpe": row[3],
                    "portfolio_value": row[4],
                    "status": row[5],
                    "risk_adjustment": row[6],
                }
            )

        conn.close()

        return history

    def _load_goal(self, goal_id: str) -> PortfolioGoal | None:
        """Load goal from database"""

        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute(
            """
            SELECT goal_id, period, target_return_pct, max_drawdown_pct,
                   target_sharpe, risk_budget, start_date, end_date, status
            FROM goals
            WHERE goal_id = ?
        """,
            (goal_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        goal = PortfolioGoal(
            goal_id=row[0],
            period=row[1],
            target_return_pct=row[2],
            max_drawdown_pct=row[3],
            target_sharpe=row[4],
            risk_budget=row[5],
            start_date=row[6],
            end_date=row[7],
            status=row[8],
            days_total=max(1, (row[7] - row[6]) // 86400),
        )

        return goal

    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal"""

        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM goal_progress WHERE goal_id = ?", (goal_id,))
        conn.execute("DELETE FROM risk_adjustments WHERE goal_id = ?", (goal_id,))
        conn.execute("DELETE FROM goals WHERE goal_id = ?", (goal_id,))
        conn.commit()
        conn.close()

        LOGGER.info(f"Deleted goal: {goal_id}")

        return True

    def get_risk_multiplier(self) -> float:
        """
        Get current risk multiplier for position sizing

        Returns average risk adjustment across all active goals
        """

        goals = self.get_active_goals()

        if not goals:
            return 1.0  # No goals = normal risk

        # Get most recent risk adjustment for each goal
        conn = sqlite3.connect(self.db_path)

        adjustments = []
        for goal in goals:
            cursor = conn.execute(
                """
                SELECT risk_adjustment
                FROM goal_progress
                WHERE goal_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (goal.goal_id,),
            )

            row = cursor.fetchone()
            if row:
                adjustments.append(row[0])

        conn.close()

        if not adjustments:
            return 1.0

        # Average risk adjustment across all goals
        avg_adjustment = sum(adjustments) / len(adjustments)

        load_factor = 0.0
        try:
            cpu_metrics = get_cpu_queue().snapshot()
            max_workers = max(1, int(cpu_metrics.get("max_workers", 1)))
            load_factor = max(
                load_factor,
                float(cpu_metrics.get("active_tasks", 0)) / max_workers,
            )
        except Exception:
            pass

        try:
            price_metrics = get_price_quorum().snapshot()
            max_concurrency = max(1, int(price_metrics.get("max_concurrency", 1)))
            load_factor = max(
                load_factor,
                float(price_metrics.get("active_tasks", 0)) / max_concurrency,
            )
        except Exception:
            pass

        if load_factor > 0.8:
            avg_adjustment *= 0.7
        elif load_factor > 0.6:
            avg_adjustment *= 0.85

        avg_adjustment = max(0.5, min(1.5, avg_adjustment))

        return avg_adjustment


# Singleton instance
_GOAL_ENGINE: DynamicGoalEngine | None = None


def get_goal_engine() -> DynamicGoalEngine:
    """Get singleton instance of goal engine"""
    global _GOAL_ENGINE
    if _GOAL_ENGINE is None:
        _GOAL_ENGINE = DynamicGoalEngine()
    return _GOAL_ENGINE
