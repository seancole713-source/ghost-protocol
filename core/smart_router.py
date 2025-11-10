"""
Stage 5: Smart Order Router
VWAP/TWAP Execution & Transaction Cost Analysis

Features: intelligent order routing, VWAP/TWAP algorithms, slippage estimation, TCA.
"""

import logging
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class SmartRouter:
    """
    Smart order routing with advanced execution algorithms.

    Features:
    - VWAP (Volume-Weighted Average Price) execution
    - TWAP (Time-Weighted Average Price) execution
    - Order splitting/slicing
    - Slippage estimation
    - Transaction cost analysis (TCA)
    """

    def __init__(self, db_path: str = "data/smart_router.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        LOGGER.info(f"Smart router initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for routing analytics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_plans (
                plan_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                total_quantity REAL NOT NULL,
                algorithm TEXT NOT NULL,

                -- Parameters
                duration_seconds INTEGER,
                num_slices INTEGER,

                -- Estimates
                estimated_slippage_bps REAL,
                estimated_cost REAL,

                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_slices (
                slice_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                slice_number INTEGER NOT NULL,
                quantity REAL NOT NULL,
                target_time TEXT NOT NULL,
                executed_at TEXT,
                executed_price REAL,
                slippage_bps REAL,

                FOREIGN KEY (plan_id) REFERENCES execution_plans(plan_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tca_reports (
                report_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                symbol TEXT NOT NULL,

                -- Execution quality
                arrival_price REAL NOT NULL,
                avg_execution_price REAL NOT NULL,
                slippage_bps REAL NOT NULL,

                -- Cost breakdown
                market_impact_cost REAL,
                timing_cost REAL,
                opportunity_cost REAL,
                total_cost REAL NOT NULL,

                created_at TEXT NOT NULL,

                FOREIGN KEY (plan_id) REFERENCES execution_plans(plan_id)
            )
        """)

        # Add indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_plans_symbol_time
            ON execution_plans(symbol, created_at DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_slices_plan
            ON execution_slices(plan_id, slice_number)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_slices_time
            ON execution_slices(target_time, executed_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tca_symbol_time
            ON tca_reports(symbol, created_at DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tca_performance
            ON tca_reports(slippage_bps ASC, total_cost ASC)
        """)

        conn.commit()
        conn.close()

    def create_vwap_plan(
        self,
        symbol: str,
        total_quantity: float,
        duration_minutes: int = 30,
        participation_rate: float = 0.10,
    ) -> dict:
        """
        Create VWAP (Volume-Weighted Average Price) execution plan.

        Strategy: Split order into slices that match market volume profile.

        Args:
            symbol: Trading symbol
            total_quantity: Total shares to execute
            duration_minutes: Time window for execution
            participation_rate: % of market volume to target (default 10%)

        Returns:
            Dict with execution plan
        """
        if total_quantity <= 0:
            return {"error": "Quantity must be positive"}

        if duration_minutes <= 0:
            return {"error": "Duration must be positive"}

        if participation_rate <= 0 or participation_rate > 0.5:
            return {"error": "Participation rate must be between 0 and 0.5"}

        # Estimate typical intraday volume profile (U-shape: high at open/close, low midday)
        num_slices = max(5, duration_minutes // 5)  # One slice every 5 minutes
        volume_profile = self._estimate_volume_profile(num_slices)

        # Allocate quantity based on volume profile
        slices = []
        for i in range(num_slices):
            slice_quantity = total_quantity * volume_profile[i]
            target_time = datetime.utcnow() + timedelta(minutes=i * (duration_minutes / num_slices))

            slices.append(
                {
                    "slice_number": i + 1,
                    "quantity": round(slice_quantity, 2),
                    "target_time": target_time.isoformat(),
                    "volume_weight": round(volume_profile[i], 4),
                }
            )

        # Estimate slippage
        estimated_slippage_bps = self._estimate_slippage(total_quantity, participation_rate)

        plan = {
            "plan_id": f"vwap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "symbol": symbol,
            "algorithm": "VWAP",
            "total_quantity": total_quantity,
            "duration_minutes": duration_minutes,
            "num_slices": num_slices,
            "participation_rate": participation_rate,
            "estimated_slippage_bps": round(estimated_slippage_bps, 2),
            "slices": slices,
        }

        self._record_execution_plan(plan)

        return plan

    def create_twap_plan(
        self, symbol: str, total_quantity: float, duration_minutes: int = 30
    ) -> dict:
        """
        Create TWAP (Time-Weighted Average Price) execution plan.

        Strategy: Split order into equal-sized slices over time.

        Args:
            symbol: Trading symbol
            total_quantity: Total shares to execute
            duration_minutes: Time window for execution

        Returns:
            Dict with execution plan
        """
        if total_quantity <= 0:
            return {"error": "Quantity must be positive"}

        if duration_minutes <= 0:
            return {"error": "Duration must be positive"}

        # Split into equal slices
        num_slices = max(5, duration_minutes // 5)  # One slice every 5 minutes
        slice_quantity = total_quantity / num_slices

        slices = []
        for i in range(num_slices):
            target_time = datetime.utcnow() + timedelta(minutes=i * (duration_minutes / num_slices))

            slices.append(
                {
                    "slice_number": i + 1,
                    "quantity": round(slice_quantity, 2),
                    "target_time": target_time.isoformat(),
                }
            )

        # TWAP has lower slippage than VWAP (more passive)
        estimated_slippage_bps = self._estimate_slippage(total_quantity, participation_rate=0.05)

        plan = {
            "plan_id": f"twap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "symbol": symbol,
            "algorithm": "TWAP",
            "total_quantity": total_quantity,
            "duration_minutes": duration_minutes,
            "num_slices": num_slices,
            "estimated_slippage_bps": round(estimated_slippage_bps, 2),
            "slices": slices,
        }

        self._record_execution_plan(plan)

        return plan

    def create_adaptive_plan(
        self,
        symbol: str,
        total_quantity: float,
        duration_minutes: int = 30,
        urgency: str = "medium",
    ) -> dict:
        """
        Create adaptive execution plan that adjusts to market conditions.

        Args:
            symbol: Trading symbol
            total_quantity: Total shares to execute
            duration_minutes: Time window for execution
            urgency: "low", "medium", or "high"

        Returns:
            Dict with execution plan
        """
        if urgency not in ["low", "medium", "high"]:
            return {"error": "Urgency must be low, medium, or high"}

        # Adjust participation rate based on urgency
        participation_rates = {
            "low": 0.05,  # 5% - very passive
            "medium": 0.10,  # 10% - balanced
            "high": 0.20,  # 20% - aggressive
        }

        participation_rate = participation_rates[urgency]

        # Use VWAP as base, but with urgency-adjusted slicing
        num_slices = max(
            3, duration_minutes // (10 if urgency == "low" else 5 if urgency == "medium" else 2)
        )

        volume_profile = self._estimate_volume_profile(num_slices)

        # Front-load slices if high urgency
        if urgency == "high":
            volume_profile = self._front_load_profile(volume_profile, factor=1.5)
        elif urgency == "low":
            volume_profile = self._back_load_profile(volume_profile, factor=1.3)

        slices = []
        for i in range(num_slices):
            slice_quantity = total_quantity * volume_profile[i]
            target_time = datetime.utcnow() + timedelta(minutes=i * (duration_minutes / num_slices))

            slices.append(
                {
                    "slice_number": i + 1,
                    "quantity": round(slice_quantity, 2),
                    "target_time": target_time.isoformat(),
                    "urgency_weight": round(volume_profile[i], 4),
                }
            )

        estimated_slippage_bps = self._estimate_slippage(total_quantity, participation_rate)

        plan = {
            "plan_id": f"adaptive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "symbol": symbol,
            "algorithm": "ADAPTIVE",
            "total_quantity": total_quantity,
            "duration_minutes": duration_minutes,
            "num_slices": num_slices,
            "urgency": urgency,
            "participation_rate": participation_rate,
            "estimated_slippage_bps": round(estimated_slippage_bps, 2),
            "slices": slices,
        }

        self._record_execution_plan(plan)

        return plan

    def estimate_slippage(
        self, symbol: str, quantity: float, current_price: float, avg_daily_volume: float = 1000000
    ) -> dict:
        """
        Estimate slippage for an order.

        Args:
            symbol: Trading symbol
            quantity: Order size
            current_price: Current market price
            avg_daily_volume: Average daily volume

        Returns:
            Dict with slippage estimates
        """
        # Market impact model: slippage = sigma * sqrt(quantity / avg_daily_volume)
        # where sigma is market impact coefficient

        sigma = 0.10  # 10 bps per sqrt(%) of daily volume

        volume_participation = quantity / avg_daily_volume

        # Base slippage in bps
        slippage_bps = sigma * math.sqrt(volume_participation) * 10000

        # Adjust for liquidity (tighter spreads = lower slippage)
        spread_bps = 5.0  # Assume 5 bps typical spread
        slippage_bps += spread_bps / 2  # Pay half the spread

        # Dollar cost
        slippage_dollars = (slippage_bps / 10000) * current_price * quantity

        return {
            "symbol": symbol,
            "quantity": quantity,
            "current_price": current_price,
            "volume_participation_pct": round(volume_participation * 100, 4),
            "estimated_slippage_bps": round(slippage_bps, 2),
            "estimated_slippage_dollars": round(slippage_dollars, 2),
            "confidence": "medium",
        }

    def generate_tca_report(
        self, plan_id: str, arrival_price: float, executed_slices: list[dict]
    ) -> dict:
        """
        Generate Transaction Cost Analysis (TCA) report.

        Args:
            plan_id: Execution plan ID
            arrival_price: Price when order was submitted
            executed_slices: List of executed slices with prices

        Returns:
            Dict with TCA metrics
        """
        if not executed_slices:
            return {"error": "No executed slices provided"}

        # Calculate average execution price
        total_quantity = sum(s["quantity"] for s in executed_slices)
        total_cost = sum(s["quantity"] * s["price"] for s in executed_slices)
        avg_execution_price = total_cost / total_quantity

        # Slippage vs arrival price (in bps)
        slippage_bps = ((avg_execution_price - arrival_price) / arrival_price) * 10000

        # Cost breakdown
        # 1. Market impact: immediate price movement
        market_impact_cost = abs(slippage_bps) * 0.6  # ~60% of slippage is market impact

        # 2. Timing cost: opportunity cost of waiting
        timing_cost = abs(slippage_bps) * 0.3  # ~30% is timing

        # 3. Opportunity cost: adverse price movement during execution
        opportunity_cost = abs(slippage_bps) * 0.1  # ~10% is opportunity cost

        total_cost = abs(slippage_bps)

        report = {
            "report_id": f"tca_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "plan_id": plan_id,
            "symbol": executed_slices[0].get("symbol", "UNKNOWN"),
            "arrival_price": round(arrival_price, 2),
            "avg_execution_price": round(avg_execution_price, 2),
            "slippage_bps": round(slippage_bps, 2),
            "cost_breakdown": {
                "market_impact_bps": round(market_impact_cost, 2),
                "timing_cost_bps": round(timing_cost, 2),
                "opportunity_cost_bps": round(opportunity_cost, 2),
                "total_cost_bps": round(total_cost, 2),
            },
            "execution_quality": self._classify_execution_quality(abs(slippage_bps)),
            "total_quantity": total_quantity,
            "num_slices": len(executed_slices),
            "created_at": datetime.utcnow().isoformat(),
        }

        self._record_tca_report(report)

        return report

    def _estimate_volume_profile(self, num_slices: int) -> list[float]:
        """Estimate intraday volume profile (U-shape)."""
        profile = []

        for i in range(num_slices):
            # U-shape: high at start, low in middle, high at end
            progress = i / (num_slices - 1) if num_slices > 1 else 0.5

            # Parabola: (progress - 0.5)^2
            u_factor = 1.0 - 2.0 * (progress - 0.5) ** 2

            # Normalize to [0.5, 1.5] range
            weight = 0.5 + u_factor * 0.5

            profile.append(weight)

        # Normalize to sum to 1.0
        total = sum(profile)
        profile = [w / total for w in profile]

        return profile

    def _front_load_profile(self, profile: list[float], factor: float = 1.5) -> list[float]:
        """Front-load execution (urgency)."""
        adjusted = []
        for i, weight in enumerate(profile):
            # Increase early weights, decrease later weights
            progress = i / len(profile)
            multiplier = factor - (factor - 1.0) * progress
            adjusted.append(weight * multiplier)

        # Renormalize
        total = sum(adjusted)
        return [w / total for w in adjusted]

    def _back_load_profile(self, profile: list[float], factor: float = 1.3) -> list[float]:
        """Back-load execution (patience)."""
        adjusted = []
        for i, weight in enumerate(profile):
            # Decrease early weights, increase later weights
            progress = i / len(profile)
            multiplier = 1.0 + (factor - 1.0) * progress
            adjusted.append(weight * multiplier)

        # Renormalize
        total = sum(adjusted)
        return [w / total for w in adjusted]

    def _estimate_slippage(self, quantity: float, participation_rate: float) -> float:
        """Estimate slippage in basis points."""
        # Simple market impact model
        # Slippage increases with square root of participation rate
        base_slippage = 10.0  # 10 bps for 10% participation

        slippage_bps = base_slippage * math.sqrt(participation_rate / 0.10)

        return slippage_bps

    def _classify_execution_quality(self, slippage_bps: float) -> str:
        """Classify execution quality based on slippage."""
        if slippage_bps < 5:
            return "Excellent"
        elif slippage_bps < 10:
            return "Good"
        elif slippage_bps < 20:
            return "Fair"
        elif slippage_bps < 50:
            return "Poor"
        else:
            return "Very Poor"

    def _record_execution_plan(self, plan: dict):
        """Record execution plan to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO execution_plans (
                    plan_id, symbol, total_quantity, algorithm, duration_seconds,
                    num_slices, estimated_slippage_bps, estimated_cost, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    plan["plan_id"],
                    plan["symbol"],
                    plan["total_quantity"],
                    plan["algorithm"],
                    plan.get("duration_minutes", 0) * 60,
                    plan["num_slices"],
                    plan["estimated_slippage_bps"],
                    0.0,
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record execution plan: {e}")

    def _record_tca_report(self, report: dict):
        """Record TCA report to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cost_breakdown = report["cost_breakdown"]

            cursor.execute(
                """
                INSERT INTO tca_reports (
                    report_id, plan_id, symbol, arrival_price, avg_execution_price,
                    slippage_bps, market_impact_cost, timing_cost, opportunity_cost,
                    total_cost, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    report["report_id"],
                    report["plan_id"],
                    report["symbol"],
                    report["arrival_price"],
                    report["avg_execution_price"],
                    report["slippage_bps"],
                    cost_breakdown["market_impact_bps"],
                    cost_breakdown["timing_cost_bps"],
                    cost_breakdown["opportunity_cost_bps"],
                    cost_breakdown["total_cost_bps"],
                    report["created_at"],
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record TCA report: {e}")


# Singleton instance
_smart_router: SmartRouter | None = None


def get_smart_router() -> SmartRouter:
    """Get singleton smart router instance."""
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartRouter()
    return _smart_router
