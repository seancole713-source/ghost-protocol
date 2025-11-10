"""
Stage 5: Execution Analytics
Real-time Execution Performance Monitoring

Features: fill quality metrics, latency tracking, performance dashboards, execution reports.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class ExecutionAnalytics:
    """
    Real-time execution analytics and performance monitoring.

    Features:
    - Fill quality metrics
    - Latency monitoring
    - Execution performance dashboards
    - Historical execution reports
    """

    def __init__(self, db_path: str = "data/execution_analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        LOGGER.info(f"Execution analytics initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for execution analytics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS execution_metrics (
                metric_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,

                -- Timing
                order_created_at TEXT NOT NULL,
                order_submitted_at TEXT NOT NULL,
                first_fill_at TEXT NOT NULL,
                last_fill_at TEXT NOT NULL,

                -- Latency (milliseconds)
                submission_latency_ms INTEGER,
                execution_latency_ms INTEGER,
                total_latency_ms INTEGER,

                -- Fill quality
                num_fills INTEGER NOT NULL,
                avg_fill_price REAL NOT NULL,
                price_improvement_bps REAL,
                fill_rate REAL NOT NULL,

                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,

                -- Volume stats
                total_orders INTEGER DEFAULT 0,
                total_filled INTEGER DEFAULT 0,
                total_cancelled INTEGER DEFAULT 0,
                total_volume REAL DEFAULT 0.0,

                -- Performance stats
                avg_latency_ms INTEGER DEFAULT 0,
                avg_fill_rate REAL DEFAULT 0.0,
                avg_slippage_bps REAL DEFAULT 0.0,

                -- Quality score (0-100)
                execution_quality_score REAL DEFAULT 0.0,

                updated_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def record_execution_metrics(
        self,
        order_id: str,
        symbol: str,
        order_created_at: str,
        order_submitted_at: str,
        fills: list[dict],
    ) -> dict:
        """
        Record execution metrics for an order.

        Args:
            order_id: Order ID
            symbol: Trading symbol
            order_created_at: ISO timestamp when order created
            order_submitted_at: ISO timestamp when order submitted
            fills: List of fill dictionaries

        Returns:
            Dict with execution metrics
        """
        if not fills:
            return {"error": "No fills provided"}

        # Parse timestamps
        created = datetime.fromisoformat(order_created_at.replace("Z", "+00:00"))
        submitted = datetime.fromisoformat(order_submitted_at.replace("Z", "+00:00"))

        first_fill = datetime.fromisoformat(fills[0]["filled_at"].replace("Z", "+00:00"))
        last_fill = datetime.fromisoformat(fills[-1]["filled_at"].replace("Z", "+00:00"))

        # Calculate latencies (in milliseconds)
        submission_latency_ms = int((submitted - created).total_seconds() * 1000)
        execution_latency_ms = int((first_fill - submitted).total_seconds() * 1000)
        total_latency_ms = int((last_fill - created).total_seconds() * 1000)

        # Calculate fill metrics
        total_quantity = sum(f["quantity"] for f in fills)
        total_cost = sum(f["quantity"] * f["price"] for f in fills)
        avg_fill_price = total_cost / total_quantity if total_quantity > 0 else 0.0

        # Price improvement (compare to first fill price)
        reference_price = fills[0]["price"]
        price_improvement_bps = ((reference_price - avg_fill_price) / reference_price) * 10000

        # Fill rate (assume target was met)
        fill_rate = 1.0  # 100% filled

        metrics = {
            "metric_id": f"exec_{order_id}",
            "order_id": order_id,
            "symbol": symbol,
            "order_created_at": order_created_at,
            "order_submitted_at": order_submitted_at,
            "first_fill_at": fills[0]["filled_at"],
            "last_fill_at": fills[-1]["filled_at"],
            "submission_latency_ms": submission_latency_ms,
            "execution_latency_ms": execution_latency_ms,
            "total_latency_ms": total_latency_ms,
            "num_fills": len(fills),
            "avg_fill_price": round(avg_fill_price, 4),
            "price_improvement_bps": round(price_improvement_bps, 2),
            "fill_rate": fill_rate,
        }

        self._record_metrics(metrics)
        self._update_daily_stats(metrics)

        return metrics

    def get_execution_dashboard(self, lookback_days: int = 7) -> dict:
        """
        Get execution performance dashboard.

        Args:
            lookback_days: Number of days to analyze

        Returns:
            Dict with dashboard metrics
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get recent metrics
            cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()

            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_orders,
                    AVG(total_latency_ms) as avg_latency,
                    AVG(fill_rate) as avg_fill_rate,
                    AVG(price_improvement_bps) as avg_price_improvement
                FROM execution_metrics
                WHERE DATE(created_at) >= ?
            """,
                (cutoff_date,),
            )

            row = cursor.fetchone()

            if row[0] == 0:  # No data
                conn.close()
                return {
                    "lookback_days": lookback_days,
                    "total_orders": 0,
                    "avg_latency_ms": 0,
                    "avg_fill_rate_pct": 0.0,
                    "avg_price_improvement_bps": 0.0,
                    "execution_quality": "No Data",
                }

            total_orders = row[0]
            avg_latency = row[1] or 0
            avg_fill_rate = row[2] or 0.0
            avg_price_improvement = row[3] or 0.0

            # Calculate quality score (0-100)
            # Components: latency (30%), fill rate (40%), price improvement (30%)
            latency_score = max(0, 100 - (avg_latency / 10))  # Penalty for high latency
            fill_rate_score = avg_fill_rate * 100
            price_improvement_score = min(100, max(0, 50 + avg_price_improvement))  # Center at 50

            quality_score = (
                latency_score * 0.3 + fill_rate_score * 0.4 + price_improvement_score * 0.3
            )

            # Classify quality
            if quality_score >= 90:
                quality = "Excellent"
            elif quality_score >= 75:
                quality = "Good"
            elif quality_score >= 60:
                quality = "Fair"
            else:
                quality = "Poor"

            conn.close()

            return {
                "lookback_days": lookback_days,
                "total_orders": total_orders,
                "avg_latency_ms": round(avg_latency, 0),
                "avg_fill_rate_pct": round(avg_fill_rate * 100, 2),
                "avg_price_improvement_bps": round(avg_price_improvement, 2),
                "execution_quality_score": round(quality_score, 1),
                "execution_quality": quality,
            }
        except Exception as e:
            LOGGER.error(f"Failed to get execution dashboard: {e}")
            return {"error": str(e)}

    def get_latency_distribution(self, lookback_days: int = 7) -> dict:
        """Get latency distribution statistics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()

            cursor.execute(
                """
                SELECT total_latency_ms
                FROM execution_metrics
                WHERE DATE(created_at) >= ?
            """,
                (cutoff_date,),
            )

            latencies = [row[0] for row in cursor.fetchall()]
            conn.close()

            if not latencies:
                return {"error": "No data available"}

            latencies_array = np.array(latencies)

            return {
                "min_ms": int(np.min(latencies_array)),
                "p50_ms": int(np.percentile(latencies_array, 50)),
                "p95_ms": int(np.percentile(latencies_array, 95)),
                "p99_ms": int(np.percentile(latencies_array, 99)),
                "max_ms": int(np.max(latencies_array)),
                "mean_ms": round(float(np.mean(latencies_array)), 1),
                "std_ms": round(float(np.std(latencies_array)), 1),
            }
        except Exception as e:
            LOGGER.error(f"Failed to get latency distribution: {e}")
            return {"error": str(e)}

    def get_daily_report(self, date: str | None = None) -> dict:
        """Get daily execution report."""
        if date is None:
            date = datetime.utcnow().date().isoformat()

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (date,))
            row = cursor.fetchone()

            if row is None:
                conn.close()
                return {"error": f"No data for {date}"}

            columns = [desc[0] for desc in cursor.description]
            report = dict(zip(columns, row, strict=False))

            conn.close()
            return report
        except Exception as e:
            LOGGER.error(f"Failed to get daily report: {e}")
            return {"error": str(e)}

    def _record_metrics(self, metrics: dict):
        """Record execution metrics to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO execution_metrics (
                    metric_id, order_id, symbol, order_created_at, order_submitted_at,
                    first_fill_at, last_fill_at, submission_latency_ms, execution_latency_ms,
                    total_latency_ms, num_fills, avg_fill_price, price_improvement_bps,
                    fill_rate, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics["metric_id"],
                    metrics["order_id"],
                    metrics["symbol"],
                    metrics["order_created_at"],
                    metrics["order_submitted_at"],
                    metrics["first_fill_at"],
                    metrics["last_fill_at"],
                    metrics["submission_latency_ms"],
                    metrics["execution_latency_ms"],
                    metrics["total_latency_ms"],
                    metrics["num_fills"],
                    metrics["avg_fill_price"],
                    metrics["price_improvement_bps"],
                    metrics["fill_rate"],
                    datetime.utcnow().isoformat(),
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record metrics: {e}")

    def _update_daily_stats(self, metrics: dict):
        """Update daily statistics."""
        try:
            date = datetime.utcnow().date().isoformat()

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get or create daily stats
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (date,))
            row = cursor.fetchone()

            if row is None:
                # Create new entry
                cursor.execute(
                    """
                    INSERT INTO daily_stats (
                        date, total_orders, avg_latency_ms, avg_fill_rate,
                        avg_slippage_bps, updated_at
                    ) VALUES (?, 1, ?, ?, 0.0, ?)
                """,
                    (
                        date,
                        metrics["total_latency_ms"],
                        metrics["fill_rate"],
                        datetime.utcnow().isoformat(),
                    ),
                )
            else:
                # Update existing entry (running average)
                columns = [desc[0] for desc in cursor.description]
                stats = dict(zip(columns, row, strict=False))

                n = stats["total_orders"]
                new_avg_latency = (stats["avg_latency_ms"] * n + metrics["total_latency_ms"]) / (
                    n + 1
                )
                new_avg_fill_rate = (stats["avg_fill_rate"] * n + metrics["fill_rate"]) / (n + 1)

                cursor.execute(
                    """
                    UPDATE daily_stats
                    SET total_orders = total_orders + 1,
                        avg_latency_ms = ?,
                        avg_fill_rate = ?,
                        updated_at = ?
                    WHERE date = ?
                """,
                    (new_avg_latency, new_avg_fill_rate, datetime.utcnow().isoformat(), date),
                )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to update daily stats: {e}")


# Singleton instance
_execution_analytics: ExecutionAnalytics | None = None


def get_execution_analytics() -> ExecutionAnalytics:
    """Get singleton execution analytics instance."""
    global _execution_analytics
    if _execution_analytics is None:
        _execution_analytics = ExecutionAnalytics()
    return _execution_analytics
