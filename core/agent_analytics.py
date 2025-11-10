"""
Agent Analytics Module
Provides decision quality metrics, confidence analysis, and performance tracking.
"""

import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass
class DecisionStats:
    """Statistics about AI decisions over a time period."""

    total_decisions: int
    unique_symbols: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    action_distribution: dict[str, int]
    decision_type_distribution: dict[str, int]
    symbols_tracked: list[str]
    timespan_hours: int
    oldest_decision_ts: str | None
    newest_decision_ts: str | None


@dataclass
class SymbolPerformance:
    """Performance metrics for a specific symbol."""

    symbol: str
    decision_count: int
    avg_confidence: float
    most_common_action: str
    action_distribution: dict[str, int]
    last_decision_ts: str
    risk_mentions: int


@dataclass
class ToolCallMetrics:
    """Metrics for tool invocations."""

    tool_name: str
    total_calls: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    success_rate: float
    last_called_ts: str | None
    most_used_symbol: str | None
    total_data_bytes: int


class AgentAnalytics:
    """Analytics engine for Ghost agent decisions and performance."""

    def __init__(self, db_path: str = "./data/ghost_agent.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_decision_stats(self, hours: int = 168) -> DecisionStats:
        """
        Compute aggregate statistics for decisions within time window.

        Args:
            hours: Lookback period in hours (default: 7 days)

        Returns:
            DecisionStats object with computed metrics
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()

            # Total count and confidence stats
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    AVG(confidence) as avg_conf,
                    MIN(confidence) as min_conf,
                    MAX(confidence) as max_conf,
                    MIN(created_ts) as oldest,
                    MAX(created_ts) as newest
                FROM ai_decisions
                WHERE created_ts >= ?
            """,
                (cutoff,),
            )

            row = cur.fetchone()
            total = row["total"]
            unique_symbols = row["unique_symbols"]
            avg_conf = row["avg_conf"] or 0.0
            min_conf = row["min_conf"] or 0.0
            max_conf = row["max_conf"] or 0.0
            oldest = row["oldest"]
            newest = row["newest"]

            # Action distribution
            cur.execute(
                """
                SELECT action, COUNT(*) as count
                FROM ai_decisions
                WHERE created_ts >= ?
                GROUP BY action
                ORDER BY count DESC
            """,
                (cutoff,),
            )

            action_dist = {row["action"]: row["count"] for row in cur.fetchall()}

            # Decision type distribution
            cur.execute(
                """
                SELECT decision_type, COUNT(*) as count
                FROM ai_decisions
                WHERE created_ts >= ?
                GROUP BY decision_type
                ORDER BY count DESC
            """,
                (cutoff,),
            )

            type_dist = {row["decision_type"]: row["count"] for row in cur.fetchall()}

            # Symbols tracked
            cur.execute(
                """
                SELECT DISTINCT symbol
                FROM ai_decisions
                WHERE created_ts >= ?
                ORDER BY symbol
            """,
                (cutoff,),
            )

            symbols = [row["symbol"] for row in cur.fetchall()]

            return DecisionStats(
                total_decisions=total,
                unique_symbols=unique_symbols,
                avg_confidence=round(avg_conf, 3),
                min_confidence=round(min_conf, 3),
                max_confidence=round(max_conf, 3),
                action_distribution=action_dist,
                decision_type_distribution=type_dist,
                symbols_tracked=symbols,
                timespan_hours=hours,
                oldest_decision_ts=oldest,
                newest_decision_ts=newest,
            )

    def get_symbol_performance(self, symbol: str, hours: int = 168) -> SymbolPerformance | None:
        """
        Get performance metrics for a specific symbol.

        Args:
            symbol: Ticker symbol to analyze
            hours: Lookback period

        Returns:
            SymbolPerformance object or None if no decisions found
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()

            # Basic stats
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    AVG(confidence) as avg_conf,
                    MAX(created_ts) as last_ts
                FROM ai_decisions
                WHERE symbol = ? AND created_ts >= ?
            """,
                (symbol, cutoff),
            )

            row = cur.fetchone()
            if row["total"] == 0:
                return None

            total = row["total"]
            avg_conf = row["avg_conf"]
            last_ts = row["last_ts"]

            # Action distribution
            cur.execute(
                """
                SELECT action, COUNT(*) as count
                FROM ai_decisions
                WHERE symbol = ? AND created_ts >= ?
                GROUP BY action
                ORDER BY count DESC
            """,
                (symbol, cutoff),
            )

            action_dist = {row["action"]: row["count"] for row in cur.fetchall()}
            most_common = (
                max(action_dist.items(), key=lambda x: x[1])[0] if action_dist else "UNKNOWN"
            )

            # Risk mentions
            cur.execute(
                """
                SELECT COUNT(*) as risk_count
                FROM ai_decisions
                WHERE symbol = ?
                  AND created_ts >= ?
                  AND (risks_json IS NOT NULL AND risks_json != '[]')
            """,
                (symbol, cutoff),
            )

            risk_count = cur.fetchone()["risk_count"]

            return SymbolPerformance(
                symbol=symbol,
                decision_count=total,
                avg_confidence=round(avg_conf, 3),
                most_common_action=most_common,
                action_distribution=action_dist,
                last_decision_ts=last_ts,
                risk_mentions=risk_count,
            )

    def get_confidence_distribution(self, hours: int = 168, buckets: int = 10) -> dict[str, int]:
        """
        Get histogram of confidence scores.

        Args:
            hours: Lookback period
            buckets: Number of histogram buckets (default: 10)

        Returns:
            Dict mapping confidence range to count
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT confidence
                FROM ai_decisions
                WHERE created_ts >= ? AND confidence IS NOT NULL
            """,
                (cutoff,),
            )

            confidences = [row["confidence"] for row in cur.fetchall()]

            if not confidences:
                return {}

            # Create histogram
            bucket_size = 1.0 / buckets
            histogram = {}

            for conf in confidences:
                bucket_idx = min(int(conf / bucket_size), buckets - 1)
                bucket_min = bucket_idx * bucket_size
                bucket_max = bucket_min + bucket_size
                bucket_label = f"{bucket_min:.1f}-{bucket_max:.1f}"
                histogram[bucket_label] = histogram.get(bucket_label, 0) + 1

            return dict(sorted(histogram.items()))

    def get_decision_timeline(
        self, hours: int = 168, interval_hours: int = 24
    ) -> list[dict[str, Any]]:
        """
        Get time-series of decision counts and avg confidence.

        Args:
            hours: Total lookback period
            interval_hours: Size of each time bucket

        Returns:
            List of dicts with timestamp, count, avg_confidence
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT
                    created_ts,
                    confidence
                FROM ai_decisions
                WHERE created_ts >= ?
                ORDER BY created_ts ASC
            """,
                (cutoff,),
            )

            rows = cur.fetchall()

            if not rows:
                return []

            # Group by interval
            timeline = []
            current_bucket = None
            bucket_decisions = []

            for row in rows:
                ts = datetime.fromisoformat(row["created_ts"])
                bucket_start = ts.replace(minute=0, second=0, microsecond=0)
                bucket_start = bucket_start - timedelta(hours=bucket_start.hour % interval_hours)

                if current_bucket != bucket_start:
                    # Save previous bucket
                    if bucket_decisions and current_bucket is not None:
                        confidences = [d for d in bucket_decisions if d is not None]
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                        timeline.append(
                            {
                                "timestamp": current_bucket.isoformat(),
                                "count": len(bucket_decisions),
                                "avg_confidence": round(avg_conf, 3),
                            }
                        )

                    # Start new bucket
                    current_bucket = bucket_start
                    bucket_decisions = []

                bucket_decisions.append(row["confidence"])

            # Save last bucket
            if bucket_decisions and current_bucket is not None:
                confidences = [d for d in bucket_decisions if d is not None]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                timeline.append(
                    {
                        "timestamp": current_bucket.isoformat(),
                        "count": len(bucket_decisions),
                        "avg_confidence": round(avg_conf, 3),
                    }
                )

            return timeline

    def get_low_confidence_decisions(
        self, threshold: float = 0.5, hours: int = 168
    ) -> list[dict[str, Any]]:
        """
        Find decisions with confidence below threshold (potential issues).

        Args:
            threshold: Confidence threshold (default: 0.5)
            hours: Lookback period

        Returns:
            List of low-confidence decisions with key details
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT
                    id,
                    created_ts,
                    symbol,
                    action,
                    confidence,
                    rationale,
                    decision_type
                FROM ai_decisions
                WHERE created_ts >= ? AND confidence < ?
                ORDER BY confidence ASC, created_ts DESC
                LIMIT 50
            """,
                (cutoff, threshold),
            )

            return [dict(row) for row in cur.fetchall()]

    def get_stale_symbols(self, hours_since_decision: int = 48) -> list[dict[str, Any]]:
        """
        Find symbols with no recent decisions (potential coverage gaps).

        Args:
            hours_since_decision: Hours threshold for staleness

        Returns:
            List of symbols with last decision timestamp
        """
        cutoff = (datetime.now(UTC) - timedelta(hours=hours_since_decision)).isoformat()

        with self._get_connection() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT
                    symbol,
                    MAX(created_ts) as last_decision_ts,
                    COUNT(*) as total_decisions
                FROM ai_decisions
                WHERE created_ts < ?
                GROUP BY symbol
                ORDER BY last_decision_ts ASC
            """,
                (cutoff,),
            )

            return [dict(row) for row in cur.fetchall()]

    def compute_decision_quality_score(self, hours: int = 168) -> float:
        """
        Compute overall decision quality score (0-100).

        Factors:
        - Average confidence (weight: 40%)
        - Decision coverage (weight: 30%)
        - Decision recency (weight: 20%)
        - Risk assessment rate (weight: 10%)

        Args:
            hours: Lookback period

        Returns:
            Quality score 0-100
        """
        stats = self.get_decision_stats(hours)

        if stats.total_decisions == 0:
            return 0.0

        # Component 1: Average confidence (0-40 points)
        conf_score = stats.avg_confidence * 40

        # Component 2: Coverage (0-30 points)
        # Assume 5+ symbols is good coverage
        coverage_score = min(stats.unique_symbols / 5.0, 1.0) * 30

        # Component 3: Recency (0-20 points)
        if stats.newest_decision_ts:
            newest = datetime.fromisoformat(stats.newest_decision_ts)
            age_hours = (datetime.now(UTC) - newest).total_seconds() / 3600
            recency_score = max(0, (24 - age_hours) / 24) * 20  # Decay over 24h
        else:
            recency_score = 0

        # Component 4: Risk assessment rate (0-10 points)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN risks_json IS NOT NULL AND risks_json != '[]' THEN 1 ELSE 0 END) as with_risks
                FROM ai_decisions
                WHERE created_ts >= ?
            """,
                (cutoff,),
            )
            row = cur.fetchone()
            risk_rate = row["with_risks"] / row["total"] if row["total"] > 0 else 0
            risk_score = risk_rate * 10

        total_score = conf_score + coverage_score + recency_score + risk_score
        return round(total_score, 2)


def format_stats_for_api(stats: DecisionStats) -> dict[str, Any]:
    """Convert DecisionStats to API-friendly dict."""
    return asdict(stats)


def format_performance_for_api(perf: SymbolPerformance) -> dict[str, Any]:
    """Convert SymbolPerformance to API-friendly dict."""
    return asdict(perf)


def get_tool_call_analytics(db_path: str, hours: int = 24) -> list[ToolCallMetrics]:
    """
    Aggregate tool call metrics by tool name.

    Args:
        db_path: Path to database
        hours: Lookback period

    Returns:
        List of ToolCallMetrics for each tool
    """
    cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            tool_name,
            COUNT(*) as total,
            SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successes,
            SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures,
            AVG(latency_ms) as avg_latency,
            MAX(created_ts) as last_called,
            SUM(COALESCE(data_size_bytes, 0)) as total_bytes
        FROM tool_calls
        WHERE created_ts >= ?
        GROUP BY tool_name
        ORDER BY total DESC
    """,
        (cutoff,),
    )

    results = []
    for row in cur.fetchall():
        total = row["total"]
        successes = row["successes"]
        success_rate = successes / total if total > 0 else 0.0

        # Get most used symbol for this tool
        cur.execute(
            """
            SELECT symbol, COUNT(*) as count
            FROM tool_calls
            WHERE tool_name = ? AND created_ts >= ? AND symbol IS NOT NULL
            GROUP BY symbol
            ORDER BY count DESC
            LIMIT 1
        """,
            (row["tool_name"], cutoff),
        )

        symbol_row = cur.fetchone()
        most_used_symbol = symbol_row["symbol"] if symbol_row else None

        results.append(
            ToolCallMetrics(
                tool_name=row["tool_name"],
                total_calls=total,
                success_count=successes,
                failure_count=row["failures"],
                avg_latency_ms=round(row["avg_latency"], 2) if row["avg_latency"] else 0.0,
                success_rate=round(success_rate, 3),
                last_called_ts=row["last_called"],
                most_used_symbol=most_used_symbol,
                total_data_bytes=row["total_bytes"],
            )
        )

    conn.close()
    return results
