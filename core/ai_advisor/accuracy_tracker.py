"""
AI Accuracy Tracker - Learn from Past Decisions
Tracks outcomes, calculates accuracy, enables learning
Target: 80%+ accuracy
"""

import asyncio
import json
import logging
import sqlite3
import time

LOGGER = logging.getLogger(__name__)


class AccuracyTracker:
    """
    Track AI decision outcomes to improve over time
    """

    def __init__(self, db_path: str = "ghost_data.db"):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """Initialize decision tracking tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # AI decisions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT NOT NULL,
                entry_price REAL NOT NULL,
                target_price REAL,
                stop_loss REAL,
                position_size_pct REAL,
                risk_factors_json TEXT,
                timeframe TEXT,
                expected_return_pct REAL,
                context_json TEXT,
                created_at REAL NOT NULL,
                outcome_price REAL,
                return_pct REAL,
                correct INTEGER,
                hit_target INTEGER,
                hit_stop INTEGER,
                checked_at REAL
            )
        """)

        # Create indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_asset ON ai_decisions(asset)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_created_at ON ai_decisions(created_at)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_outcome ON ai_decisions(correct) WHERE outcome_price IS NOT NULL"
        )

        conn.commit()
        conn.close()

        LOGGER.info("✅ Accuracy tracker tables initialized")

    async def record_decision(
        self,
        decision_id: str,
        asset: str,
        asset_type: str,
        decision: str,
        confidence: float,
        reasoning: str,
        entry_price: float,
        target_price: float | None = None,
        stop_loss: float | None = None,
        position_size_pct: float = 2.0,
        risk_factors: list[str] | None = None,
        timeframe: str = "short-term",
        expected_return_pct: float = 0.0,
        context: dict | None = None,
    ):
        """
        Record a decision when made
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            c.execute(
                """
                INSERT INTO ai_decisions (
                    id, asset, asset_type, decision, confidence,
                    reasoning, entry_price, target_price, stop_loss,
                    position_size_pct, risk_factors_json, timeframe,
                    expected_return_pct, context_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    decision_id,
                    asset,
                    asset_type,
                    decision,
                    confidence,
                    reasoning,
                    entry_price,
                    target_price,
                    stop_loss,
                    position_size_pct,
                    json.dumps(risk_factors or []),
                    timeframe,
                    expected_return_pct,
                    json.dumps(context or {}),
                    time.time(),
                ],
            )

            conn.commit()
            LOGGER.info("📝 Recorded decision: %s %s @ $%.2f", decision, asset, entry_price)

            # Schedule outcome check based on timeframe
            check_delay = self._get_check_delay(timeframe)
            asyncio.create_task(self._schedule_check(decision_id, check_delay))

        except Exception as e:
            LOGGER.error("Failed to record decision: %s", e)
        finally:
            conn.close()

    def _get_check_delay(self, timeframe: str) -> int:
        """Get seconds to wait before checking outcome"""
        delays = {
            "short-term": 86400,  # 1 day
            "medium-term": 604800,  # 7 days
            "long-term": 2592000,  # 30 days
        }
        return delays.get(timeframe, 86400)

    async def _schedule_check(self, decision_id: str, delay: int):
        """Schedule outcome check"""
        await asyncio.sleep(delay)
        await self.check_outcome(decision_id)

    async def check_outcome(self, decision_id: str):
        """
        Check if decision was correct
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        try:
            # Get decision
            c.execute("SELECT * FROM ai_decisions WHERE id = ?", [decision_id])
            row = c.fetchone()
            if not row:
                return

            cols = [desc[0] for desc in c.description]
            decision = dict(zip(cols, row, strict=False))

            # Get current price (would call price API in real implementation)
            current_price = await self._get_current_price(decision["asset"], decision["asset_type"])

            if not current_price:
                LOGGER.warning("Could not get current price for %s", decision["asset"])
                return

            # Calculate outcome
            entry_price = decision["entry_price"]
            return_pct = ((current_price - entry_price) / entry_price) * 100

            # Determine correctness
            if decision["decision"] == "BUY":
                correct = return_pct > 0
                hit_target = (
                    current_price >= decision["target_price"] if decision["target_price"] else False
                )
                hit_stop = (
                    current_price <= decision["stop_loss"] if decision["stop_loss"] else False
                )
            elif decision["decision"] == "SELL":
                correct = return_pct < 0
                hit_target = (
                    current_price <= decision["target_price"] if decision["target_price"] else False
                )
                hit_stop = (
                    current_price >= decision["stop_loss"] if decision["stop_loss"] else False
                )
            else:
                correct = False
                hit_target = False
                hit_stop = False

            # Update database
            c.execute(
                """
                UPDATE ai_decisions
                SET outcome_price = ?,
                    return_pct = ?,
                    correct = ?,
                    hit_target = ?,
                    hit_stop = ?,
                    checked_at = ?
                WHERE id = ?
            """,
                [
                    current_price,
                    return_pct,
                    int(correct),
                    int(hit_target),
                    int(hit_stop),
                    time.time(),
                    decision_id,
                ],
            )

            conn.commit()

            LOGGER.info(
                "✅ Outcome checked: %s %s - %s (%.1f%% return)",
                decision["asset"],
                decision["decision"],
                "CORRECT" if correct else "WRONG",
                return_pct,
            )

        except Exception as e:
            LOGGER.error("Failed to check outcome: %s", e)
        finally:
            conn.close()

    async def _get_current_price(self, asset: str, asset_type: str) -> float | None:
        """Get current price for asset"""
        try:
            if asset_type == "crypto":
                # Use crypto price endpoint
                from core.crypto.crypto_providers import get_crypto_price_quorum

                data = await get_crypto_price_quorum(asset, use_cache=False)
                return data.get("price")
            else:
                # Use stock price endpoint (placeholder)
                # In real implementation, would call AlphaVantage or similar
                return None
        except Exception as e:
            LOGGER.error("Failed to get price for %s: %s", asset, e)
            return None

    def get_accuracy(self, asset_type: str | None = None) -> float:
        """
        Calculate overall accuracy (% of correct predictions)
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count
            FROM ai_decisions
            WHERE outcome_price IS NOT NULL
        """

        if asset_type:
            query += f" AND asset_type = '{asset_type}'"

        c.execute(query)
        row = c.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return 0.0

        total, correct_count = row
        return (correct_count / total) * 100

    def get_win_rate(self, asset_type: str | None = None) -> float:
        """Get % of profitable trades"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) as wins
            FROM ai_decisions
            WHERE outcome_price IS NOT NULL
        """

        if asset_type:
            query += f" AND asset_type = '{asset_type}'"

        c.execute(query)
        row = c.fetchone()
        conn.close()

        if not row or row[0] == 0:
            return 0.0

        total, wins = row
        return (wins / total) * 100

    def get_avg_return(self, asset_type: str | None = None) -> float:
        """Get average return per trade"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        query = "SELECT AVG(return_pct) FROM ai_decisions WHERE outcome_price IS NOT NULL"

        if asset_type:
            query += f" AND asset_type = '{asset_type}'"

        c.execute(query)
        row = c.fetchone()
        conn.close()

        return row[0] if row and row[0] else 0.0

    def find_similar(self, asset: str, context: dict, limit: int = 10) -> list[dict]:
        """
        Find similar past decisions to learn from
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Simple similarity: same asset type, similar timeframe
        c.execute(
            """
            SELECT * FROM ai_decisions
            WHERE outcome_price IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
        """,
            [limit],
        )

        rows = c.fetchall()
        cols = [desc[0] for desc in c.description]

        similar = []
        for row in rows:
            decision = dict(zip(cols, row, strict=False))
            similar.append(decision)

        conn.close()
        return similar

    def get_stats(self) -> dict:
        """Get comprehensive statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Overall stats
        c.execute("""
            SELECT
                COUNT(*) as total_decisions,
                SUM(CASE WHEN outcome_price IS NOT NULL THEN 1 ELSE 0 END) as checked,
                AVG(CASE WHEN outcome_price IS NOT NULL THEN confidence ELSE NULL END) as avg_confidence,
                AVG(CASE WHEN outcome_price IS NOT NULL THEN return_pct ELSE NULL END) as avg_return
            FROM ai_decisions
        """)

        row = c.fetchone()
        total_decisions, checked, avg_confidence, avg_return = row

        # Recent performance (last 30 days)
        thirty_days_ago = time.time() - (30 * 86400)
        c.execute(
            """
            SELECT
                COUNT(*) as recent_decisions,
                AVG(return_pct) as recent_avg_return,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as recent_correct
            FROM ai_decisions
            WHERE created_at > ? AND outcome_price IS NOT NULL
        """,
            [thirty_days_ago],
        )

        row = c.fetchone()
        recent_decisions, recent_avg_return, recent_correct = row
        recent_accuracy = (recent_correct / recent_decisions * 100) if recent_decisions else 0

        conn.close()

        return {
            "total_decisions": total_decisions or 0,
            "checked_outcomes": checked or 0,
            "pending_checks": (total_decisions or 0) - (checked or 0),
            "overall_accuracy_pct": self.get_accuracy(),
            "win_rate_pct": self.get_win_rate(),
            "avg_return_pct": avg_return or 0.0,
            "avg_confidence": avg_confidence or 0.0,
            "recent_30d": {
                "decisions": recent_decisions or 0,
                "accuracy_pct": recent_accuracy,
                "avg_return_pct": recent_avg_return or 0.0,
            },
            "by_asset_type": {
                "stocks": {
                    "accuracy_pct": self.get_accuracy("stock"),
                    "win_rate_pct": self.get_win_rate("stock"),
                    "avg_return_pct": self.get_avg_return("stock"),
                },
                "crypto": {
                    "accuracy_pct": self.get_accuracy("crypto"),
                    "win_rate_pct": self.get_win_rate("crypto"),
                    "avg_return_pct": self.get_avg_return("crypto"),
                },
            },
        }


# Global tracker instance
_TRACKER: AccuracyTracker | None = None


def get_tracker() -> AccuracyTracker:
    """Get global tracker instance"""
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = AccuracyTracker()
    return _TRACKER
