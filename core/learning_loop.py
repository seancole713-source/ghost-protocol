"""
GHOST Stage 2: Learning Loop
=============================
Auto-tunes model parameters based on accuracy feedback.

Features:
- Monitor MAP and trigger retuning when > 5%
- Adjust confidence thresholds
- Update risk parameters
- Store learning history
- Version model configurations

Intelligence Level: 8 → 9 (Self-Evaluation System)

Author: Ghost AI
Date: 2025-10-05
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.accuracy_tracker import get_accuracy_tracker

logger = logging.getLogger(__name__)

# Learning memory path
MEMORY_PATH = Path(__file__).parent.parent / "data" / "model_memory.json"


class LearningLoop:
    """
    Manages model self-tuning based on performance feedback.

    Workflow:
    1. check_performance() - Check if MAP > threshold
    2. analyze_bias() - Detect systematic errors
    3. adjust_parameters() - Update model config
    4. save_learning() - Store adjustment history
    """

    def __init__(
        self, memory_path: str | None = None, mape_threshold: float = 5.0, min_samples: int = 10
    ):
        """
        Initialize learning loop.

        Args:
            memory_path: Path to learning memory JSON
            mape_threshold: Trigger retuning when MAP exceeds this (%)
            min_samples: Minimum forecasts before tuning
        """
        self.memory_path = memory_path or str(MEMORY_PATH)
        self.mape_threshold = mape_threshold
        self.min_samples = min_samples
        self.tracker = get_accuracy_tracker()
        self._load_memory()

    def _load_memory(self):
        """Load learning history from PostgreSQL first, then disk fallback."""
        import os
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            try:
                import psycopg2
                conn = psycopg2.connect(database_url)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ghost_kv_store (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL,
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                cursor.execute("SELECT value FROM ghost_kv_store WHERE key = 'learning_memory'")
                row = cursor.fetchone()
                conn.close()
                if row:
                    self.memory = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    logger.info(
                        f"Loaded learning memory from PostgreSQL: {len(self.memory.get('history', []))} entries"
                    )
                    return
            except Exception as e:
                logger.warning(f"PostgreSQL memory load failed, trying disk: {e}")
        
        # Disk fallback
        if Path(self.memory_path).exists():
            try:
                with open(self.memory_path) as f:
                    self.memory = json.load(f)
                logger.info(
                    f"Loaded learning memory from disk: {len(self.memory.get('history', []))} entries"
                )
            except Exception as e:
                logger.warning(f"Failed to load memory: {e}")
                self.memory = self._init_memory()
        else:
            self.memory = self._init_memory()

    def _init_memory(self) -> dict[str, Any]:
        """Initialize empty memory structure."""
        return {
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "last_tune": None,
            "tune_count": 0,
            "history": [],
            "current_config": {
                "confidence_threshold": 0.7,
                "risk_multiplier": 1.0,
                "bias_correction": 0.0,
                "volatility_adjustment": 1.0,
            },
        }

    def _save_memory(self):
        """Save learning history to PostgreSQL (survives Railway redeploys) + disk fallback."""
        # Try PostgreSQL first (production)
        import os
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            try:
                import psycopg2
                conn = psycopg2.connect(database_url)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ghost_kv_store (
                        key TEXT PRIMARY KEY,
                        value JSONB NOT NULL,
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                cursor.execute("""
                    INSERT INTO ghost_kv_store (key, value, updated_at)
                    VALUES ('learning_memory', %s, NOW())
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """, (json.dumps(self.memory),))
                conn.commit()
                conn.close()
                logger.info("Saved learning memory to PostgreSQL")
                return
            except Exception as e:
                logger.warning(f"PostgreSQL memory save failed, falling back to disk: {e}")
        
        # Disk fallback (development)
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.memory_path, "w") as f:
                json.dump(self.memory, f, indent=2)
            logger.info(f"Saved learning memory: {self.memory_path}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _get_postgres_direction_accuracy(self, days: int = 7) -> dict[str, Any]:
        """
        Get REAL direction accuracy from Postgres outcomes.
        This is where the 25,691 predictions live with actual win/loss data.
        """
        import os
        try:
            import psycopg2
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                return {"error": "DATABASE_URL not set", "count": 0}
            
            conn = psycopg2.connect(database_url)
            cursor = conn.cursor()
            
            # Get accuracy from paper_trades (the primary accuracy source)
            # This aligns with paper_tracker.get_stats() and /api/v3/accuracy/summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN outcome IN ('LOSS', 'STOPPED') THEN 1 ELSE 0 END) as incorrect
                FROM paper_trades
                WHERE outcome IN ('WIN', 'LOSS', 'STOPPED')
                AND created_at > NOW() - INTERVAL '%s days'
            """, (days,))
            
            row = cursor.fetchone()
            total = row[0] or 0
            correct = row[1] or 0
            incorrect = row[2] or 0
            
            cursor.close()
            conn.close()
            
            if total == 0:
                return {"error": "No outcomes found", "count": 0}
            
            accuracy_pct = correct / total * 100 if total > 0 else 0
            error_rate = 100 - accuracy_pct  # "MAP" equivalent for direction
            
            return {
                "count": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy_pct": accuracy_pct,
                "map": error_rate,  # Error rate (higher = needs tuning)
                "mape": error_rate,
                "bias_pct": accuracy_pct - 50.0,  # Bias from coin flip
                "data_source": "postgres_outcomes"
            }
            
        except Exception as e:
            logger.warning(f"Could not get Postgres accuracy: {e}")
            return {"error": str(e), "count": 0}

    def check_performance(self, symbol: str | None = None, days: int = 7) -> dict[str, Any]:
        """
        Check if model performance requires tuning.
        
        UPDATED: Now uses Postgres direction outcomes (25,691 predictions)
        instead of empty SQLite forecast_price tracker.

        Args:
            symbol: Check specific symbol (None = all)
            days: Look back window

        Returns:
            Dict with needs_tuning flag and metrics
        """
        # Try Postgres first (where the real data lives!)
        metrics = self._get_postgres_direction_accuracy(days=days)
        
        # Fall back to SQLite tracker only if Postgres fails
        if "error" in metrics and metrics.get("count", 0) == 0:
            logger.warning(f"Postgres accuracy unavailable, trying SQLite: {metrics.get('error')}")
            metrics = self.tracker.calculate_metrics(symbol=symbol, days=days)

        if "error" in metrics:
            return {"needs_tuning": False, "reason": metrics["error"], "metrics": metrics}

        # Check conditions
        map = metrics["map"]
        count = metrics["count"]

        needs_tuning = False
        reasons = []

        if count < self.min_samples:
            reasons.append(f"Insufficient samples ({count} < {self.min_samples})")
        elif map > self.mape_threshold:
            needs_tuning = True
            reasons.append(f"MAP too high ({map:.2f}% > {self.mape_threshold}%)")

        # Check bias
        bias_pct = abs(metrics.get("bias_pct", 0.0))
        if bias_pct > 3.0:
            needs_tuning = True
            reasons.append(f"High bias detected ({bias_pct:.2f}%)")

        return {
            "needs_tuning": needs_tuning,
            "reasons": reasons,
            "metrics": metrics,
            "symbol": symbol or "all",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def analyze_bias(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze bias patterns and recommend adjustments.

        Args:
            metrics: Output from calculate_metrics()

        Returns:
            Dict with bias analysis and adjustments
        """
        bias_pct = metrics.get("bias_pct", 0.0)
        map = metrics.get("map", 0.0)

        analysis = {
            "bias_detected": abs(bias_pct) > 1.0,
            "bias_direction": "over" if bias_pct > 0 else "under" if bias_pct < 0 else "none",
            "bias_magnitude": abs(bias_pct),
            "recommendations": [],
        }

        # Bias correction
        if abs(bias_pct) > 3.0:
            correction = -bias_pct / 100.0  # Opposite direction
            analysis["recommendations"].append(
                {
                    "parameter": "bias_correction",
                    "current": self.memory["current_config"]["bias_correction"],
                    "suggested": correction,
                    "reason": f"Correct {analysis['bias_direction']}-prediction by {abs(bias_pct):.2f}%",
                }
            )

        # Confidence threshold
        if map > 7.0:
            # High error → increase confidence threshold
            current = self.memory["current_config"]["confidence_threshold"]
            suggested = min(0.9, current + 0.05)
            analysis["recommendations"].append(
                {
                    "parameter": "confidence_threshold",
                    "current": current,
                    "suggested": suggested,
                    "reason": f"Increase threshold to filter low-confidence forecasts (MAP={map:.2f}%)",
                }
            )
        elif map < 3.0 and metrics["count"] > 20:
            # Low error → decrease threshold to capture more opportunities
            current = self.memory["current_config"]["confidence_threshold"]
            suggested = max(0.5, current - 0.05)
            analysis["recommendations"].append(
                {
                    "parameter": "confidence_threshold",
                    "current": current,
                    "suggested": suggested,
                    "reason": f"Decrease threshold to capture more opportunities (MAP={map:.2f}%)",
                }
            )

        return analysis

    def adjust_parameters(
        self, recommendations: list[dict[str, Any]], auto_apply: bool = False
    ) -> dict[str, Any]:
        """
        Adjust model parameters based on recommendations.

        Args:
            recommendations: List from analyze_bias()
            auto_apply: If True, immediately apply changes

        Returns:
            Dict with adjustments made
        """
        adjustments = {
            "timestamp": datetime.now(UTC).isoformat(),
            "applied": auto_apply,
            "changes": [],
        }

        for rec in recommendations:
            param = rec["parameter"]
            old_value = rec["current"]
            new_value = rec["suggested"]
            reason = rec["reason"]

            adjustments["changes"].append(
                {
                    "parameter": param,
                    "old_value": old_value,
                    "new_value": new_value,
                    "reason": reason,
                }
            )

            if auto_apply:
                self.memory["current_config"][param] = new_value
                logger.info(f"Parameter adjusted: {param} = {old_value} → {new_value} ({reason})")

        if auto_apply and adjustments["changes"]:
            self.memory["last_tune"] = datetime.now(UTC).isoformat()
            self.memory["tune_count"] += 1
            self.memory["history"].append(adjustments)
            self._save_memory()

        return adjustments

    def run_learning_cycle(
        self, symbol: str | None = None, days: int = 7, auto_apply: bool = True
    ) -> dict[str, Any]:
        """
        Execute full learning cycle: check → analyze → adjust.

        Args:
            symbol: Target symbol (None = all)
            days: Look back window
            auto_apply: Auto-apply adjustments

        Returns:
            Dict with cycle results
        """
        logger.info(f"Starting learning cycle: symbol={symbol}, days={days}")

        # Step 1: Check performance
        perf = self.check_performance(symbol=symbol, days=days)

        if not perf["needs_tuning"]:
            return {
                "cycle_run": True,
                "tuning_needed": False,
                "performance": perf,
                "summary": f"Performance OK: MAP={perf['metrics'].get('mape', 0):.2f}%",
            }

        # Step 2: Analyze bias
        analysis = self.analyze_bias(perf["metrics"])

        if not analysis["recommendations"]:
            return {
                "cycle_run": True,
                "tuning_needed": True,
                "adjustments_made": False,
                "performance": perf,
                "analysis": analysis,
                "summary": "Tuning needed but no clear adjustments identified",
            }

        # Step 3: Adjust parameters
        adjustments = self.adjust_parameters(analysis["recommendations"], auto_apply=auto_apply)

        return {
            "cycle_run": True,
            "tuning_needed": True,
            "adjustments_made": auto_apply,
            "performance": perf,
            "analysis": analysis,
            "adjustments": adjustments,
            "summary": (
                f"Tuned {len(adjustments['changes'])} parameters "
                f"(MAP={perf['metrics']['mape']:.2f}%, "
                f"bias={perf['metrics']['bias_pct']:+.2f}%)"
            ),
        }

    def get_current_config(self) -> dict[str, Any]:
        """Get current model configuration."""
        return self.memory["current_config"].copy()

    def get_learning_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent learning adjustments.

        Args:
            limit: Max entries to return

        Returns:
            List of adjustment dicts
        """
        history = self.memory.get("history", [])
        return history[-limit:] if limit else history

    def get_learning_stats(self) -> dict[str, Any]:
        """Get learning loop statistics."""
        return {
            "tune_count": self.memory.get("tune_count", 0),
            "last_tune": self.memory.get("last_tune"),
            "mape_threshold": self.mape_threshold,
            "min_samples": self.min_samples,
            "current_config": self.get_current_config(),
            "history_count": len(self.memory.get("history", [])),
        }

    def reset_config(self):
        """Reset configuration to defaults."""
        self.memory["current_config"] = self._init_memory()["current_config"]
        self.memory["last_tune"] = datetime.now(UTC).isoformat()
        self.memory["tune_count"] += 1
        self.memory["history"].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "reset",
                "reason": "Manual reset to defaults",
            }
        )
        self._save_memory()
        logger.info("Configuration reset to defaults")


# Singleton instance
_learning_loop = None


def get_learning_loop() -> LearningLoop:
    """Get or create the global learning loop instance."""
    global _learning_loop
    if _learning_loop is None:
        _learning_loop = LearningLoop()
    return _learning_loop


# Convenience functions
def check_performance(*args, **kwargs) -> dict[str, Any]:
    """Check performance (convenience wrapper)."""
    return get_learning_loop().check_performance(*args, **kwargs)


def run_learning_cycle(*args, **kwargs) -> dict[str, Any]:
    """Run learning cycle (convenience wrapper)."""
    return get_learning_loop().run_learning_cycle(*args, **kwargs)


def get_current_config() -> dict[str, Any]:
    """Get current config (convenience wrapper)."""
    return get_learning_loop().get_current_config()


def get_learning_stats() -> dict[str, Any]:
    """Get learning stats (convenience wrapper)."""
    return get_learning_loop().get_learning_stats()
