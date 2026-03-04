"""
APEX Online Calibration System
Mini-batch retraining and adaptive weight adjustment

Expected Impact: +30% model adaptability

Migrated from SQLite → PostgreSQL so data survives Railway deploys.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


def _pg_conn():
    """Context manager that guarantees connection is returned to pool.

    Usage:
        with _pg_conn() as conn:
            cur = conn.cursor()
            cur.execute(...)
            conn.commit()
    """
    from contextlib import contextmanager as _cm
    @_cm
    def _pg():
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            yield conn
    return _pg()


@dataclass
class PerformanceMetrics:
    """Performance metrics for a forecast or strategy"""

    accuracy: float  # % correct predictions
    mape: float  # Mean Absolute Percentage Error
    sharpe: float  # Risk-adjusted returns
    win_rate: float  # % winning trades
    avg_return: float  # Average return per prediction
    sample_count: int  # Number of samples evaluated


@dataclass
class CalibrationResult:
    """Result of a calibration run"""

    timestamp: int
    calibration_type: str  # 'ensemble_weights' | 'horizon_weights' | 'strategy_weights'
    old_weights: dict[str, float]
    new_weights: dict[str, float]
    performance_gain: float  # Expected improvement %
    reason: str


class OnlineCalibrator:
    """
    APEX Online Calibration System
    Continuously adjusts model weights based on recent performance.
    Uses PostgreSQL (DATABASE_URL) for persistence across Railway deploys.
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.min_samples = 10  # Minimum samples needed for calibration
        # Store latest calibration results for consumer access
        self._latest_horizon_weights: dict[str, float] = {
            "nowcast": 0.20, "swing": 0.40, "position": 0.40
        }
        self._latest_strategy_weights: dict[str, float] = {
            "Momentum": 0.50, "NewsShock": 0.40, "PairsTrading": 0.10
        }
        self._load_latest_weights()  # Restore from DB on init
        self._init_db()

    def _init_db(self):
        """Initialize calibration tracking tables in PostgreSQL."""
        try:
            with _pg_conn() as conn:
                cur = conn.cursor()

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS calibrator_forecast_performance (
                        id SERIAL PRIMARY KEY,
                        timestamp BIGINT NOT NULL,
                        horizon TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        predicted_price DOUBLE PRECISION,
                        actual_price DOUBLE PRECISION,
                        predicted_return DOUBLE PRECISION,
                        actual_return DOUBLE PRECISION,
                        confidence DOUBLE PRECISION,
                        error_pct DOUBLE PRECISION,
                        was_correct INTEGER
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS calibrator_strategy_performance (
                        id SERIAL PRIMARY KEY,
                        timestamp BIGINT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence DOUBLE PRECISION,
                        entry_price DOUBLE PRECISION,
                        exit_price DOUBLE PRECISION,
                        return_pct DOUBLE PRECISION,
                        was_profitable INTEGER
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS calibrator_calibration_history (
                        id SERIAL PRIMARY KEY,
                        timestamp BIGINT NOT NULL,
                        calibration_type TEXT NOT NULL,
                        old_weights TEXT,
                        new_weights TEXT,
                        performance_gain DOUBLE PRECISION,
                        reason TEXT
                    )
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS calibrator_model_drift_log (
                        id SERIAL PRIMARY KEY,
                        timestamp BIGINT NOT NULL,
                        model_name TEXT NOT NULL,
                        baseline_mape DOUBLE PRECISION,
                        current_mape DOUBLE PRECISION,
                        drift_pct DOUBLE PRECISION,
                        triggered_recalibration INTEGER
                    )
                """)

                conn.commit()
                cur.close()

                LOGGER.info("Online Calibrator initialized (PostgreSQL)")

        except Exception as e:
            LOGGER.error(f"Online Calibrator DB init failed: {e}")

    # ------------------------------------------------------------------
    # Weight access (for consumers)
    # ------------------------------------------------------------------

    def _load_latest_weights(self):
        """Restore latest calibration weights from DB on startup."""
        try:
            with _pg_conn() as conn:
                cur = conn.cursor()
                for cal_type, attr in [
                    ("horizon_weights", "_latest_horizon_weights"),
                    ("strategy_weights", "_latest_strategy_weights"),
                ]:
                    cur.execute(
                        """
                        SELECT new_weights FROM calibrator_calibration_history
                        WHERE calibration_type LIKE %s
                        ORDER BY timestamp DESC LIMIT 1
                        """,
                        (f"{cal_type}%",),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        weights = json.loads(row[0])
                        if isinstance(weights, dict) and len(weights) >= 2:
                            setattr(self, attr, weights)
                            LOGGER.info(f"Restored {cal_type} from DB: {weights}")
                cur.close()
        except Exception as e:
            LOGGER.warning(f"Could not restore calibration weights: {e}")

    def get_latest_weights(self, weight_type: str = "horizon") -> dict[str, float]:
        """Get the latest calibrated weights.

        Args:
            weight_type: 'horizon' or 'strategy'

        Returns:
            Dict of weight name → float, summing to 1.0
        """
        if weight_type == "strategy":
            return dict(self._latest_strategy_weights)
        return dict(self._latest_horizon_weights)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_forecast_result(
        self,
        horizon: str,
        symbol: str,
        predicted_price: float,
        actual_price: float,
        confidence: float,
    ):
        """Log a forecast result for later calibration analysis."""
        try:
            predicted_return = (
                (predicted_price - actual_price) / actual_price if actual_price > 0 else 0
            )
            actual_return = 0  # Updated when actual outcome is known
            error_pct = (
                abs(predicted_price - actual_price) / actual_price if actual_price > 0 else 0
            )
            was_correct = 1 if abs(error_pct) < 0.05 else 0  # Within 5% = correct

            with _pg_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO calibrator_forecast_performance
                    (timestamp, horizon, symbol, predicted_price, actual_price,
                     predicted_return, actual_return, confidence, error_pct, was_correct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        int(time.time()),
                        horizon,
                        symbol,
                        predicted_price,
                        actual_price,
                        predicted_return,
                        actual_return,
                        confidence,
                        error_pct,
                        was_correct,
                    ),
                )
                conn.commit()
                cur.close()

            LOGGER.debug(f"Logged forecast: {horizon} {symbol} err={error_pct * 100:.1f}%")

        except Exception as e:
            LOGGER.error(f"Failed to log forecast result: {e}")

    def log_strategy_result(
        self,
        strategy_name: str,
        symbol: str,
        action: str,
        confidence: float,
        entry_price: float,
        exit_price: float,
    ):
        """Log a strategy result for later calibration analysis."""
        try:
            return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
            was_profitable = 1 if return_pct > 0 else 0

            with _pg_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO calibrator_strategy_performance
                    (timestamp, strategy_name, symbol, action, confidence,
                     entry_price, exit_price, return_pct, was_profitable)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        int(time.time()),
                        strategy_name,
                        symbol,
                        action,
                        confidence,
                        entry_price,
                        exit_price,
                        return_pct,
                        was_profitable,
                    ),
                )
                conn.commit()
                cur.close()

                LOGGER.debug(
                    f"Logged strategy: {strategy_name} {action} return={return_pct * 100:.1f}%"
                )

        except Exception as e:
            LOGGER.error(f"Failed to log strategy result: {e}")

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------

    def calibrate_horizon_weights(self) -> CalibrationResult | None:
        """
        Calibrate Multi-Horizon Brain weights based on recent MAPE.
        Returns CalibrationResult if calibration performed, None otherwise.
        """
        try:
          with _pg_conn() as conn:
            cur = conn.cursor()
            cutoff = int(time.time()) - (self.lookback_days * 86400)

            cur.execute(
                """
                SELECT horizon,
                       AVG(error_pct) AS avg_mape,
                       COUNT(*)       AS sample_count,
                       SUM(was_correct) * 1.0 / COUNT(*) AS accuracy
                FROM calibrator_forecast_performance
                WHERE timestamp > %s
                GROUP BY horizon
                """,
                (cutoff,),
            )

            horizon_metrics: dict[str, dict] = {}
            for row in cur.fetchall():
                horizon, mape, count, accuracy = row
                if count >= self.min_samples:
                    horizon_metrics[horizon] = {"mape": float(mape), "accuracy": float(accuracy), "count": int(count)}

            cur.close()

            if len(horizon_metrics) < 2:
                LOGGER.info("Not enough horizon data for calibration")
                return None

            # Current weights (baseline from multi_horizon_forecaster.py)
            old_weights = {"nowcast": 0.20, "swing": 0.40, "position": 0.40}

            # Score = (1 / MAPE) * accuracy — rewards low error + high accuracy
            scores: dict[str, float] = {}
            for horizon, metrics in horizon_metrics.items():
                if metrics["mape"] > 0:
                    scores[horizon] = (1.0 / metrics["mape"]) * metrics["accuracy"]
                else:
                    scores[horizon] = metrics["accuracy"]

            # Normalize to sum to 1.0
            total_score = sum(scores.values())
            new_weights = {h: score / total_score for h, score in scores.items()}

            for h in ["nowcast", "swing", "position"]:
                if h not in new_weights:
                    new_weights[h] = 0.01
            total = sum(new_weights.values())
            new_weights = {h: w / total for h, w in new_weights.items()}

            # Expected improvement
            old_w_mape = sum(old_weights.get(h, 0.33) * m["mape"] for h, m in horizon_metrics.items())
            new_w_mape = sum(new_weights.get(h, 0.33) * m["mape"] for h, m in horizon_metrics.items())
            performance_gain = (old_w_mape - new_w_mape) / old_w_mape if old_w_mape > 0 else 0

            if performance_gain < 0.05:
                LOGGER.info(f"Horizon calibration gain too small: {performance_gain * 100:.1f}%")
                return None

            result = CalibrationResult(
                timestamp=int(time.time()),
                calibration_type="horizon_weights",
                old_weights=old_weights,
                new_weights=new_weights,
                performance_gain=performance_gain,
                reason=(
                    f"Adjusted based on {self.lookback_days}d MAPE: "
                    + ", ".join(f"{h}={m['mape'] * 100:.1f}%" for h, m in horizon_metrics.items())
                ),
            )

            self._log_calibration(result)

            # Store weights so consumers can access them
            self._latest_horizon_weights = dict(new_weights)

            LOGGER.info(f"✅ Horizon calibration: {performance_gain * 100:.1f}% improvement expected")
            LOGGER.info(f"   Old weights: {old_weights}")
            LOGGER.info(f"   New weights: {new_weights}")

            return result

        except Exception as e:
            LOGGER.error(f"Horizon calibration failed: {e}", exc_info=True)
            return None

    def calibrate_strategy_weights(
        self, current_regime: str = "NORMAL"
    ) -> CalibrationResult | None:
        """
        Calibrate Strategy Ensemble weights based on recent profitability.
        Returns CalibrationResult if calibration performed, None otherwise.
        """
        try:
            with _pg_conn() as conn:
                cur = conn.cursor()
                cutoff = int(time.time()) - (self.lookback_days * 86400)

                cur.execute(
                    """
                    SELECT strategy_name,
                           AVG(return_pct) AS avg_return,
                           SUM(was_profitable) * 1.0 / COUNT(*) AS win_rate,
                           COUNT(*) AS sample_count
                    FROM calibrator_strategy_performance
                    WHERE timestamp > %s
                    GROUP BY strategy_name
                    """,
                    (cutoff,),
                )

                strategy_metrics: dict[str, dict] = {}
                for row in cur.fetchall():
                    strategy, avg_return, win_rate, count = row
                    if count >= self.min_samples:
                        strategy_metrics[strategy] = {
                            "avg_return": float(avg_return),
                            "win_rate": float(win_rate),
                            "count": int(count),
                        }

                cur.close()

                if len(strategy_metrics) < 2:
                    LOGGER.info("Not enough strategy data for calibration")
                    return None

                old_weights = {"Momentum": 0.50, "NewsShock": 0.40, "PairsTrading": 0.10}

                # Score = avg_return * win_rate
                scores: dict[str, float] = {}
                for strategy, metrics in strategy_metrics.items():
                    scores[strategy] = metrics["avg_return"] * metrics["win_rate"]

                min_score = min(scores.values()) if scores else 0
                if min_score < 0:
                    scores = {s: score - min_score + 0.1 for s, score in scores.items()}

                total_score = sum(scores.values())
                if total_score <= 0:
                    LOGGER.info("All strategy scores are 0, skipping calibration")
                    return None

                new_weights = {s: score / total_score for s, score in scores.items()}
                for s in ["Momentum", "NewsShock", "PairsTrading"]:
                    if s not in new_weights:
                        new_weights[s] = 0.01
                total = sum(new_weights.values())
                new_weights = {s: w / total for s, w in new_weights.items()}

                old_w_ret = sum(old_weights.get(s, 0.33) * m["avg_return"] for s, m in strategy_metrics.items())
                new_w_ret = sum(new_weights.get(s, 0.33) * m["avg_return"] for s, m in strategy_metrics.items())
                performance_gain = (new_w_ret - old_w_ret) / abs(old_w_ret) if old_w_ret != 0 else 0

                if performance_gain < 0.05:
                    LOGGER.info(f"Strategy calibration gain too small: {performance_gain * 100:.1f}%")
                    return None

                result = CalibrationResult(
                    timestamp=int(time.time()),
                    calibration_type=f"strategy_weights_{current_regime}",
                    old_weights=old_weights,
                    new_weights=new_weights,
                    performance_gain=performance_gain,
                    reason=(
                        f"Adjusted based on {self.lookback_days}d profitability: "
                        + ", ".join(f"{s}={m['avg_return'] * 100:.1f}%" for s, m in strategy_metrics.items())
                    ),
                )

                self._log_calibration(result)

                # Store weights so consumers can access them
                self._latest_strategy_weights = dict(new_weights)

                LOGGER.info(f"✅ Strategy calibration: {performance_gain * 100:.1f}% improvement expected")
                LOGGER.info(f"   Old weights: {old_weights}")
                LOGGER.info(f"   New weights: {new_weights}")

                return result

        except Exception as e:
            LOGGER.error(f"Strategy calibration failed: {e}", exc_info=True)
            return None

    def detect_model_drift(
        self, model_name: str, baseline_mape: float, current_mape: float
    ) -> bool:
        """
        Detect if model has drifted beyond acceptable threshold.
        Returns True if drift detected (triggers recalibration).
        """
        try:
            drift_pct = (current_mape - baseline_mape) / baseline_mape if baseline_mape > 0 else 0
            drift_threshold = 0.10  # 10% degradation triggers recalibration
            triggered = abs(drift_pct) > drift_threshold

            with _pg_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO calibrator_model_drift_log
                    (timestamp, model_name, baseline_mape, current_mape, drift_pct, triggered_recalibration)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (int(time.time()), model_name, baseline_mape, current_mape, drift_pct, 1 if triggered else 0),
                )
                conn.commit()
                cur.close()

                if triggered:
                    LOGGER.warning(f"⚠️ Model drift detected: {model_name} drift={drift_pct * 100:.1f}%")

                return triggered

        except Exception as e:
            LOGGER.error(f"Model drift detection failed: {e}")
            return False

    def get_adaptive_horizon(self) -> str:
        """
        Select best-performing forecast horizon based on recent MAPE.
        Returns 'nowcast' | 'swing' | 'position'.
        """
        try:
            with _pg_conn() as conn:
                cur = conn.cursor()
                cutoff = int(time.time()) - (7 * 86400)  # Last 7 days

                cur.execute(
                    """
                    SELECT horizon, AVG(error_pct) AS avg_mape
                    FROM calibrator_forecast_performance
                    WHERE timestamp > %s
                    GROUP BY horizon
                    HAVING COUNT(*) >= %s
                    ORDER BY avg_mape ASC
                    LIMIT 1
                    """,
                    (cutoff, 5),
                )

                row = cur.fetchone()
                cur.close()

                if row:
                    best_horizon = row[0]
                    LOGGER.info(f"Adaptive horizon selected: {best_horizon}")
                    return best_horizon
                else:
                    return "swing"

        except Exception as e:
            LOGGER.error(f"Adaptive horizon selection failed: {e}")
            return "swing"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_calibration(self, result: CalibrationResult):
        """Log calibration result to database."""
        try:
            with _pg_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO calibrator_calibration_history
                    (timestamp, calibration_type, old_weights, new_weights, performance_gain, reason)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result.timestamp,
                        result.calibration_type,
                        json.dumps(result.old_weights),
                        json.dumps(result.new_weights),
                        result.performance_gain,
                        result.reason,
                    ),
                )
                conn.commit()
                cur.close()
        except Exception as e:
            LOGGER.error(f"Failed to log calibration: {e}")

    def get_calibration_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent calibration history."""
        try:
            with _pg_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT timestamp, calibration_type, old_weights, new_weights,
                           performance_gain, reason
                    FROM calibrator_calibration_history
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )

                history = []
                for row in cur.fetchall():
                    history.append(
                        {
                            "timestamp": row[0],
                            "calibration_type": row[1],
                            "old_weights": json.loads(row[2]) if row[2] else {},
                            "new_weights": json.loads(row[3]) if row[3] else {},
                            "performance_gain": float(row[4]) if row[4] is not None else 0.0,
                            "reason": row[5],
                        }
                    )

                cur.close()
                return history

        except Exception as e:
            LOGGER.error(f"Failed to get calibration history: {e}")
            return []

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary for dashboard."""
        try:
            with _pg_conn() as conn:
                cur = conn.cursor()
                cutoff = int(time.time()) - (self.lookback_days * 86400)

                # Forecast performance
                cur.execute(
                    """
                    SELECT
                        horizon,
                        COUNT(*)                              AS total,
                        AVG(error_pct)                        AS avg_mape,
                        SUM(was_correct) * 1.0 / COUNT(*)     AS accuracy
                    FROM calibrator_forecast_performance
                    WHERE timestamp > %s
                    GROUP BY horizon
                    """,
                    (cutoff,),
                )

                forecast_perf: dict[str, dict] = {}
                for row in cur.fetchall():
                    forecast_perf[row[0]] = {"total": int(row[1]), "mape": float(row[2]), "accuracy": float(row[3])}

                # Strategy performance
                cur.execute(
                    """
                    SELECT
                        strategy_name,
                        COUNT(*)                                AS total,
                        AVG(return_pct)                          AS avg_return,
                        SUM(was_profitable) * 1.0 / COUNT(*)     AS win_rate
                    FROM calibrator_strategy_performance
                    WHERE timestamp > %s
                    GROUP BY strategy_name
                    """,
                    (cutoff,),
                )

                strategy_perf: dict[str, dict] = {}
                for row in cur.fetchall():
                    strategy_perf[row[0]] = {"total": int(row[1]), "avg_return": float(row[2]), "win_rate": float(row[3])}

                # Recent calibrations
                cur.execute(
                    "SELECT COUNT(*) FROM calibrator_calibration_history WHERE timestamp > %s",
                    (cutoff,),
                )
                calibration_count = cur.fetchone()[0]

                cur.close()

                return {
                    "lookback_days": self.lookback_days,
                    "forecast_performance": forecast_perf,
                    "strategy_performance": strategy_perf,
                    "calibration_count": calibration_count,
                    "last_updated": int(time.time()),
                }

        except Exception as e:
            LOGGER.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}


# Singleton instance
import threading as _threading
_ONLINE_CALIBRATOR: OnlineCalibrator | None = None
_ONLINE_CALIBRATOR_LOCK = _threading.Lock()


def get_online_calibrator() -> OnlineCalibrator:
    """Get singleton instance of online calibrator."""
    global _ONLINE_CALIBRATOR
    if _ONLINE_CALIBRATOR is None:
        with _ONLINE_CALIBRATOR_LOCK:
            if _ONLINE_CALIBRATOR is None:
                _ONLINE_CALIBRATOR = OnlineCalibrator()
    return _ONLINE_CALIBRATOR
