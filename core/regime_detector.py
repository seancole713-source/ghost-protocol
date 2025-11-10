"""
Stage 3: Market Regime Detection
Level 9→10 (90%→100%)

Detects market regimes (bull/bear/sideways/volatile) using Hidden Markov Model.
Adapts trading strategy based on current regime.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class RegimeDetector:
    """
    Market regime classifier using simple HMM-inspired logic.

    Regimes:
    - BULL: Strong uptrend (returns > 0.2%, low volatility)
    - BEAR: Strong downtrend (returns < -0.2%, low volatility)
    - SIDEWAYS: Range-bound (|returns| < 0.2%, low volatility)
    - VOLATILE: High volatility (std > 1.5%)
    """

    def __init__(self, db_path: str = "data/market_regimes.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.regimes = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
        self.current_regime = "SIDEWAYS"
        self.confidence = 0.5

        self._init_db()
        self._load_current_regime()
        LOGGER.info(f"Regime detector initialized: {self.db_path}")

    def _init_db(self):
        """Initialize database for regime history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                confidence REAL NOT NULL,

                -- Features used for detection
                mean_return REAL,
                volatility REAL,
                trend_strength REAL,

                -- Market data
                spy_price REAL,
                vix_level REAL,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_time
            ON regime_history(timestamp DESC)
        """)

        conn.commit()
        conn.close()

    def _load_current_regime(self):
        """Load most recent regime from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT regime, confidence
                FROM regime_history
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            if row:
                self.current_regime, self.confidence = row

            conn.close()
        except Exception as e:
            LOGGER.warning(f"Could not load regime: {e}")

    def detect_regime(
        self, prices: list[float], spy_price: float | None = None, vix_level: float | None = None
    ) -> dict:
        """
        Detect current market regime based on recent price action.

        Args:
            prices: Recent prices (last 20-50 periods recommended)
            spy_price: Current SPY price (optional context)
            vix_level: Current VIX level (optional context)

        Returns:
            Dict with regime, confidence, probabilities, features
        """
        if len(prices) < 10:
            return {
                "regime": "SIDEWAYS",
                "confidence": 0.5,
                "probabilities": {r: 0.25 for r in self.regimes},
                "features": {},
                "error": "Insufficient data (need 10+ prices)",
            }

        # Extract features
        features = self._extract_features(np.array(prices))

        # Classify regime
        regime, confidence, probs = self._classify_regime(features, vix_level)

        # Record regime
        self._record_regime(regime, confidence, features, spy_price, vix_level)

        # Update current state
        self.current_regime = regime
        self.confidence = confidence

        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "probabilities": {r: round(p, 3) for r, p in probs.items()},
            "features": features,
            "strategy_adjustments": self._get_strategy_adjustments(regime),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _extract_features(self, prices: np.ndarray) -> dict:
        """Extract regime features from price history."""
        prices = np.array(prices)

        # Returns
        returns = np.diff(prices) / prices[:-1]
        mean_return = np.mean(returns) * 100  # As percentage

        # Volatility
        volatility = np.std(returns) * 100  # As percentage

        # Trend strength (linear regression slope)
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        trend_strength = (slope / np.mean(prices)) * 100

        # Price momentum (10-period vs 20-period MA)
        ma10 = np.mean(prices[-10:])
        ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else ma10
        momentum = ((ma10 - ma20) / ma20) * 100 if ma20 > 0 else 0

        # Recent volatility spike (last 5 vs previous 15)
        recent_vol = np.std(returns[-5:]) * 100 if len(returns) >= 5 else volatility
        prev_vol = np.std(returns[-20:-5]) * 100 if len(returns) >= 20 else volatility
        vol_ratio = recent_vol / (prev_vol + 0.1)

        return {
            "mean_return": round(mean_return, 4),
            "volatility": round(volatility, 4),
            "trend_strength": round(trend_strength, 4),
            "momentum": round(momentum, 4),
            "vol_ratio": round(vol_ratio, 2),
        }

    def _classify_regime(
        self, features: dict, vix_level: float | None = None
    ) -> tuple[str, float, dict[str, float]]:
        """
        Classify regime based on features.

        Returns:
            (regime, confidence, probabilities_dict)
        """
        mean_return = features["mean_return"]
        volatility = features["volatility"]
        trend_strength = features["trend_strength"]
        vol_ratio = features["vol_ratio"]

        # Initialize probabilities
        probs = {r: 0.0 for r in self.regimes}

        # Volatile regime check (high priority)
        if volatility > 1.5 or vol_ratio > 2.0 or (vix_level and vix_level > 25):
            probs["VOLATILE"] = 0.7
            probs["SIDEWAYS"] = 0.2
            probs["BULL"] = 0.05
            probs["BEAR"] = 0.05
            regime = "VOLATILE"
            confidence = min(0.9, 0.5 + volatility / 5.0)

        # Bull regime: positive returns + uptrend
        elif mean_return > 0.2 and trend_strength > 0.1:
            probs["BULL"] = 0.7
            probs["SIDEWAYS"] = 0.2
            probs["VOLATILE"] = 0.05
            probs["BEAR"] = 0.05
            regime = "BULL"
            confidence = min(0.9, 0.6 + abs(trend_strength) / 2.0)

        # Bear regime: negative returns + downtrend
        elif mean_return < -0.2 and trend_strength < -0.1:
            probs["BEAR"] = 0.7
            probs["SIDEWAYS"] = 0.2
            probs["VOLATILE"] = 0.05
            probs["BULL"] = 0.05
            regime = "BEAR"
            confidence = min(0.9, 0.6 + abs(trend_strength) / 2.0)

        # Sideways regime: low volatility + weak trend
        else:
            probs["SIDEWAYS"] = 0.7
            probs["BULL"] = 0.1
            probs["BEAR"] = 0.1
            probs["VOLATILE"] = 0.1
            regime = "SIDEWAYS"
            confidence = 0.6

        return regime, confidence, probs

    def _record_regime(
        self,
        regime: str,
        confidence: float,
        features: dict,
        spy_price: float | None,
        vix_level: float | None,
    ):
        """Record regime detection in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO regime_history (
                    timestamp, regime, confidence,
                    mean_return, volatility, trend_strength,
                    spy_price, vix_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    regime,
                    confidence,
                    features["mean_return"],
                    features["volatility"],
                    features["trend_strength"],
                    spy_price,
                    vix_level,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"Failed to record regime: {e}")

    def _get_strategy_adjustments(self, regime: str) -> dict:
        """
        Get recommended strategy adjustments for current regime.
        """
        adjustments = {
            "BULL": {
                "risk_tolerance": "high",
                "position_size_multiplier": 1.2,
                "stop_loss_pct": 0.08,  # Wider stops
                "take_profit_pct": 0.15,
                "strategy": "momentum_following",
                "description": "Bull market: Follow momentum, wider stops, larger positions",
            },
            "BEAR": {
                "risk_tolerance": "low",
                "position_size_multiplier": 0.6,
                "stop_loss_pct": 0.05,  # Tighter stops
                "take_profit_pct": 0.08,
                "strategy": "mean_reversion",
                "description": "Bear market: Mean reversion, tight stops, smaller positions",
            },
            "SIDEWAYS": {
                "risk_tolerance": "medium",
                "position_size_multiplier": 0.8,
                "stop_loss_pct": 0.06,
                "take_profit_pct": 0.10,
                "strategy": "range_trading",
                "description": "Sideways market: Range trading, moderate risk, quick profits",
            },
            "VOLATILE": {
                "risk_tolerance": "very_low",
                "position_size_multiplier": 0.5,
                "stop_loss_pct": 0.04,  # Very tight stops
                "take_profit_pct": 0.06,
                "strategy": "defensive",
                "description": "Volatile market: Defensive, small positions, very tight stops",
            },
        }

        return adjustments.get(regime, adjustments["SIDEWAYS"])

    def get_regime_history(self, limit: int = 50) -> list[dict]:
        """Get recent regime history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, regime, confidence, mean_return, volatility, trend_strength
            FROM regime_history
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append(
                {
                    "timestamp": row[0],
                    "regime": row[1],
                    "confidence": row[2],
                    "features": {
                        "mean_return": row[3],
                        "volatility": row[4],
                        "trend_strength": row[5],
                    },
                }
            )

        return history

    def get_regime_distribution(self, days: int = 30) -> dict:
        """Get regime distribution over last N days."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT regime, COUNT(*) as count
            FROM regime_history
            WHERE timestamp > ?
            GROUP BY regime
        """,
            (cutoff,),
        )

        rows = cursor.fetchall()
        conn.close()

        total = sum(row[1] for row in rows)

        if total == 0:
            return {r: 0.0 for r in self.regimes}

        distribution = {r: 0.0 for r in self.regimes}
        for regime, count in rows:
            distribution[regime] = round(count / total, 3)

        return distribution


# Singleton instance
_regime_detector: RegimeDetector | None = None


def get_regime_detector() -> RegimeDetector:
    """Get singleton regime detector instance."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector()
    return _regime_detector
