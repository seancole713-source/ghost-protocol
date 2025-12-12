#!/usr/bin/env python3
"""
Ghost Protocol - Real-Time Accuracy Feedback Loop (Task #4)
===========================================================
Continuously learns from prediction outcomes to improve accuracy.

Features:
- Track prediction success/failure rates
- Auto-adjust feature weights based on performance
- Boost signal importance for winning patterns
- Reduce emphasis on losing patterns
- Update ensemble model weights dynamically

Target: +8-12% accuracy improvement over 3-5 days
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "feedback_loop.db"


@dataclass
class PredictionOutcome:
    """Single prediction outcome for learning"""
    prediction_id: int
    symbol: str
    direction: str  # UP/DOWN/FLAT
    confidence: float
    predicted_price: float
    actual_price: float
    was_correct: bool
    accuracy_pct: float  # How close was the prediction
    signals_used: list[str]  # Which signals fired
    features: dict[str, float]  # Feature values at prediction time
    timestamp: float


@dataclass
class FeaturePerformance:
    """Track how well a feature predicts outcomes"""
    feature_name: str
    total_predictions: int
    correct_predictions: int
    accuracy_rate: float
    avg_confidence_when_correct: float
    avg_confidence_when_wrong: float
    weight_adjustment: float  # -1.0 to +1.0


class FeedbackLoop:
    """Real-time learning system that improves predictions"""
    
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()
        
        # In-memory cache for fast lookups
        self.feature_weights = {}  # feature_name -> weight multiplier
        self.signal_performance = defaultdict(lambda: {"correct": 0, "total": 0})
        self.recent_outcomes = deque(maxlen=1000)  # Last 1000 predictions
        
        # Load existing weights
        self._load_weights()
    
    def _init_db(self):
        """Initialize feedback database"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Outcomes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL NOT NULL,
                    was_correct INTEGER NOT NULL,
                    accuracy_pct REAL NOT NULL,
                    signals_used TEXT,
                    features TEXT,
                    timestamp REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Feature weights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_weights (
                    feature_name TEXT PRIMARY KEY,
                    weight_multiplier REAL NOT NULL,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy_rate REAL DEFAULT 0.5,
                    last_updated REAL NOT NULL
                )
            """)
            
            # Signal performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_performance (
                    signal_name TEXT PRIMARY KEY,
                    correct_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    accuracy_rate REAL DEFAULT 0.5,
                    avg_confidence_boost REAL DEFAULT 0.0,
                    last_updated REAL NOT NULL
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_symbol ON prediction_outcomes(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_timestamp ON prediction_outcomes(timestamp)")
            
            conn.commit()
            logger.info(f"✅ Feedback loop initialized: {self.db_path}")
    
    def _load_weights(self):
        """Load feature weights from database"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT feature_name, weight_multiplier FROM feature_weights").fetchall()
            self.feature_weights = {name: weight for name, weight in rows}
            
            signal_rows = conn.execute(
                "SELECT signal_name, correct_count, total_count FROM signal_performance"
            ).fetchall()
            for signal, correct, total in signal_rows:
                self.signal_performance[signal] = {"correct": correct, "total": total}
        
        if self.feature_weights:
            logger.info(f"📊 Loaded {len(self.feature_weights)} feature weights from database")
    
    def record_outcome(self, outcome: PredictionOutcome) -> None:
        """Record a prediction outcome and trigger learning"""
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prediction_outcomes (
                    prediction_id, symbol, direction, confidence, predicted_price,
                    actual_price, was_correct, accuracy_pct, signals_used, features, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.prediction_id,
                outcome.symbol,
                outcome.direction,
                outcome.confidence,
                outcome.predicted_price,
                outcome.actual_price,
                1 if outcome.was_correct else 0,
                outcome.accuracy_pct,
                json.dumps(outcome.signals_used),
                json.dumps(outcome.features),
                outcome.timestamp
            ))
            conn.commit()
        
        # Add to recent cache
        self.recent_outcomes.append(outcome)
        
        # Update signal performance
        for signal in outcome.signals_used:
            self.signal_performance[signal]["total"] += 1
            if outcome.was_correct:
                self.signal_performance[signal]["correct"] += 1
        
        # Trigger learning if we have enough data
        if len(self.recent_outcomes) >= 50 and len(self.recent_outcomes) % 10 == 0:
            self._update_feature_weights()
        
        logger.info(
            f"📝 Recorded outcome: {outcome.symbol} {outcome.direction} "
            f"{'✅' if outcome.was_correct else '❌'} ({outcome.accuracy_pct:.1f}% accurate)"
        )
    
    def _update_feature_weights(self):
        """Update feature weights based on recent performance"""
        # Group outcomes by features that were present
        feature_outcomes = defaultdict(lambda: {"correct": 0, "total": 0, "confidence_sum": 0})
        
        for outcome in self.recent_outcomes:
            for feature_name, feature_value in outcome.features.items():
                if abs(feature_value) > 0.01:  # Feature was significant
                    feature_outcomes[feature_name]["total"] += 1
                    feature_outcomes[feature_name]["confidence_sum"] += outcome.confidence
                    if outcome.was_correct:
                        feature_outcomes[feature_name]["correct"] += 1
        
        # Calculate new weights
        updates = []
        for feature_name, stats in feature_outcomes.items():
            if stats["total"] < 10:  # Need minimum sample size
                continue
            
            accuracy_rate = stats["correct"] / stats["total"]
            
            # Weight adjustment: +20% for 70%+ accuracy, -20% for <40% accuracy
            if accuracy_rate >= 0.70:
                weight_multiplier = 1.20
            elif accuracy_rate >= 0.60:
                weight_multiplier = 1.10
            elif accuracy_rate >= 0.50:
                weight_multiplier = 1.00
            elif accuracy_rate >= 0.40:
                weight_multiplier = 0.90
            else:
                weight_multiplier = 0.80
            
            # Store in cache
            old_weight = self.feature_weights.get(feature_name, 1.0)
            self.feature_weights[feature_name] = weight_multiplier
            
            updates.append((
                feature_name,
                weight_multiplier,
                stats["total"],
                stats["correct"],
                accuracy_rate,
                time.time()
            ))
            
            if abs(weight_multiplier - old_weight) > 0.05:
                logger.info(
                    f"🎯 Feature '{feature_name}' weight: {old_weight:.2f} → {weight_multiplier:.2f} "
                    f"(accuracy: {accuracy_rate:.1%}, n={stats['total']})"
                )
        
        # Batch update database
        if updates:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO feature_weights 
                    (feature_name, weight_multiplier, total_predictions, correct_predictions, accuracy_rate, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, updates)
                conn.commit()
            
            logger.info(f"✅ Updated {len(updates)} feature weights")
    
    def get_adjusted_features(self, features: dict[str, float]) -> dict[str, float]:
        """Apply learned weights to features"""
        if not self.feature_weights:
            return features  # No adjustments yet
        
        adjusted = {}
        adjustments_made = 0
        
        for feature_name, value in features.items():
            weight = self.feature_weights.get(feature_name, 1.0)
            adjusted[feature_name] = value * weight
            if abs(weight - 1.0) > 0.01:
                adjustments_made += 1
        
        if adjustments_made > 0:
            logger.debug(f"🔧 Applied {adjustments_made} feature weight adjustments")
        
        return adjusted
    
    def get_signal_boost(self, signal_name: str) -> float:
        """Get confidence boost for a signal based on performance"""
        stats = self.signal_performance.get(signal_name)
        if not stats or stats["total"] < 20:
            return 0.0  # Need minimum sample size
        
        accuracy = stats["correct"] / stats["total"]
        
        # Boost high-performing signals, penalize low-performing
        if accuracy >= 0.75:
            return 0.12  # +12% for very accurate signals
        elif accuracy >= 0.65:
            return 0.08  # +8% for good signals
        elif accuracy >= 0.55:
            return 0.05  # +5% for decent signals
        elif accuracy >= 0.45:
            return 0.0   # Neutral
        else:
            return -0.05  # -5% penalty for inaccurate signals
    
    def get_performance_report(self, days: int = 7) -> dict[str, Any]:
        """Generate performance report"""
        cutoff = time.time() - (days * 86400)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall stats
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(was_correct) as correct,
                    AVG(accuracy_pct) as avg_accuracy,
                    AVG(confidence) as avg_confidence
                FROM prediction_outcomes
                WHERE timestamp >= ?
            """, (cutoff,)).fetchone()
            
            # Top performing features
            top_features = conn.execute("""
                SELECT feature_name, accuracy_rate, total_predictions, weight_multiplier
                FROM feature_weights
                WHERE total_predictions >= 10
                ORDER BY accuracy_rate DESC
                LIMIT 10
            """).fetchall()
            
            # Top signals
            top_signals = conn.execute("""
                SELECT signal_name, accuracy_rate, total_count
                FROM signal_performance
                WHERE total_count >= 10
                ORDER BY accuracy_rate DESC
                LIMIT 10
            """).fetchall()
        
        total, correct, avg_accuracy, avg_confidence = stats or (0, 0, 0, 0)
        
        return {
            "period_days": days,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy_rate": correct / total if total > 0 else 0,
            "avg_accuracy_pct": avg_accuracy or 0,
            "avg_confidence": avg_confidence or 0,
            "top_features": [
                {
                    "name": name,
                    "accuracy": acc,
                    "predictions": n,
                    "weight": weight
                } for name, acc, n, weight in (top_features or [])
            ],
            "top_signals": [
                {
                    "name": name,
                    "accuracy": acc,
                    "count": n
                } for name, acc, n in (top_signals or [])
            ],
            "learning_status": "active" if len(self.feature_weights) > 0 else "training"
        }
    
    def update_ensemble_weights(self, model_name: str, was_correct: bool):
        """Update ensemble model performance"""
        from core.ensemble_predictor import get_ensemble_predictor
        
        ensemble = get_ensemble_predictor()
        ensemble.update_performance(model_name, was_correct)
        
        logger.debug(f"📊 Updated ensemble: {model_name} = {'✅' if was_correct else '❌'}")


# Global instance
_feedback_loop: FeedbackLoop | None = None


def get_feedback_loop() -> FeedbackLoop:
    """Get or create global feedback loop"""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
        logger.info("🔄 Feedback loop initialized - continuous learning enabled")
    return _feedback_loop


if __name__ == "__main__":
    # Test feedback loop
    logging.basicConfig(level=logging.INFO)
    
    print("🔄 Testing Feedback Loop")
    print("=" * 60)
    
    loop = get_feedback_loop()
    
    # Simulate some outcomes
    for i in range(100):
        outcome = PredictionOutcome(
            prediction_id=i,
            symbol="BTC" if i % 2 == 0 else "ETH",
            direction="UP" if i % 3 != 0 else "DOWN",
            confidence=0.5 + (i % 40) / 100,
            predicted_price=50000.0,
            actual_price=50100.0 if i % 3 != 0 else 49900.0,
            was_correct=i % 3 != 0,
            accuracy_pct=98.0 if i % 3 != 0 else 95.0,
            signals_used=["RSI_OVERSOLD", "MACD_BULLISH"] if i % 3 != 0 else ["RSI_OVERBOUGHT"],
            features={"RSI_14": 30 + i % 40, "MACD_HISTOGRAM": 0.5},
            timestamp=time.time() - (100 - i) * 3600
        )
        loop.record_outcome(outcome)
    
    # Generate report
    report = loop.get_performance_report(days=7)
    
    print(f"\n📊 Performance Report:")
    print(f"  Total Predictions: {report['total_predictions']}")
    print(f"  Accuracy Rate: {report['accuracy_rate']:.1%}")
    print(f"  Avg Confidence: {report['avg_confidence']:.1%}")
    print(f"  Learning Status: {report['learning_status']}")
    
    print(f"\n🎯 Top Features:")
    for feat in report['top_features'][:5]:
        print(f"  {feat['name']}: {feat['accuracy']:.1%} accuracy, weight={feat['weight']:.2f}")
    
    print(f"\n🚀 Top Signals:")
    for sig in report['top_signals'][:5]:
        print(f"  {sig['name']}: {sig['accuracy']:.1%} ({sig['count']} predictions)")
    
    print("\n✅ Feedback loop test complete")
