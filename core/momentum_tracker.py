#!/usr/bin/env python3
"""
Ghost Protocol - Momentum Score System
========================================

Tracks prediction confidence changes over time to identify strengthening/weakening signals.

Momentum Classification:
- HOT 🔥: Confidence rising +5% or more (strong buy/sell signal strengthening)
- WARMING 📈: Confidence rising +2-5% (signal improving)
- STABLE ➡️: Confidence change between -2% and +2% (steady signal)
- COOLING 📉: Confidence falling -2% to -5% (signal weakening)
- COLD ❄️: Confidence falling -5% or more (signal collapsing)

Usage:
    from core.momentum_tracker import get_momentum_tracker
    
    tracker = get_momentum_tracker()
    momentum = tracker.calculate_momentum(symbol="BTC", current_confidence=0.72)
    
    # Returns:
    # {
    #     "status": "HOT",
    #     "emoji": "🔥",
    #     "arrow": "↗️",
    #     "confidence_delta": 0.08,
    #     "confidence_delta_pct": 8.0,
    #     "description": "Signal strengthening rapidly",
    #     "alert_worthy": True
    # }
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger("core.momentum_tracker")

# Momentum thresholds (percentage change)
MOMENTUM_HOT_THRESHOLD = 5.0  # +5% confidence increase
MOMENTUM_WARMING_THRESHOLD = 2.0  # +2% confidence increase
MOMENTUM_COOLING_THRESHOLD = -2.0  # -2% confidence decrease
MOMENTUM_COLD_THRESHOLD = -5.0  # -5% confidence decrease

# Alert thresholds
ALERT_ON_HOT = True
ALERT_ON_COLD = True

# Database path (same as predictions)
DB_PATH = Path("./data/ghost_predictions.db")


class MomentumStatus:
    """Momentum status classification"""
    
    HOT = "HOT"
    WARMING = "WARMING"
    STABLE = "STABLE"
    COOLING = "COOLING"
    COLD = "COLD"
    
    EMOJIS = {
        "HOT": "🔥",
        "WARMING": "📈",
        "STABLE": "➡️",
        "COOLING": "📉",
        "COLD": "❄️"
    }
    
    ARROWS = {
        "HOT": "↗️",
        "WARMING": "↗️",
        "STABLE": "→",
        "COOLING": "↘️",
        "COLD": "↘️"
    }
    
    DESCRIPTIONS = {
        "HOT": "Signal strengthening rapidly",
        "WARMING": "Signal gaining confidence",
        "STABLE": "Signal holding steady",
        "COOLING": "Signal weakening",
        "COLD": "Signal collapsing"
    }


class MomentumTracker:
    """
    Track prediction momentum by comparing recent confidence levels.
    
    Analyzes the last 3-5 predictions for the same symbol to determine
    if the signal is getting stronger (bullish for UP predictions, bearish for DOWN)
    or weaker (losing confidence).
    """
    
    def __init__(self, db_path: Path | str | None = None):
        """
        Initialize momentum tracker.
        
        Args:
            db_path: Path to SQLite database (defaults to ghost_predictions.db)
        """
        self.db_path = Path(db_path) if db_path else DB_PATH
        self._ensure_momentum_table()
    
    def _ensure_momentum_table(self):
        """Create momentum_history table if it doesn't exist"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS momentum_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    direction TEXT NOT NULL,
                    momentum_status TEXT,
                    confidence_delta REAL,
                    confidence_delta_pct REAL,
                    previous_confidence REAL,
                    lookback_count INTEGER,
                    INDEX idx_momentum_symbol_time (symbol, timestamp DESC)
                )
            """)
            conn.commit()
            conn.close()
            
            LOGGER.debug(f"Momentum tracker initialized (DB: {self.db_path})")
        except Exception as e:
            LOGGER.error(f"Failed to create momentum_history table: {e}")
    
    def _get_recent_predictions(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Get recent predictions for a symbol from ghost_predictions table.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            limit: Number of recent predictions to retrieve
        
        Returns:
            List of predictions sorted by timestamp (newest first)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT 
                    symbol,
                    predicted_at as timestamp,
                    confidence,
                    predicted_direction as direction
                FROM ghost_predictions
                WHERE symbol = ?
                ORDER BY predicted_at DESC
                LIMIT ?
            """, (symbol, limit))
            
            predictions = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return predictions
        except Exception as e:
            LOGGER.error(f"Failed to retrieve recent predictions for {symbol}: {e}")
            return []
    
    def calculate_momentum(
        self,
        symbol: str,
        current_confidence: float,
        current_direction: str = "UP"
    ) -> dict[str, Any]:
        """
        Calculate momentum status for current prediction.
        
        Compares current confidence with recent predictions to determine
        if the signal is strengthening (HOT/WARMING) or weakening (COOLING/COLD).
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC, ETH)
            current_confidence: Current prediction confidence (0.0-1.0)
            current_direction: Prediction direction (UP, DOWN, FLAT)
        
        Returns:
            Momentum data dictionary with status, emoji, delta, etc.
        """
        # Get recent predictions (last 3-5)
        recent = self._get_recent_predictions(symbol, limit=5)
        
        # Default to STABLE if no history
        if not recent or len(recent) < 2:
            return {
                "status": MomentumStatus.STABLE,
                "emoji": MomentumStatus.EMOJIS[MomentumStatus.STABLE],
                "arrow": MomentumStatus.ARROWS[MomentumStatus.STABLE],
                "confidence_delta": 0.0,
                "confidence_delta_pct": 0.0,
                "description": "Insufficient history",
                "alert_worthy": False,
                "previous_confidence": None,
                "lookback_count": 0
            }
        
        # Calculate average confidence from last 3 predictions
        lookback = min(3, len(recent))
        previous_confidences = [p["confidence"] for p in recent[:lookback]]
        avg_previous_confidence = sum(previous_confidences) / len(previous_confidences)
        
        # Calculate confidence delta
        confidence_delta = current_confidence - avg_previous_confidence
        confidence_delta_pct = (confidence_delta / avg_previous_confidence) * 100 if avg_previous_confidence > 0 else 0.0
        
        # Classify momentum status
        if confidence_delta_pct >= MOMENTUM_HOT_THRESHOLD:
            status = MomentumStatus.HOT
            alert_worthy = ALERT_ON_HOT
        elif confidence_delta_pct >= MOMENTUM_WARMING_THRESHOLD:
            status = MomentumStatus.WARMING
            alert_worthy = False
        elif confidence_delta_pct <= MOMENTUM_COLD_THRESHOLD:
            status = MomentumStatus.COLD
            alert_worthy = ALERT_ON_COLD
        elif confidence_delta_pct <= MOMENTUM_COOLING_THRESHOLD:
            status = MomentumStatus.COOLING
            alert_worthy = False
        else:
            status = MomentumStatus.STABLE
            alert_worthy = False
        
        momentum_data = {
            "status": status,
            "emoji": MomentumStatus.EMOJIS[status],
            "arrow": MomentumStatus.ARROWS[status],
            "confidence_delta": round(confidence_delta, 4),
            "confidence_delta_pct": round(confidence_delta_pct, 2),
            "description": MomentumStatus.DESCRIPTIONS[status],
            "alert_worthy": alert_worthy,
            "previous_confidence": round(avg_previous_confidence, 4),
            "lookback_count": lookback
        }
        
        # Store momentum history
        self._record_momentum(symbol, current_confidence, current_direction, momentum_data)
        
        return momentum_data
    
    def _record_momentum(
        self,
        symbol: str,
        confidence: float,
        direction: str,
        momentum_data: dict[str, Any]
    ):
        """Record momentum calculation in history table"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                INSERT INTO momentum_history (
                    symbol, timestamp, confidence, direction,
                    momentum_status, confidence_delta, confidence_delta_pct,
                    previous_confidence, lookback_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                int(time.time()),
                confidence,
                direction,
                momentum_data["status"],
                momentum_data["confidence_delta"],
                momentum_data["confidence_delta_pct"],
                momentum_data.get("previous_confidence"),
                momentum_data["lookback_count"]
            ))
            conn.commit()
            conn.close()
            
            LOGGER.debug(
                f"[{symbol}] Momentum recorded: {momentum_data['status']} "
                f"({momentum_data['confidence_delta_pct']:+.1f}%)"
            )
        except Exception as e:
            LOGGER.error(f"Failed to record momentum for {symbol}: {e}")
    
    def get_momentum_history(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get momentum history for a symbol.
        
        Args:
            symbol: Cryptocurrency symbol
            limit: Number of history entries to retrieve
        
        Returns:
            List of momentum history entries
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT *
                FROM momentum_history
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, limit))
            
            history = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return history
        except Exception as e:
            LOGGER.error(f"Failed to get momentum history for {symbol}: {e}")
            return []
    
    def get_hot_signals(self, min_confidence: float = 0.65) -> list[dict[str, Any]]:
        """
        Get all HOT momentum signals across all symbols.
        
        Args:
            min_confidence: Minimum confidence threshold (default 0.65)
        
        Returns:
            List of HOT signals with symbol, confidence, momentum data
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            # Get latest momentum status for each symbol
            cursor = conn.execute("""
                WITH latest_momentum AS (
                    SELECT 
                        symbol,
                        timestamp,
                        confidence,
                        direction,
                        momentum_status,
                        confidence_delta_pct,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                    FROM momentum_history
                )
                SELECT 
                    symbol,
                    timestamp,
                    confidence,
                    direction,
                    momentum_status,
                    confidence_delta_pct
                FROM latest_momentum
                WHERE rn = 1
                  AND momentum_status = 'HOT'
                  AND confidence >= ?
                ORDER BY confidence_delta_pct DESC
            """, (min_confidence,))
            
            hot_signals = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return hot_signals
        except Exception as e:
            LOGGER.error(f"Failed to get HOT signals: {e}")
            return []
    
    def get_cold_signals(self, max_confidence: float = 0.55) -> list[dict[str, Any]]:
        """
        Get all COLD momentum signals (signals losing confidence).
        
        Args:
            max_confidence: Maximum confidence threshold (default 0.55)
        
        Returns:
            List of COLD signals
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                WITH latest_momentum AS (
                    SELECT 
                        symbol,
                        timestamp,
                        confidence,
                        direction,
                        momentum_status,
                        confidence_delta_pct,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                    FROM momentum_history
                )
                SELECT 
                    symbol,
                    timestamp,
                    confidence,
                    direction,
                    momentum_status,
                    confidence_delta_pct
                FROM latest_momentum
                WHERE rn = 1
                  AND momentum_status = 'COLD'
                  AND confidence <= ?
                ORDER BY confidence_delta_pct ASC
            """, (max_confidence,))
            
            cold_signals = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return cold_signals
        except Exception as e:
            LOGGER.error(f"Failed to get COLD signals: {e}")
            return []


# Singleton instance
_MOMENTUM_TRACKER: Optional[MomentumTracker] = None


def get_momentum_tracker(db_path: Path | str | None = None) -> MomentumTracker:
    """
    Get singleton momentum tracker instance.
    
    Args:
        db_path: Optional database path (only used on first call)
    
    Returns:
        MomentumTracker instance
    """
    global _MOMENTUM_TRACKER
    if _MOMENTUM_TRACKER is None:
        _MOMENTUM_TRACKER = MomentumTracker(db_path)
    return _MOMENTUM_TRACKER
