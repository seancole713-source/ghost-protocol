"""
GHOST Trust Ladder System
========================

Progressive accuracy system that promotes symbols through trust levels:

LEVEL 1 (Default): 48hr predictions
- Check accuracy at 48hr mark
- If WIN → promote to Level 2
- If LOSS → stay at Level 1

LEVEL 2 (Extended): 120hr predictions (5 days)
- Check at 60hr mark (midpoint checkpoint)
- Check at 120hr mark (final)
- Both WIN → promote to Level 3
- Any LOSS → demote to Level 1

LEVEL 3 (Focused): Symbol gets priority treatment
- Higher confidence boost
- Priority in TOP 10 selection
- Stays focused until 2 consecutive losses
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

LOGGER = logging.getLogger("ghost.trust_ladder")

# Trust level configurations
TRUST_LEVELS = {
    1: {
        "name": "Standard",
        "prediction_hours": 48,
        "checkpoints": [48],  # Single check at 48hr
        "wins_to_promote": 1,
        "confidence_boost": 1.0,  # No boost
    },
    2: {
        "name": "Extended",
        "prediction_hours": 120,  # 5 days
        "checkpoints": [60, 120],  # Check at 60hr and 120hr
        "wins_to_promote": 2,  # Need both checkpoints to pass
        "confidence_boost": 1.10,  # 10% boost
    },
    3: {
        "name": "Focused",
        "prediction_hours": 168,  # 7 days
        "checkpoints": [72, 168],  # Check at 3 days and 7 days
        "wins_to_promote": None,  # Max level
        "confidence_boost": 1.20,  # 20% boost
        "top10_priority": True,  # Gets priority in TOP 10
    }
}

# How many losses before demotion
LOSSES_TO_DEMOTE = 2

# PostgreSQL availability
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


@dataclass
class SymbolTrust:
    """Trust data for a symbol"""
    symbol: str
    trust_level: int
    consecutive_wins: int
    consecutive_losses: int
    checkpoint_wins: int  # Wins in current checkpoint cycle
    total_predictions: int
    total_wins: int
    last_updated: datetime
    
    @property
    def accuracy_pct(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return (self.total_wins / self.total_predictions) * 100
    
    @property
    def level_config(self) -> dict:
        return TRUST_LEVELS.get(self.trust_level, TRUST_LEVELS[1])
    
    @property
    def confidence_boost(self) -> float:
        return self.level_config.get("confidence_boost", 1.0)
    
    @property
    def is_focused(self) -> bool:
        return self.trust_level >= 3


class TrustLadder:
    """Manages symbol trust levels and promotion/demotion logic."""
    
    def __init__(self):
        self.use_postgres = PSYCOPG2_AVAILABLE and os.getenv("DATABASE_URL")
        self._ensure_table()
        self._cache: Dict[str, SymbolTrust] = {}
        self._cache_time = 0
        self._cache_ttl = 300  # 5 minutes
    
    def _get_postgres_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(os.getenv("DATABASE_URL"))
    
    def _ensure_table(self):
        """Create trust ladder table if not exists."""
        if not self.use_postgres:
            LOGGER.warning("[TRUST] PostgreSQL not available, using memory only")
            return
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ghost_symbol_trust (
                    symbol VARCHAR(20) PRIMARY KEY,
                    trust_level INTEGER DEFAULT 1,
                    consecutive_wins INTEGER DEFAULT 0,
                    consecutive_losses INTEGER DEFAULT 0,
                    checkpoint_wins INTEGER DEFAULT 0,
                    total_predictions INTEGER DEFAULT 0,
                    total_wins INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
            cur.close()
            conn.close()
            LOGGER.info("[TRUST] ✅ ghost_symbol_trust table ready")
        except Exception as e:
            LOGGER.error(f"[TRUST] Failed to create table: {e}")
    
    def get_trust(self, symbol: str) -> SymbolTrust:
        """Get trust data for a symbol (creates default if not exists)."""
        symbol = symbol.upper()
        
        if not self.use_postgres:
            # Memory-only mode
            if symbol not in self._cache:
                self._cache[symbol] = SymbolTrust(
                    symbol=symbol,
                    trust_level=1,
                    consecutive_wins=0,
                    consecutive_losses=0,
                    checkpoint_wins=0,
                    total_predictions=0,
                    total_wins=0,
                    last_updated=datetime.utcnow()
                )
            return self._cache[symbol]
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT symbol, trust_level, consecutive_wins, consecutive_losses,
                       checkpoint_wins, total_predictions, total_wins, last_updated
                FROM ghost_symbol_trust
                WHERE symbol = %s
            """, (symbol,))
            
            row = cur.fetchone()
            
            if row:
                trust = SymbolTrust(
                    symbol=row[0],
                    trust_level=row[1],
                    consecutive_wins=row[2],
                    consecutive_losses=row[3],
                    checkpoint_wins=row[4],
                    total_predictions=row[5],
                    total_wins=row[6],
                    last_updated=row[7]
                )
            else:
                # Create new entry
                cur.execute("""
                    INSERT INTO ghost_symbol_trust (symbol)
                    VALUES (%s)
                    ON CONFLICT (symbol) DO NOTHING
                """, (symbol,))
                conn.commit()
                trust = SymbolTrust(
                    symbol=symbol,
                    trust_level=1,
                    consecutive_wins=0,
                    consecutive_losses=0,
                    checkpoint_wins=0,
                    total_predictions=0,
                    total_wins=0,
                    last_updated=datetime.utcnow()
                )
            
            cur.close()
            conn.close()
            return trust
            
        except Exception as e:
            LOGGER.error(f"[TRUST] Failed to get trust for {symbol}: {e}")
            return SymbolTrust(
                symbol=symbol,
                trust_level=1,
                consecutive_wins=0,
                consecutive_losses=0,
                checkpoint_wins=0,
                total_predictions=0,
                total_wins=0,
                last_updated=datetime.utcnow()
            )
    
    def record_outcome(self, symbol: str, is_win: bool, is_checkpoint: bool = False) -> Dict:
        """
        Record prediction outcome and handle promotion/demotion.
        
        MULTI-CHECKPOINT LOGIC:
        - is_checkpoint=True: Intermediate checkpoint (e.g., 60hr of 120hr window)
          - WIN: Increment checkpoint_wins, no promotion yet
          - LOSS: Immediate demotion to Level 1
        - is_checkpoint=False: Final checkpoint (e.g., 120hr of 120hr window)
          - WIN: Check if all checkpoints passed → promote
          - LOSS: Demotion to Level 1
        
        Args:
            symbol: The symbol
            is_win: Whether the prediction was correct
            is_checkpoint: True for intermediate checkpoints, False for final
            
        Returns:
            Dict with new trust state and any level changes
        """
        symbol = symbol.upper()
        trust = self.get_trust(symbol)
        
        old_level = trust.trust_level
        level_config = TRUST_LEVELS[trust.trust_level]
        
        # Update stats
        trust.total_predictions += 1
        if is_win:
            trust.total_wins += 1
            trust.consecutive_wins += 1
            trust.consecutive_losses = 0
            trust.checkpoint_wins += 1
            
            LOGGER.info(
                f"[TRUST] [{symbol}] Checkpoint WIN - checkpoint_wins={trust.checkpoint_wins}, "
                f"is_checkpoint={is_checkpoint}, level={trust.trust_level}"
            )
        else:
            trust.consecutive_wins = 0
            trust.consecutive_losses += 1
            trust.checkpoint_wins = 0  # Reset checkpoint progress on any loss
            
            LOGGER.info(
                f"[TRUST] [{symbol}] Checkpoint LOSS - resetting checkpoint_wins, "
                f"is_checkpoint={is_checkpoint}, level={trust.trust_level}"
            )
        
        # Check for promotion (ONLY on final checkpoint, not intermediate)
        promoted = False
        demoted = False
        
        if not is_checkpoint and is_win:
            # Final checkpoint - check if we can promote
            wins_needed = level_config.get("wins_to_promote")
            if wins_needed and trust.checkpoint_wins >= wins_needed:
                # All checkpoints passed - can promote to next level
                if trust.trust_level < 3:
                    trust.trust_level += 1
                    trust.checkpoint_wins = 0  # Reset for new level
                    promoted = True
                    LOGGER.info(
                        f"[TRUST] 🚀 {symbol} PROMOTED to Level {trust.trust_level} "
                        f"({TRUST_LEVELS[trust.trust_level]['name']}) - ALL checkpoints passed!"
                    )
        
        # Check for demotion (on ANY loss, intermediate or final)
        if not is_win:
            if trust.trust_level > 1:
                trust.trust_level = 1  # Demote back to Level 1
                trust.consecutive_losses = 0
                trust.checkpoint_wins = 0
                demoted = True
                checkpoint_type = "intermediate" if is_checkpoint else "final"
                LOGGER.info(
                    f"[TRUST] 📉 {symbol} DEMOTED to Level 1 (Standard) - "
                    f"failed {checkpoint_type} checkpoint"
                )
        
        trust.last_updated = datetime.utcnow()
        
        # Save to database
        self._save_trust(trust)
        
        return {
            "symbol": symbol,
            "is_win": is_win,
            "is_checkpoint": is_checkpoint,
            "old_level": old_level,
            "new_level": trust.trust_level,
            "promoted": promoted,
            "demoted": demoted,
            "consecutive_wins": trust.consecutive_wins,
            "consecutive_losses": trust.consecutive_losses,
            "checkpoint_wins": trust.checkpoint_wins,
            "accuracy_pct": trust.accuracy_pct,
            "confidence_boost": trust.confidence_boost,
            "is_focused": trust.is_focused
        }
    
    def _save_trust(self, trust: SymbolTrust):
        """Save trust data to database."""
        if not self.use_postgres:
            self._cache[trust.symbol] = trust
            return
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ghost_symbol_trust 
                    (symbol, trust_level, consecutive_wins, consecutive_losses,
                     checkpoint_wins, total_predictions, total_wins, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    trust_level = EXCLUDED.trust_level,
                    consecutive_wins = EXCLUDED.consecutive_wins,
                    consecutive_losses = EXCLUDED.consecutive_losses,
                    checkpoint_wins = EXCLUDED.checkpoint_wins,
                    total_predictions = EXCLUDED.total_predictions,
                    total_wins = EXCLUDED.total_wins,
                    last_updated = EXCLUDED.last_updated
            """, (
                trust.symbol,
                trust.trust_level,
                trust.consecutive_wins,
                trust.consecutive_losses,
                trust.checkpoint_wins,
                trust.total_predictions,
                trust.total_wins,
                trust.last_updated
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            LOGGER.error(f"[TRUST] Failed to save trust for {trust.symbol}: {e}")
    
    def get_prediction_window(self, symbol: str) -> Dict:
        """
        Get the prediction window configuration for a symbol based on trust level.
        
        Returns:
            {
                "prediction_hours": int (48, 120, or 168),
                "checkpoints": list of hours to check,
                "confidence_boost": float,
                "trust_level": int,
                "level_name": str
            }
        """
        trust = self.get_trust(symbol)
        config = TRUST_LEVELS[trust.trust_level]
        
        return {
            "symbol": symbol,
            "prediction_hours": config["prediction_hours"],
            "checkpoints": config["checkpoints"],
            "confidence_boost": config["confidence_boost"],
            "trust_level": trust.trust_level,
            "level_name": config["name"],
            "is_focused": trust.is_focused,
            "consecutive_wins": trust.consecutive_wins,
            "accuracy_pct": trust.accuracy_pct
        }
    
    def get_focused_symbols(self) -> list:
        """Get all symbols at Level 3 (Focused)."""
        if not self.use_postgres:
            return [s for s, t in self._cache.items() if t.trust_level >= 3]
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol, consecutive_wins, accuracy_pct
                FROM ghost_symbol_trust
                WHERE trust_level >= 3
                ORDER BY consecutive_wins DESC
            """)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            return [{"symbol": r[0], "consecutive_wins": r[1]} for r in rows]
        except Exception as e:
            LOGGER.error(f"[TRUST] Failed to get focused symbols: {e}")
            return []
    
    def get_leaderboard(self, limit: int = 20) -> list:
        """Get trust leaderboard sorted by level and accuracy."""
        if not self.use_postgres:
            trusts = sorted(
                self._cache.values(),
                key=lambda t: (t.trust_level, t.accuracy_pct),
                reverse=True
            )[:limit]
            return [
                {
                    "symbol": t.symbol,
                    "trust_level": t.trust_level,
                    "level_name": TRUST_LEVELS[t.trust_level]["name"],
                    "accuracy_pct": t.accuracy_pct,
                    "consecutive_wins": t.consecutive_wins,
                    "total_predictions": t.total_predictions
                }
                for t in trusts
            ]
        
        try:
            conn = self._get_postgres_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol, trust_level, consecutive_wins, total_predictions, total_wins
                FROM ghost_symbol_trust
                WHERE total_predictions >= 5
                ORDER BY trust_level DESC, 
                         (total_wins::float / NULLIF(total_predictions, 0)) DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            return [
                {
                    "symbol": r[0],
                    "trust_level": r[1],
                    "level_name": TRUST_LEVELS[r[1]]["name"],
                    "consecutive_wins": r[2],
                    "total_predictions": r[3],
                    "total_wins": r[4],
                    "accuracy_pct": (r[4] / r[3] * 100) if r[3] > 0 else 0
                }
                for r in rows
            ]
        except Exception as e:
            LOGGER.error(f"[TRUST] Failed to get leaderboard: {e}")
            return []


# Global instance
_trust_ladder: Optional[TrustLadder] = None


def get_trust_ladder() -> TrustLadder:
    """Get or create the global TrustLadder instance."""
    global _trust_ladder
    if _trust_ladder is None:
        _trust_ladder = TrustLadder()
    return _trust_ladder


def get_symbol_prediction_window(symbol: str) -> Dict:
    """Convenience function to get prediction window for a symbol."""
    return get_trust_ladder().get_prediction_window(symbol)


def record_prediction_outcome(symbol: str, is_win: bool, is_checkpoint: bool = False) -> Dict:
    """Convenience function to record an outcome."""
    return get_trust_ladder().record_outcome(symbol, is_win, is_checkpoint)
