#!/usr/bin/env python3
"""
🏆 GHOST PROTOCOL V3 - DYNAMIC COMPETITION SYSTEM

NO BLACKLIST - Everyone Gets a Fair Chance!

Philosophy:
- All assets compete in a pool
- Ghost makes "shadow predictions" for EVERYONE
- Only TOP 10 stocks + TOP 10 crypto get sent to Telegram
- Constant competition - pending assets fight to get into TOP 10
- If you drop below threshold, you're demoted back to pending
- If pending asset beats a TOP 10, they swap places

This is a self-improving, fair system where performance = rank.

Database Tables:
- v3_competition_pool: All assets and their performance metrics
- v3_shadow_predictions: Silent predictions for ranking purposes
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

LOGGER = logging.getLogger("ghost.v3_competition")

DATABASE_URL = os.getenv("DATABASE_URL")


class CompetitionTier(Enum):
    """Competition tiers - no blacklist, just rankings"""
    TOP_10 = "top_10"           # Elite - gets sent to Telegram
    PENDING = "pending"         # Competing to enter TOP 10
    NEW = "new"                 # Just added, building history


@dataclass
class CompetitorMetrics:
    """Performance metrics for a competing asset"""
    symbol: str
    asset_type: str  # "stock" or "crypto"
    tier: CompetitionTier
    
    # Performance (rolling 30-day)
    total_predictions: int
    wins: int
    losses: int
    win_rate: float  # 0.0 to 1.0
    
    # Trend analysis
    recent_wins: int      # Last 7 days
    recent_total: int     # Last 7 days
    recent_win_rate: float
    trend: str            # "hot", "stable", "cooling"
    
    # Rankings
    rank: int             # 1-N within asset type
    rank_change: int      # +/- since last update
    
    # Timestamps
    first_prediction: datetime
    last_prediction: datetime
    last_updated: datetime
    
    # Competition status
    days_in_top_10: int
    times_promoted: int
    times_demoted: int


class V3CompetitionSystem:
    """
    Fair Competition System - No Blacklist!
    
    Rules:
    1. ALL assets can compete (stocks and crypto)
    2. Need minimum 10 predictions to be ranked
    3. TOP 10 = highest win rate in each category
    4. Daily ranking updates - best performers rise, poor performers fall
    5. "Hot streak" bonus: Recent performance weighted 1.5x
    """
    
    def __init__(self):
        self.MIN_PREDICTIONS = int(os.getenv("V3_MIN_PREDICTIONS", "10"))
        self.TOP_N = int(os.getenv("V3_TOP_N", "10"))
        self.HOT_STREAK_BONUS = float(os.getenv("V3_HOT_BONUS", "1.5"))
        self.LOOKBACK_DAYS = int(os.getenv("V3_LOOKBACK_DAYS", "30"))
        self.RECENT_DAYS = int(os.getenv("V3_RECENT_DAYS", "7"))
        
        # Competition pools
        self._stock_pool: Dict[str, CompetitorMetrics] = {}
        self._crypto_pool: Dict[str, CompetitorMetrics] = {}
        
        # TOP 10 lists (derived from pools)
        self._top_stocks: List[str] = []
        self._top_crypto: List[str] = []
        
        # PostgreSQL support
        self.use_postgres = bool(DATABASE_URL)
        self._ensure_tables()
        self._load_pools()
        
        LOGGER.info(f"[V3-COMPETITION] 🏆 Initialized: {len(self._stock_pool)} stocks, {len(self._crypto_pool)} crypto competing")
    
    def _get_connection(self):
        """Get PostgreSQL connection"""
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    
    def _ensure_tables(self):
        """Create competition tables"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Main competition pool table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS v3_competition_pool (
                    symbol TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    tier TEXT DEFAULT 'new',
                    total_predictions INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    recent_wins INTEGER DEFAULT 0,
                    recent_total INTEGER DEFAULT 0,
                    recent_win_rate REAL DEFAULT 0.0,
                    trend TEXT DEFAULT 'stable',
                    rank INTEGER DEFAULT 999,
                    rank_change INTEGER DEFAULT 0,
                    first_prediction TIMESTAMP WITH TIME ZONE,
                    last_prediction TIMESTAMP WITH TIME ZONE,
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    days_in_top_10 INTEGER DEFAULT 0,
                    times_promoted INTEGER DEFAULT 0,
                    times_demoted INTEGER DEFAULT 0
                )
            """)
            
            # Shadow predictions table (for tracking without alerting)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS v3_shadow_predictions (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    target_time TIMESTAMP WITH TIME ZONE,
                    outcome TEXT,
                    resolved_at TIMESTAMP WITH TIME ZONE,
                    final_price REAL
                )
            """)
            
            # Index for faster queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_v3_shadow_symbol 
                ON v3_shadow_predictions(symbol)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_v3_shadow_outcome 
                ON v3_shadow_predictions(outcome)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_v3_pool_type_rank 
                ON v3_competition_pool(asset_type, rank)
            """)
            
            conn.commit()
            conn.close()
            LOGGER.info("[V3-COMPETITION] ✅ PostgreSQL tables ready")
        except Exception as e:
            LOGGER.error(f"[V3-COMPETITION] Failed to create tables: {e}")
    
    def _load_pools(self):
        """Load competition pools from database"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT symbol, asset_type, tier, total_predictions, wins, losses,
                       win_rate, recent_wins, recent_total, recent_win_rate, trend,
                       rank, rank_change, first_prediction, last_prediction,
                       last_updated, days_in_top_10, times_promoted, times_demoted
                FROM v3_competition_pool
                ORDER BY asset_type, rank
            """)
            
            for row in cur.fetchall():
                metrics = CompetitorMetrics(
                    symbol=row[0],
                    asset_type=row[1],
                    tier=CompetitionTier(row[2]) if row[2] else CompetitionTier.NEW,
                    total_predictions=row[3] or 0,
                    wins=row[4] or 0,
                    losses=row[5] or 0,
                    win_rate=row[6] or 0.0,
                    recent_wins=row[7] or 0,
                    recent_total=row[8] or 0,
                    recent_win_rate=row[9] or 0.0,
                    trend=row[10] or "stable",
                    rank=row[11] or 999,
                    rank_change=row[12] or 0,
                    first_prediction=row[13] or datetime.utcnow(),
                    last_prediction=row[14] or datetime.utcnow(),
                    last_updated=row[15] or datetime.utcnow(),
                    days_in_top_10=row[16] or 0,
                    times_promoted=row[17] or 0,
                    times_demoted=row[18] or 0
                )
                
                if metrics.asset_type == "stock":
                    self._stock_pool[metrics.symbol] = metrics
                else:
                    self._crypto_pool[metrics.symbol] = metrics
            
            # Rebuild TOP 10 lists
            self._rebuild_top_lists()
            
            conn.close()
            LOGGER.info(f"[V3-COMPETITION] Loaded {len(self._stock_pool)} stocks, {len(self._crypto_pool)} crypto")
        except Exception as e:
            LOGGER.error(f"[V3-COMPETITION] Failed to load pools: {e}")
    
    def _rebuild_top_lists(self):
        """Rebuild TOP 10 lists from pool rankings"""
        # Get stocks with enough predictions, sorted by rank
        qualified_stocks = [
            s for s, m in self._stock_pool.items() 
            if m.total_predictions >= self.MIN_PREDICTIONS
        ]
        qualified_stocks.sort(key=lambda s: self._stock_pool[s].rank)
        self._top_stocks = qualified_stocks[:self.TOP_N]
        
        # Get crypto with enough predictions, sorted by rank
        qualified_crypto = [
            s for s, m in self._crypto_pool.items() 
            if m.total_predictions >= self.MIN_PREDICTIONS
        ]
        qualified_crypto.sort(key=lambda s: self._crypto_pool[s].rank)
        self._top_crypto = qualified_crypto[:self.TOP_N]
        
        LOGGER.debug(f"[V3-COMPETITION] TOP 10 stocks: {self._top_stocks}")
        LOGGER.debug(f"[V3-COMPETITION] TOP 10 crypto: {self._top_crypto}")
    
    def add_competitor(self, symbol: str, asset_type: str) -> CompetitorMetrics:
        """Add a new competitor to the pool"""
        pool = self._stock_pool if asset_type == "stock" else self._crypto_pool
        
        if symbol in pool:
            return pool[symbol]
        
        metrics = CompetitorMetrics(
            symbol=symbol,
            asset_type=asset_type,
            tier=CompetitionTier.NEW,
            total_predictions=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            recent_wins=0,
            recent_total=0,
            recent_win_rate=0.0,
            trend="stable",
            rank=999,
            rank_change=0,
            first_prediction=datetime.utcnow(),
            last_prediction=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            days_in_top_10=0,
            times_promoted=0,
            times_demoted=0
        )
        
        pool[symbol] = metrics
        self._save_competitor(metrics)
        
        LOGGER.info(f"[V3-COMPETITION] 🆕 New competitor: {symbol} ({asset_type})")
        return metrics
    
    def record_shadow_prediction(
        self,
        symbol: str,
        asset_type: str,
        direction: str,
        entry_price: float,
        target_price: float,
        confidence: float,
        target_time: datetime
    ) -> int:
        """
        Record a shadow prediction (for ranking purposes).
        These are NOT sent to Telegram - they're for competition tracking.
        
        Returns: prediction_id
        """
        # Ensure competitor exists
        self.add_competitor(symbol, asset_type)
        
        if not self.use_postgres:
            return -1
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO v3_shadow_predictions 
                (symbol, asset_type, direction, entry_price, target_price, 
                 confidence, target_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (symbol, asset_type, direction, entry_price, target_price,
                  confidence, target_time))
            
            pred_id = cur.fetchone()[0]
            conn.commit()
            conn.close()
            
            LOGGER.debug(f"[V3-COMPETITION] 📊 Shadow prediction #{pred_id}: {symbol} {direction}")
            return pred_id
        except Exception as e:
            LOGGER.error(f"[V3-COMPETITION] Failed to record shadow prediction: {e}")
            return -1
    
    def resolve_shadow_prediction(
        self,
        pred_id: int,
        outcome: str,  # "WIN" or "LOSS"
        final_price: float
    ):
        """Resolve a shadow prediction and update competitor metrics"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Update the prediction
            cur.execute("""
                UPDATE v3_shadow_predictions
                SET outcome = %s, final_price = %s, resolved_at = NOW()
                WHERE id = %s
                RETURNING symbol, asset_type
            """, (outcome, final_price, pred_id))
            
            row = cur.fetchone()
            if not row:
                conn.close()
                return
            
            symbol, asset_type = row
            
            # Update competitor metrics
            cur.execute("""
                UPDATE v3_competition_pool
                SET 
                    total_predictions = total_predictions + 1,
                    wins = wins + CASE WHEN %s = 'WIN' THEN 1 ELSE 0 END,
                    losses = losses + CASE WHEN %s = 'LOSS' THEN 1 ELSE 0 END,
                    last_prediction = NOW(),
                    last_updated = NOW()
                WHERE symbol = %s
            """, (outcome, outcome, symbol))
            
            conn.commit()
            conn.close()
            
            LOGGER.info(f"[V3-COMPETITION] ✅ Resolved #{pred_id}: {symbol} = {outcome}")
        except Exception as e:
            LOGGER.error(f"[V3-COMPETITION] Failed to resolve prediction: {e}")
    
    def update_rankings(self) -> Dict:
        """
        Recalculate all rankings based on performance.
        This is the MAIN competition logic - run daily!
        
        Returns dict with promotion/demotion changes.
        """
        if not self.use_postgres:
            return {"error": "No database"}
        
        LOGGER.info("[V3-COMPETITION] 🏆 Updating rankings...")
        
        changes = {
            "stocks": {"promoted": [], "demoted": []},
            "crypto": {"promoted": [], "demoted": []},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Calculate metrics from shadow predictions (last 30 days)
            cutoff = datetime.utcnow() - timedelta(days=self.LOOKBACK_DAYS)
            recent_cutoff = datetime.utcnow() - timedelta(days=self.RECENT_DAYS)
            
            for asset_type in ["stock", "crypto"]:
                # Get all-time stats
                cur.execute("""
                    SELECT symbol,
                           COUNT(*) as total,
                           SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
                    FROM v3_shadow_predictions
                    WHERE asset_type = %s 
                      AND outcome IS NOT NULL
                      AND created_at >= %s
                    GROUP BY symbol
                """, (asset_type, cutoff))
                
                all_stats = {row[0]: {"total": row[1], "wins": row[2]} 
                            for row in cur.fetchall()}
                
                # Get recent stats (last 7 days)
                cur.execute("""
                    SELECT symbol,
                           COUNT(*) as total,
                           SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
                    FROM v3_shadow_predictions
                    WHERE asset_type = %s 
                      AND outcome IS NOT NULL
                      AND created_at >= %s
                    GROUP BY symbol
                """, (asset_type, recent_cutoff))
                
                recent_stats = {row[0]: {"total": row[1], "wins": row[2]} 
                               for row in cur.fetchall()}
                
                # Calculate composite score with hot streak bonus
                scores = []
                for symbol in all_stats:
                    stats = all_stats[symbol]
                    total = stats["total"]
                    wins = stats["wins"]
                    
                    if total < self.MIN_PREDICTIONS:
                        continue
                    
                    win_rate = wins / total if total > 0 else 0
                    
                    # Recent performance
                    recent = recent_stats.get(symbol, {"total": 0, "wins": 0})
                    recent_total = recent["total"]
                    recent_wins = recent["wins"]
                    recent_wr = recent_wins / recent_total if recent_total > 0 else win_rate
                    
                    # Determine trend
                    if recent_total >= 3:
                        if recent_wr > win_rate + 0.1:
                            trend = "hot"
                        elif recent_wr < win_rate - 0.1:
                            trend = "cooling"
                        else:
                            trend = "stable"
                    else:
                        trend = "stable"
                    
                    # Composite score: base win rate + hot streak bonus
                    # Hot assets get 1.5x weight on recent performance
                    if trend == "hot":
                        score = (win_rate * 0.5) + (recent_wr * 0.5 * self.HOT_STREAK_BONUS)
                    else:
                        score = (win_rate * 0.7) + (recent_wr * 0.3)
                    
                    scores.append({
                        "symbol": symbol,
                        "score": score,
                        "win_rate": win_rate,
                        "total": total,
                        "wins": wins,
                        "recent_total": recent_total,
                        "recent_wins": recent_wins,
                        "recent_wr": recent_wr,
                        "trend": trend
                    })
                
                # Sort by score (highest first)
                scores.sort(key=lambda x: x["score"], reverse=True)
                
                # Assign ranks and update database
                pool = self._stock_pool if asset_type == "stock" else self._crypto_pool
                old_top = self._top_stocks[:] if asset_type == "stock" else self._top_crypto[:]
                
                for rank, entry in enumerate(scores, start=1):
                    symbol = entry["symbol"]
                    old_rank = pool[symbol].rank if symbol in pool else 999
                    rank_change = old_rank - rank  # Positive = moved up
                    
                    # Determine tier
                    if rank <= self.TOP_N:
                        new_tier = CompetitionTier.TOP_10
                    else:
                        new_tier = CompetitionTier.PENDING
                    
                    # Track promotions/demotions
                    if symbol in pool:
                        old_tier = pool[symbol].tier
                        if old_tier != CompetitionTier.TOP_10 and new_tier == CompetitionTier.TOP_10:
                            changes[asset_type]["promoted"].append({
                                "symbol": symbol,
                                "new_rank": rank,
                                "win_rate": f"{entry['win_rate']*100:.1f}%",
                                "trend": entry["trend"]
                            })
                        elif old_tier == CompetitionTier.TOP_10 and new_tier != CompetitionTier.TOP_10:
                            changes[asset_type]["demoted"].append({
                                "symbol": symbol,
                                "old_rank": old_rank,
                                "new_rank": rank,
                                "win_rate": f"{entry['win_rate']*100:.1f}%"
                            })
                    
                    # Update database
                    cur.execute("""
                        UPDATE v3_competition_pool
                        SET 
                            tier = %s,
                            total_predictions = %s,
                            wins = %s,
                            losses = %s,
                            win_rate = %s,
                            recent_wins = %s,
                            recent_total = %s,
                            recent_win_rate = %s,
                            trend = %s,
                            rank = %s,
                            rank_change = %s,
                            days_in_top_10 = days_in_top_10 + CASE WHEN %s = 'top_10' THEN 1 ELSE 0 END,
                            times_promoted = times_promoted + CASE WHEN %s THEN 1 ELSE 0 END,
                            times_demoted = times_demoted + CASE WHEN %s THEN 1 ELSE 0 END,
                            last_updated = NOW()
                        WHERE symbol = %s
                    """, (
                        new_tier.value,
                        entry["total"],
                        entry["wins"],
                        entry["total"] - entry["wins"],
                        entry["win_rate"],
                        entry["recent_wins"],
                        entry["recent_total"],
                        entry["recent_wr"],
                        entry["trend"],
                        rank,
                        rank_change,
                        new_tier.value,
                        symbol in changes[asset_type]["promoted"],
                        symbol in [d["symbol"] for d in changes[asset_type]["demoted"]],
                        symbol
                    ))
                    
                    # Update in-memory pool
                    if symbol in pool:
                        pool[symbol].tier = new_tier
                        pool[symbol].rank = rank
                        pool[symbol].rank_change = rank_change
                        pool[symbol].win_rate = entry["win_rate"]
                        pool[symbol].trend = entry["trend"]
            
            conn.commit()
            conn.close()
            
            # Rebuild TOP 10 lists
            self._rebuild_top_lists()
            
            LOGGER.info(f"[V3-COMPETITION] 🏆 Rankings updated!")
            LOGGER.info(f"  Stocks: {len(changes['stocks']['promoted'])} promoted, {len(changes['stocks']['demoted'])} demoted")
            LOGGER.info(f"  Crypto: {len(changes['crypto']['promoted'])} promoted, {len(changes['crypto']['demoted'])} demoted")
            
            return changes
        except Exception as e:
            LOGGER.error(f"[V3-COMPETITION] Failed to update rankings: {e}")
            return {"error": str(e)}
    
    def _save_competitor(self, metrics: CompetitorMetrics):
        """Save competitor to database"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO v3_competition_pool 
                (symbol, asset_type, tier, total_predictions, wins, losses, 
                 win_rate, rank, first_prediction, last_prediction, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    tier = EXCLUDED.tier,
                    total_predictions = EXCLUDED.total_predictions,
                    wins = EXCLUDED.wins,
                    losses = EXCLUDED.losses,
                    win_rate = EXCLUDED.win_rate,
                    last_updated = NOW()
            """, (
                metrics.symbol,
                metrics.asset_type,
                metrics.tier.value,
                metrics.total_predictions,
                metrics.wins,
                metrics.losses,
                metrics.win_rate,
                metrics.rank,
                metrics.first_prediction,
                metrics.last_prediction,
                metrics.last_updated
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"[V3-COMPETITION] Failed to save competitor: {e}")
    
    def get_top_stocks(self) -> List[str]:
        """Get current TOP 10 stocks"""
        return self._top_stocks[:]
    
    def get_top_crypto(self) -> List[str]:
        """Get current TOP 10 crypto"""
        return self._top_crypto[:]
    
    def get_leaderboard(self, asset_type: str, limit: int = 20) -> List[Dict]:
        """Get leaderboard for stocks or crypto"""
        pool = self._stock_pool if asset_type == "stock" else self._crypto_pool
        
        qualified = [
            m for m in pool.values() 
            if m.total_predictions >= self.MIN_PREDICTIONS
        ]
        qualified.sort(key=lambda m: m.rank)
        
        return [
            {
                "rank": m.rank,
                "symbol": m.symbol,
                "tier": m.tier.value,
                "win_rate": f"{m.win_rate*100:.1f}%",
                "total_predictions": m.total_predictions,
                "wins": m.wins,
                "losses": m.losses,
                "trend": m.trend,
                "rank_change": m.rank_change,
                "days_in_top_10": m.days_in_top_10
            }
            for m in qualified[:limit]
        ]
    
    def get_pending_contenders(self, asset_type: str, limit: int = 10) -> List[Dict]:
        """Get pending assets closest to breaking into TOP 10"""
        pool = self._stock_pool if asset_type == "stock" else self._crypto_pool
        
        # Get pending assets sorted by rank (closest to TOP 10)
        pending = [
            m for m in pool.values() 
            if m.tier == CompetitionTier.PENDING and m.total_predictions >= self.MIN_PREDICTIONS
        ]
        pending.sort(key=lambda m: m.rank)
        
        return [
            {
                "rank": m.rank,
                "symbol": m.symbol,
                "win_rate": f"{m.win_rate*100:.1f}%",
                "total_predictions": m.total_predictions,
                "trend": m.trend,
                "gap_to_top_10": m.rank - self.TOP_N
            }
            for m in pending[:limit]
        ]
    
    def get_competition_status(self) -> Dict:
        """Get full competition status"""
        return {
            "top_10_stocks": self._top_stocks,
            "top_10_crypto": self._top_crypto,
            "total_stock_competitors": len(self._stock_pool),
            "total_crypto_competitors": len(self._crypto_pool),
            "min_predictions_required": self.MIN_PREDICTIONS,
            "lookback_days": self.LOOKBACK_DAYS,
            "hot_streak_bonus": f"{self.HOT_STREAK_BONUS}x",
            "stock_leaderboard": self.get_leaderboard("stock", 15),
            "crypto_leaderboard": self.get_leaderboard("crypto", 15),
            "stock_contenders": self.get_pending_contenders("stock", 5),
            "crypto_contenders": self.get_pending_contenders("crypto", 5)
        }
    
    def should_alert(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if this symbol should be included in TOP 20 alerts.
        
        Returns: (should_alert: bool, reason: str)
        """
        # Check stocks
        if symbol in self._stock_pool:
            if symbol in self._top_stocks:
                rank = self._top_stocks.index(symbol) + 1
                return True, f"TOP 10 stock (rank #{rank})"
            else:
                return False, f"Stock pending (rank #{self._stock_pool[symbol].rank})"
        
        # Check crypto
        if symbol in self._crypto_pool:
            if symbol in self._top_crypto:
                rank = self._top_crypto.index(symbol) + 1
                return True, f"TOP 10 crypto (rank #{rank})"
            else:
                return False, f"Crypto pending (rank #{self._crypto_pool[symbol].rank})"
        
        # Unknown asset - add to pool as NEW
        return False, "New competitor (building history)"
    
    def seed_initial_pool(self, stocks: List[str], crypto: List[str]):
        """Seed initial competitors (for bootstrapping)"""
        LOGGER.info(f"[V3-COMPETITION] 🌱 Seeding {len(stocks)} stocks, {len(crypto)} crypto")
        
        for symbol in stocks:
            self.add_competitor(symbol, "stock")
        
        for symbol in crypto:
            self.add_competitor(symbol, "crypto")
        
        LOGGER.info("[V3-COMPETITION] ✅ Initial pool seeded")


# Singleton instance
_competition_system: Optional[V3CompetitionSystem] = None


def get_competition_system() -> V3CompetitionSystem:
    """Get or create the V3 Competition System singleton"""
    global _competition_system
    if _competition_system is None:
        _competition_system = V3CompetitionSystem()
    return _competition_system
