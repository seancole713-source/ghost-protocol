#!/usr/bin/env python3
"""
🎮 GHOST PROTOCOL V3 - MONEY GAME ENGINE

Think like a VIDEO GAME:
- SCORE = MONEY EARNED
- GOAL = Find the next BULLISH MONEY MAKER
- #1 = Best profit maker, #10 = Still good but less profit
- LOSING MONEY = BAD (heavy penalty)

This isn't about win rate - it's about PROFIT POTENTIAL.

A stock that goes up 10% beats a stock that goes up 1%
even if both are "wins".

MONEY IS THE SCORE. GHOST WANTS TO WIN.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

LOGGER = logging.getLogger("ghost.money_game")

DATABASE_URL = os.getenv("DATABASE_URL")


class PlayerTier(Enum):
    """Player tiers in the money game"""
    ELITE = "elite"           # TOP 10 - The money makers
    RISING_STAR = "rising"    # Close to TOP 10, showing profit
    BENCHED = "benched"       # Not performing, needs to prove itself
    ROOKIE = "rookie"         # New, building track record


@dataclass
class PlayerStats:
    """Stats for a player (asset) in the money game"""
    symbol: str
    asset_type: str  # "stock" or "crypto"
    tier: PlayerTier
    
    # MONEY STATS - This is what matters!
    total_profit_pct: float      # Total % gained (e.g., +45.2%)
    avg_profit_per_trade: float  # Average % per prediction
    best_trade_pct: float        # Best single trade
    worst_trade_pct: float       # Worst single trade (negative = loss)
    
    # Win/Loss (secondary)
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    
    # MONEY SCORE - The MAIN ranking metric
    money_score: float  # Higher = better money maker
    
    # Trend
    recent_profit_pct: float  # Last 7 days profit
    momentum: str             # "hot", "stable", "cold"
    
    # Rankings
    rank: int
    rank_change: int
    
    # Timestamps
    last_trade: datetime
    last_updated: datetime


class MoneyGameEngine:
    """
    🎮 THE MONEY GAME
    
    RULES:
    1. SCORE = Total profit potential
    2. #1 = Best money maker
    3. Losses HURT your score (2x penalty)
    4. Big wins = BONUS points
    5. Consistency matters (steady profits beat volatile)
    
    GOAL: Find the NEXT BIG DEAL - the bullish money maker!
    """
    
    def __init__(self):
        # Game settings
        self.TOP_N = int(os.getenv("MONEY_GAME_TOP_N", "10"))
        self.MIN_TRADES = int(os.getenv("MONEY_GAME_MIN_TRADES", "3"))  # 3 trades to qualify
        self.LOSS_PENALTY = float(os.getenv("MONEY_GAME_LOSS_PENALTY", "2.0"))  # Losses hurt 2x
        self.BIG_WIN_BONUS = float(os.getenv("MONEY_GAME_BIG_WIN_BONUS", "1.5"))  # +5% gets bonus
        self.BIG_WIN_THRESHOLD = float(os.getenv("MONEY_GAME_BIG_WIN_PCT", "5.0"))  # 5%+
        self.CONSISTENCY_BONUS = float(os.getenv("MONEY_GAME_CONSISTENCY", "1.2"))  # Low variance
        
        # Player pools
        self._stock_players: Dict[str, PlayerStats] = {}
        self._crypto_players: Dict[str, PlayerStats] = {}
        
        # TOP 10 lists
        self._elite_stocks: List[str] = []
        self._elite_crypto: List[str] = []
        
        # PostgreSQL
        self.use_postgres = bool(DATABASE_URL)
        self._ensure_tables()
        self._load_players()
        
        LOGGER.info(f"🎮 [MONEY GAME] Initialized: {len(self._stock_players)} stocks, {len(self._crypto_players)} crypto competing for the bag!")
    
    def _get_connection(self):
        """Get PostgreSQL connection via shared pool bridge."""
        from core.db_pool import get_sync_connection_raw
        return get_sync_connection_raw()
    
    def _ensure_tables(self):
        """Create money game tables"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Player stats table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS money_game_players (
                    symbol TEXT PRIMARY KEY,
                    asset_type TEXT NOT NULL,
                    tier TEXT DEFAULT 'rookie',
                    total_profit_pct REAL DEFAULT 0.0,
                    avg_profit_per_trade REAL DEFAULT 0.0,
                    best_trade_pct REAL DEFAULT 0.0,
                    worst_trade_pct REAL DEFAULT 0.0,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    money_score REAL DEFAULT 0.0,
                    recent_profit_pct REAL DEFAULT 0.0,
                    momentum TEXT DEFAULT 'stable',
                    rank INTEGER DEFAULT 999,
                    rank_change INTEGER DEFAULT 0,
                    last_trade TIMESTAMP WITH TIME ZONE,
                    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Trade history for money tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS money_game_trades (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    final_price REAL,
                    profit_pct REAL,
                    confidence REAL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    resolved_at TIMESTAMP WITH TIME ZONE,
                    is_win BOOLEAN
                )
            """)
            
            # Index for fast lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_money_trades_symbol 
                ON money_game_trades(symbol)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_money_players_score 
                ON money_game_players(asset_type, money_score DESC)
            """)
            
            conn.commit()
            conn.close()
            LOGGER.info("🎮 [MONEY GAME] Database tables ready!")
        except Exception as e:
            LOGGER.error(f"🎮 [MONEY GAME] DB setup error: {e}")
    
    def _load_players(self):
        """Load all players from database"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT symbol, asset_type, tier, total_profit_pct, avg_profit_per_trade,
                       best_trade_pct, worst_trade_pct, total_trades, wins, losses,
                       win_rate, money_score, recent_profit_pct, momentum, rank,
                       rank_change, last_trade, last_updated
                FROM money_game_players
                ORDER BY asset_type, money_score DESC
            """)
            
            for row in cur.fetchall():
                stats = PlayerStats(
                    symbol=row[0],
                    asset_type=row[1],
                    tier=PlayerTier(row[2]) if row[2] else PlayerTier.ROOKIE,
                    total_profit_pct=row[3] or 0.0,
                    avg_profit_per_trade=row[4] or 0.0,
                    best_trade_pct=row[5] or 0.0,
                    worst_trade_pct=row[6] or 0.0,
                    total_trades=row[7] or 0,
                    wins=row[8] or 0,
                    losses=row[9] or 0,
                    win_rate=row[10] or 0.0,
                    money_score=row[11] or 0.0,
                    recent_profit_pct=row[12] or 0.0,
                    momentum=row[13] or "stable",
                    rank=row[14] or 999,
                    rank_change=row[15] or 0,
                    last_trade=row[16] or datetime.utcnow(),
                    last_updated=row[17] or datetime.utcnow()
                )
                
                if stats.asset_type == "stock":
                    self._stock_players[stats.symbol] = stats
                else:
                    self._crypto_players[stats.symbol] = stats
            
            self._rebuild_elite_lists()
            conn.close()
            
            LOGGER.info(f"🎮 [MONEY GAME] Loaded {len(self._stock_players)} stock players, {len(self._crypto_players)} crypto players")
        except Exception as e:
            LOGGER.error(f"🎮 [MONEY GAME] Load error: {e}")
    
    def _rebuild_elite_lists(self):
        """Rebuild TOP 10 (elite) lists based on money score"""
        # Stocks - sorted by money_score
        qualified_stocks = [
            s for s, p in self._stock_players.items()
            if p.total_trades >= self.MIN_TRADES
        ]
        qualified_stocks.sort(
            key=lambda s: self._stock_players[s].money_score, 
            reverse=True
        )
        self._elite_stocks = qualified_stocks[:self.TOP_N]
        
        # Crypto - sorted by money_score
        qualified_crypto = [
            s for s, p in self._crypto_players.items()
            if p.total_trades >= self.MIN_TRADES
        ]
        qualified_crypto.sort(
            key=lambda s: self._crypto_players[s].money_score, 
            reverse=True
        )
        self._elite_crypto = qualified_crypto[:self.TOP_N]
    
    def add_player(self, symbol: str, asset_type: str) -> PlayerStats:
        """Add a new player (rookie) to the game"""
        pool = self._stock_players if asset_type == "stock" else self._crypto_players
        
        if symbol in pool:
            return pool[symbol]
        
        stats = PlayerStats(
            symbol=symbol,
            asset_type=asset_type,
            tier=PlayerTier.ROOKIE,
            total_profit_pct=0.0,
            avg_profit_per_trade=0.0,
            best_trade_pct=0.0,
            worst_trade_pct=0.0,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            money_score=0.0,
            recent_profit_pct=0.0,
            momentum="stable",
            rank=999,
            rank_change=0,
            last_trade=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        pool[symbol] = stats
        self._save_player(stats)
        
        LOGGER.info(f"🎮 [MONEY GAME] 🆕 New rookie: {symbol} ({asset_type}) - let's see what you got!")
        return stats
    
    def record_trade(
        self,
        symbol: str,
        asset_type: str,
        direction: str,
        entry_price: float,
        target_price: float,
        confidence: float
    ) -> int:
        """
        Record a new trade (prediction) for a player.
        
        Returns: trade_id
        """
        self.add_player(symbol, asset_type)
        
        if not self.use_postgres:
            return -1
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO money_game_trades
                (symbol, asset_type, direction, entry_price, target_price, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (symbol, asset_type, direction, entry_price, target_price, confidence))
            
            trade_id = cur.fetchone()[0]
            conn.commit()
            conn.close()
            
            LOGGER.debug(f"🎮 [MONEY GAME] Trade #{trade_id}: {symbol} {direction} @ {entry_price}")
            return trade_id
        except Exception as e:
            LOGGER.error(f"🎮 [MONEY GAME] Record trade error: {e}")
            return -1
    
    def resolve_trade(self, trade_id: int, final_price: float) -> Dict:
        """
        Resolve a trade and calculate PROFIT/LOSS.
        
        This is where the MONEY gets counted!
        """
        if not self.use_postgres:
            return {"error": "No database"}
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Get trade details
            cur.execute("""
                SELECT symbol, asset_type, direction, entry_price, target_price
                FROM money_game_trades
                WHERE id = %s AND resolved_at IS NULL
            """, (trade_id,))
            
            row = cur.fetchone()
            if not row:
                conn.close()
                return {"error": "Trade not found or already resolved"}
            
            symbol, asset_type, direction, entry_price, target_price = row
            
            # CALCULATE PROFIT/LOSS
            if direction == "BUY":
                profit_pct = ((final_price - entry_price) / entry_price) * 100
                is_win = final_price >= target_price
            else:  # SELL
                profit_pct = ((entry_price - final_price) / entry_price) * 100
                is_win = final_price <= target_price
            
            # Update trade record
            cur.execute("""
                UPDATE money_game_trades
                SET final_price = %s, profit_pct = %s, is_win = %s, resolved_at = NOW()
                WHERE id = %s
            """, (final_price, profit_pct, is_win, trade_id))
            
            # Update player stats
            self._update_player_stats(cur, symbol, asset_type, profit_pct, is_win)
            
            conn.commit()
            conn.close()
            
            emoji = "💰" if profit_pct > 0 else "💸"
            LOGGER.info(f"🎮 [MONEY GAME] {emoji} {symbol}: {profit_pct:+.2f}% {'WIN' if is_win else 'LOSS'}")
            
            return {
                "symbol": symbol,
                "profit_pct": profit_pct,
                "is_win": is_win,
                "final_price": final_price
            }
        except Exception as e:
            LOGGER.error(f"🎮 [MONEY GAME] Resolve error: {e}")
            return {"error": str(e)}
    
    def _update_player_stats(self, cur, symbol: str, asset_type: str, profit_pct: float, is_win: bool):
        """Update a player's stats after a trade"""
        
        # Get current stats from database
        cur.execute("""
            SELECT total_profit_pct, total_trades, wins, losses, best_trade_pct, worst_trade_pct
            FROM money_game_players
            WHERE symbol = %s
        """, (symbol,))
        
        row = cur.fetchone()
        if not row:
            return
        
        total_profit, total_trades, wins, losses, best_trade, worst_trade = row
        
        # Update stats
        new_total_profit = (total_profit or 0) + profit_pct
        new_total_trades = (total_trades or 0) + 1
        new_wins = (wins or 0) + (1 if is_win else 0)
        new_losses = (losses or 0) + (0 if is_win else 1)
        new_best = max(best_trade or 0, profit_pct)
        new_worst = min(worst_trade or 0, profit_pct)
        new_avg = new_total_profit / new_total_trades if new_total_trades > 0 else 0
        new_win_rate = new_wins / new_total_trades if new_total_trades > 0 else 0
        
        # CALCULATE MONEY SCORE - This is the GAME SCORE!
        money_score = self._calculate_money_score(
            new_total_profit, new_avg, new_best, new_worst, 
            new_total_trades, new_win_rate
        )
        
        # Determine tier
        tier = self._determine_tier(money_score, new_total_trades)
        
        # Update database
        cur.execute("""
            UPDATE money_game_players
            SET total_profit_pct = %s,
                avg_profit_per_trade = %s,
                best_trade_pct = %s,
                worst_trade_pct = %s,
                total_trades = %s,
                wins = %s,
                losses = %s,
                win_rate = %s,
                money_score = %s,
                tier = %s,
                last_trade = NOW(),
                last_updated = NOW()
            WHERE symbol = %s
        """, (
            new_total_profit, new_avg, new_best, new_worst,
            new_total_trades, new_wins, new_losses, new_win_rate,
            money_score, tier.value, symbol
        ))
        
        # Update in-memory cache
        pool = self._stock_players if asset_type == "stock" else self._crypto_players
        if symbol in pool:
            pool[symbol].total_profit_pct = new_total_profit
            pool[symbol].avg_profit_per_trade = new_avg
            pool[symbol].total_trades = new_total_trades
            pool[symbol].wins = new_wins
            pool[symbol].losses = new_losses
            pool[symbol].win_rate = new_win_rate
            pool[symbol].money_score = money_score
            pool[symbol].tier = tier
    
    def _calculate_money_score(
        self, 
        total_profit: float, 
        avg_profit: float, 
        best_trade: float,
        worst_trade: float,
        total_trades: int,
        win_rate: float
    ) -> float:
        """
        🎮 THE MONEY SCORE FORMULA
        
        This is how Ghost ranks money makers!
        
        Components:
        1. PROFIT BASE: Total profit (most important)
        2. CONSISTENCY: Average profit per trade
        3. BIG WINS: Bonus for 5%+ trades
        4. LOSS PENALTY: Losses hurt 2x
        5. WIN RATE: Secondary factor
        
        Higher score = Better money maker = Higher rank!
        """
        if total_trades < self.MIN_TRADES:
            return 0.0
        
        # BASE: Total profit matters most
        base_score = total_profit
        
        # CONSISTENCY BONUS: Steady profits are good
        if avg_profit > 1.0:  # Averaging 1%+ per trade
            base_score *= self.CONSISTENCY_BONUS
        
        # BIG WIN BONUS: 5%+ trades get extra credit
        if best_trade >= self.BIG_WIN_THRESHOLD:
            big_win_multiplier = 1 + ((best_trade - self.BIG_WIN_THRESHOLD) / 10)
            base_score *= min(big_win_multiplier, self.BIG_WIN_BONUS)
        
        # LOSS PENALTY: Big losses hurt your score
        if worst_trade < -5.0:  # Lost more than 5% on worst trade
            loss_penalty = abs(worst_trade) / 10
            base_score -= (loss_penalty * self.LOSS_PENALTY)
        
        # WIN RATE FACTOR: Higher win rate = more reliable
        win_rate_factor = 0.5 + (win_rate * 0.5)  # 0.5 to 1.0
        base_score *= win_rate_factor
        
        # VOLUME FACTOR: More trades = more data = more confident
        volume_factor = min(total_trades / 20, 1.5)  # Caps at 1.5x for 30+ trades
        base_score *= volume_factor
        
        return round(base_score, 2)
    
    def _determine_tier(self, money_score: float, total_trades: int) -> PlayerTier:
        """Determine player tier based on score"""
        if total_trades < self.MIN_TRADES:
            return PlayerTier.ROOKIE
        
        if money_score >= 20:  # High scorers
            return PlayerTier.ELITE
        elif money_score >= 5:  # Showing promise
            return PlayerTier.RISING_STAR
        else:
            return PlayerTier.BENCHED
    
    def update_rankings(self) -> Dict:
        """
        🏆 RECALCULATE ALL RANKINGS
        
        Who's the #1 money maker? Let's find out!
        """
        if not self.use_postgres:
            return {"error": "No database"}
        
        LOGGER.info("🎮 [MONEY GAME] 🏆 Calculating rankings...")
        
        changes = {
            "stocks": {"promoted": [], "demoted": [], "new_elite": []},
            "crypto": {"promoted": [], "demoted": [], "new_elite": []},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Recalculate recent profit for all players (last 7 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            
            for asset_type in ["stock", "crypto"]:
                # Get recent profit per symbol
                cur.execute("""
                    SELECT symbol, COALESCE(SUM(profit_pct), 0) as recent_profit
                    FROM money_game_trades
                    WHERE asset_type = %s 
                      AND resolved_at >= %s
                      AND profit_pct IS NOT NULL
                    GROUP BY symbol
                """, (asset_type, recent_cutoff))
                
                recent_profits = {row[0]: row[1] for row in cur.fetchall()}
                
                # Get all players with enough trades
                cur.execute("""
                    SELECT symbol, money_score, rank
                    FROM money_game_players
                    WHERE asset_type = %s AND total_trades >= %s
                    ORDER BY money_score DESC
                """, (asset_type, self.MIN_TRADES))
                
                players = cur.fetchall()
                pool = self._stock_players if asset_type == "stock" else self._crypto_players
                old_elite = self._elite_stocks[:] if asset_type == "stock" else self._elite_crypto[:]
                
                # Assign new ranks
                for new_rank, (symbol, money_score, old_rank) in enumerate(players, start=1):
                    old_rank = old_rank or 999
                    rank_change = old_rank - new_rank
                    
                    # Determine momentum
                    recent_profit = recent_profits.get(symbol, 0)
                    if recent_profit > 5:
                        momentum = "hot"
                    elif recent_profit < -5:
                        momentum = "cold"
                    else:
                        momentum = "stable"
                    
                    # Determine tier
                    if new_rank <= self.TOP_N:
                        tier = PlayerTier.ELITE
                    elif money_score >= 5:
                        tier = PlayerTier.RISING_STAR
                    else:
                        tier = PlayerTier.BENCHED
                    
                    # Update database
                    cur.execute("""
                        UPDATE money_game_players
                        SET rank = %s, rank_change = %s, recent_profit_pct = %s,
                            momentum = %s, tier = %s, last_updated = NOW()
                        WHERE symbol = %s
                    """, (new_rank, rank_change, recent_profit, momentum, tier.value, symbol))
                    
                    # Track promotions/demotions
                    if new_rank <= self.TOP_N and symbol not in old_elite:
                        changes[asset_type]["promoted"].append({
                            "symbol": symbol,
                            "new_rank": new_rank,
                            "money_score": money_score,
                            "momentum": momentum
                        })
                    elif new_rank > self.TOP_N and symbol in old_elite:
                        changes[asset_type]["demoted"].append({
                            "symbol": symbol,
                            "old_rank": old_rank,
                            "new_rank": new_rank,
                            "money_score": money_score
                        })
                    
                    # Update in-memory
                    if symbol in pool:
                        pool[symbol].rank = new_rank
                        pool[symbol].rank_change = rank_change
                        pool[symbol].recent_profit_pct = recent_profit
                        pool[symbol].momentum = momentum
                        pool[symbol].tier = tier
                
                # New elite list
                new_elite = [p[0] for p in players[:self.TOP_N]]
                changes[asset_type]["new_elite"] = new_elite
                
                if asset_type == "stock":
                    self._elite_stocks = new_elite
                else:
                    self._elite_crypto = new_elite
            
            conn.commit()
            conn.close()
            
            LOGGER.info(f"🎮 [MONEY GAME] 🏆 Rankings updated!")
            LOGGER.info(f"  Stocks: {len(changes['stocks']['promoted'])} promoted, {len(changes['stocks']['demoted'])} demoted")
            LOGGER.info(f"  Crypto: {len(changes['crypto']['promoted'])} promoted, {len(changes['crypto']['demoted'])} demoted")
            
            return changes
        except Exception as e:
            LOGGER.error(f"🎮 [MONEY GAME] Ranking error: {e}")
            return {"error": str(e)}
    
    def _save_player(self, stats: PlayerStats):
        """Save player to database"""
        if not self.use_postgres:
            return
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO money_game_players
                (symbol, asset_type, tier, total_profit_pct, avg_profit_per_trade,
                 best_trade_pct, worst_trade_pct, total_trades, wins, losses,
                 win_rate, money_score, rank)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    tier = EXCLUDED.tier,
                    total_profit_pct = EXCLUDED.total_profit_pct,
                    avg_profit_per_trade = EXCLUDED.avg_profit_per_trade,
                    last_updated = NOW()
            """, (
                stats.symbol, stats.asset_type, stats.tier.value,
                stats.total_profit_pct, stats.avg_profit_per_trade,
                stats.best_trade_pct, stats.worst_trade_pct,
                stats.total_trades, stats.wins, stats.losses,
                stats.win_rate, stats.money_score, stats.rank
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.error(f"🎮 [MONEY GAME] Save player error: {e}")
    
    def get_elite_stocks(self) -> List[str]:
        """Get TOP 10 money-making stocks"""
        return self._elite_stocks[:]
    
    def get_elite_crypto(self) -> List[str]:
        """Get TOP 10 money-making crypto"""
        return self._elite_crypto[:]
    
    def get_leaderboard(self, asset_type: str, limit: int = 20) -> List[Dict]:
        """Get the money leaderboard"""
        pool = self._stock_players if asset_type == "stock" else self._crypto_players
        
        qualified = [p for p in pool.values() if p.total_trades >= self.MIN_TRADES]
        qualified.sort(key=lambda p: p.money_score, reverse=True)
        
        return [
            {
                "rank": p.rank,
                "symbol": p.symbol,
                "tier": p.tier.value,
                "money_score": p.money_score,
                "total_profit_pct": f"{p.total_profit_pct:+.1f}%",
                "avg_profit": f"{p.avg_profit_per_trade:+.2f}%",
                "best_trade": f"{p.best_trade_pct:+.1f}%",
                "worst_trade": f"{p.worst_trade_pct:+.1f}%",
                "total_trades": p.total_trades,
                "win_rate": f"{p.win_rate*100:.0f}%",
                "momentum": p.momentum,
                "rank_change": p.rank_change
            }
            for p in qualified[:limit]
        ]
    
    def get_rising_stars(self, asset_type: str, limit: int = 5) -> List[Dict]:
        """Get players closest to making TOP 10 (the next big deal!)"""
        pool = self._stock_players if asset_type == "stock" else self._crypto_players
        
        # Get non-elite players sorted by score
        rising = [
            p for p in pool.values()
            if p.tier != PlayerTier.ELITE and p.total_trades >= self.MIN_TRADES
        ]
        rising.sort(key=lambda p: p.money_score, reverse=True)
        
        return [
            {
                "rank": p.rank,
                "symbol": p.symbol,
                "money_score": p.money_score,
                "total_profit_pct": f"{p.total_profit_pct:+.1f}%",
                "momentum": p.momentum,
                "gap_to_top_10": p.rank - self.TOP_N if p.rank > self.TOP_N else 0
            }
            for p in rising[:limit]
        ]
    
    def get_game_status(self) -> Dict:
        """Get full game status"""
        return {
            "game_name": "GHOST MONEY GAME",
            "goal": "Find the BEST money makers - #1 = Most profit!",
            "rules": {
                "score": "Total profit potential",
                "loss_penalty": f"{self.LOSS_PENALTY}x (losses hurt!)",
                "big_win_bonus": f"+{self.BIG_WIN_THRESHOLD}% trades get {self.BIG_WIN_BONUS}x bonus",
                "min_trades": self.MIN_TRADES
            },
            "elite_stocks": self._elite_stocks,
            "elite_crypto": self._elite_crypto,
            "total_stock_players": len(self._stock_players),
            "total_crypto_players": len(self._crypto_players),
            "stock_leaderboard": self.get_leaderboard("stock", 15),
            "crypto_leaderboard": self.get_leaderboard("crypto", 15),
            "rising_stocks": self.get_rising_stars("stock", 5),
            "rising_crypto": self.get_rising_stars("crypto", 5)
        }
    
    def get_player_stats(self, symbol: str) -> Optional[Dict]:
        """Get detailed stats for a specific player"""
        # Check both pools
        if symbol in self._stock_players:
            p = self._stock_players[symbol]
        elif symbol in self._crypto_players:
            p = self._crypto_players[symbol]
        else:
            return None
        
        return {
            "symbol": p.symbol,
            "asset_type": p.asset_type,
            "tier": p.tier.value,
            "rank": p.rank,
            "rank_change": p.rank_change,
            "money_score": p.money_score,
            "total_profit_pct": f"{p.total_profit_pct:+.1f}%",
            "avg_profit_per_trade": f"{p.avg_profit_per_trade:+.2f}%",
            "best_trade": f"{p.best_trade_pct:+.1f}%",
            "worst_trade": f"{p.worst_trade_pct:+.1f}%",
            "total_trades": p.total_trades,
            "wins": p.wins,
            "losses": p.losses,
            "win_rate": f"{p.win_rate*100:.0f}%",
            "momentum": p.momentum,
            "is_elite": p.tier == PlayerTier.ELITE
        }
    
    def get_best_symbols_for_top10(self, asset_type: str, limit: int = 20) -> List[str]:
        """
        Get best symbols for TOP 10 selection based on Money Game data.
        
        Priority:
        1. Elite players (proven money makers with 5+ trades)
        2. Rising stars (profitable with some trades)  
        3. Any tracked players sorted by profit (cold start)
        
        Returns list of symbols in priority order.
        """
        pool = self._stock_players if asset_type == "stock" else self._crypto_players
        
        if not pool:
            return []
        
        # Priority 1: Elite (5+ trades, top performers)
        elite_symbols = self._elite_stocks if asset_type == "stock" else self._elite_crypto
        if elite_symbols:
            LOGGER.info(f"[MONEY-GAME] Using {len(elite_symbols)} ELITE {asset_type}s")
            return elite_symbols[:limit]
        
        # Priority 2: Anyone with trades, sorted by money_score
        # Even 1-2 trades gives us SOME data vs none
        all_players = sorted(
            [p for p in pool.values() if p.total_trades >= 1],  # At least 1 trade
            key=lambda p: (
                p.total_trades >= 3,  # Prefer 3+ trades
                p.money_score,        # Then by profit
            ),
            reverse=True
        )
        
        if all_players:
            LOGGER.info(f"[MONEY-GAME] Using {len(all_players)} ranked {asset_type}s (no elite yet)")
            return [p.symbol for p in all_players[:limit]]
        
        # Priority 3: Cold start - just return all tracked symbols
        LOGGER.info(f"[MONEY-GAME] Cold start - {len(pool)} {asset_type}s tracked but no trades yet")
        return list(pool.keys())[:limit]


# Singleton
_money_game: Optional[MoneyGameEngine] = None


def get_money_game() -> MoneyGameEngine:
    """Get or create the Money Game engine"""
    global _money_game
    if _money_game is None:
        _money_game = MoneyGameEngine()
    return _money_game
