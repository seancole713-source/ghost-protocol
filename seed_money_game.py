#!/usr/bin/env python3
"""
🎮 SEED MONEY GAME - Initialize TOP 10 with known performers

This seeds the Money Game with manual TOP 10 so we can:
1. See immediate leaderboard results
2. Watch Ghost naturally promote/demote based on real performance
3. Confirm the system works when Ghost promotes a NEW symbol

Run once to seed, then let Ghost take over!
"""

import os
import psycopg2
from datetime import datetime, timedelta

DATABASE_URL = os.getenv("DATABASE_URL")

# MANUAL TOP 10 SEEDS - Known good performers
# These get initial stats to qualify for leaderboard
STOCK_SEEDS = [
    # (symbol, initial_profit%, initial_trades, win_rate)
    ("NVDA", 15.0, 5, 0.80),   # AI leader
    ("META", 12.0, 5, 0.80),   # Strong performer
    ("PLTR", 10.0, 4, 0.75),   # AI/Defense play
    ("COIN", 8.0, 4, 0.75),    # Crypto proxy
    ("GOOGL", 7.0, 4, 0.75),   # Tech giant
    ("AMZN", 6.0, 4, 0.75),    # E-commerce leader
    ("AAPL", 5.0, 5, 0.60),    # Stable performer
    ("MSFT", 4.0, 5, 0.60),    # Cloud giant
    ("AMD", 3.0, 4, 0.75),     # Chip play
    ("TSLA", 2.0, 4, 0.50),    # Volatile but popular
]

CRYPTO_SEEDS = [
    # (symbol, initial_profit%, initial_trades, win_rate)
    ("BTC", 12.0, 5, 0.80),    # King
    ("ETH", 10.0, 5, 0.80),    # Queen
    ("SOL", 15.0, 4, 0.75),    # High performer
    ("XRP", 8.0, 4, 0.75),     # Payments
    ("RNDR", 20.0, 5, 0.80),   # AI/GPU - your best performer!
    ("TURBO", 18.0, 4, 0.75),  # High win rate
    ("ADA", 5.0, 4, 0.60),     # Cardano
    ("DOGE", 6.0, 4, 0.60),    # Meme king
    ("LINK", 7.0, 4, 0.70),    # Oracle
    ("AVAX", 8.0, 4, 0.70),    # L1 competitor
]

def seed_money_game():
    """Seed the Money Game with initial TOP 10"""
    if not DATABASE_URL:
        print("❌ No DATABASE_URL - run on Railway or set locally")
        return
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    print("🎮 SEEDING MONEY GAME...")
    print("=" * 50)
    
    # Seed stocks
    print("\n📈 STOCKS TOP 10:")
    for rank, (symbol, profit, trades, win_rate) in enumerate(STOCK_SEEDS, 1):
        wins = int(trades * win_rate)
        losses = trades - wins
        avg_profit = profit / trades
        money_score = profit * (1 + win_rate)  # Simple score formula
        
        cur.execute("""
            INSERT INTO money_game_players 
            (symbol, asset_type, tier, total_profit_pct, avg_profit_per_trade,
             best_trade_pct, worst_trade_pct, total_trades, wins, losses,
             win_rate, money_score, recent_profit_pct, momentum, rank, rank_change,
             last_trade, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                tier = EXCLUDED.tier,
                total_profit_pct = EXCLUDED.total_profit_pct,
                avg_profit_per_trade = EXCLUDED.avg_profit_per_trade,
                total_trades = EXCLUDED.total_trades,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                win_rate = EXCLUDED.win_rate,
                money_score = EXCLUDED.money_score,
                rank = EXCLUDED.rank,
                last_updated = NOW()
        """, (
            symbol, "stock", "elite", profit, avg_profit,
            profit * 0.4, -profit * 0.1, trades, wins, losses,
            win_rate, money_score, profit * 0.3, "stable", rank, 0,
            datetime.utcnow(), datetime.utcnow()
        ))
        print(f"   {rank}. {symbol} - +{profit:.1f}% profit, {trades} trades, {win_rate*100:.0f}% win rate")
    
    # Seed crypto
    print("\n🪙 CRYPTO TOP 10:")
    for rank, (symbol, profit, trades, win_rate) in enumerate(CRYPTO_SEEDS, 1):
        wins = int(trades * win_rate)
        losses = trades - wins
        avg_profit = profit / trades
        money_score = profit * (1 + win_rate)
        
        cur.execute("""
            INSERT INTO money_game_players 
            (symbol, asset_type, tier, total_profit_pct, avg_profit_per_trade,
             best_trade_pct, worst_trade_pct, total_trades, wins, losses,
             win_rate, money_score, recent_profit_pct, momentum, rank, rank_change,
             last_trade, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO UPDATE SET
                tier = EXCLUDED.tier,
                total_profit_pct = EXCLUDED.total_profit_pct,
                avg_profit_per_trade = EXCLUDED.avg_profit_per_trade,
                total_trades = EXCLUDED.total_trades,
                wins = EXCLUDED.wins,
                losses = EXCLUDED.losses,
                win_rate = EXCLUDED.win_rate,
                money_score = EXCLUDED.money_score,
                rank = EXCLUDED.rank,
                last_updated = NOW()
        """, (
            symbol, "crypto", "elite", profit, avg_profit,
            profit * 0.4, -profit * 0.1, trades, wins, losses,
            win_rate, money_score, profit * 0.3, "stable", rank, 0,
            datetime.utcnow(), datetime.utcnow()
        ))
        print(f"   {rank}. {symbol} - +{profit:.1f}% profit, {trades} trades, {win_rate*100:.0f}% win rate")
    
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 50)
    print("✅ MONEY GAME SEEDED!")
    print("\n🎯 NEXT STEPS:")
    print("   1. Ghost will scout new predictions daily")
    print("   2. After 24h, trades resolve and real profit counted")
    print("   3. If a NEW symbol beats the seeds, it gets PROMOTED!")
    print("   4. Poor performers get DEMOTED")
    print("\n👀 Watch for: A symbol YOU didn't seed making TOP 10")
    print("   That proves the system works!")


if __name__ == "__main__":
    seed_money_game()
