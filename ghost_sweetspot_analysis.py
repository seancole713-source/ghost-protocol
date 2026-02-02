#!/usr/bin/env python3
"""
Ghost Sweet Spot Analysis
=========================
Analyzes 28K+ paper trades to find WHERE Ghost actually wins.

Run locally: python ghost_sweetspot_analysis.py
Run on Railway: railway run python ghost_sweetspot_analysis.py

Requires DATABASE_URL environment variable.
"""

import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Installing psycopg2...")
    os.system("pip install psycopg2-binary --break-system-packages -q")
    import psycopg2
    from psycopg2.extras import RealDictCursor


def get_connection():
    """Connect to database."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL not set")
        print("Run with: DATABASE_URL='postgres://...' python ghost_sweetspot_analysis.py")
        print("Or on Railway: railway run python ghost_sweetspot_analysis.py")
        sys.exit(1)
    return psycopg2.connect(db_url)


def print_header(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_table(headers, rows, highlight_fn=None):
    """Print formatted table."""
    if not rows:
        print("  No data")
        return
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-' * len(header_line)}")
    
    # Print rows
    for row in rows:
        line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        if highlight_fn and highlight_fn(row):
            print(f"  {line}  ← 🎯")
        else:
            print(f"  {line}")


def analyze_overall(cur):
    """Overall statistics."""
    print_header("OVERALL STATS")
    
    cur.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN final_outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN final_outcome IS NULL THEN 1 ELSE 0 END) as pending
        FROM paper_trades
    """)
    row = cur.fetchone()
    
    total = row['total']
    wins = row['wins'] or 0
    losses = row['losses'] or 0
    pending = row['pending'] or 0
    resolved = wins + losses
    win_rate = (wins / resolved * 100) if resolved > 0 else 0
    
    print(f"  Total trades:    {total:,}")
    print(f"  Resolved:        {resolved:,}")
    print(f"  Pending:         {pending:,}")
    print(f"  Wins:            {wins:,}")
    print(f"  Losses:          {losses:,}")
    print(f"  Win Rate:        {win_rate:.1f}%")
    
    if win_rate >= 60:
        print(f"\n  ✅ Overall win rate is TRADEABLE")
    elif win_rate >= 55:
        print(f"\n  ⚠️ Overall win rate is MARGINAL")
    else:
        print(f"\n  ❌ Overall win rate is NOT TRADEABLE (need 60%+)")
    
    return win_rate


def analyze_by_symbol(cur, min_trades=20):
    """Accuracy by symbol."""
    print_header(f"ACCURACY BY SYMBOL (min {min_trades} trades)")
    
    cur.execute("""
        SELECT 
            symbol,
            COUNT(*) as total,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL
        GROUP BY symbol
        HAVING COUNT(*) >= %s
        ORDER BY win_rate DESC
    """, (min_trades,))
    
    rows = cur.fetchall()
    
    # Top performers
    print("  TOP 15 (Highest Win Rate):")
    table_rows = [(r['symbol'], r['total'], r['wins'], f"{r['win_rate']}%") for r in rows[:15]]
    print_table(['Symbol', 'Trades', 'Wins', 'Win Rate'], table_rows, 
                lambda r: float(r[3].replace('%', '')) >= 60)
    
    # Bottom performers
    print("\n  BOTTOM 10 (Lowest Win Rate):")
    table_rows = [(r['symbol'], r['total'], r['wins'], f"{r['win_rate']}%") for r in rows[-10:]]
    print_table(['Symbol', 'Trades', 'Wins', 'Win Rate'], table_rows)
    
    # Tradeable symbols (60%+)
    tradeable = [r for r in rows if r['win_rate'] >= 60]
    print(f"\n  🎯 TRADEABLE SYMBOLS (60%+ win rate): {len(tradeable)}")
    if tradeable:
        for r in tradeable:
            print(f"     {r['symbol']}: {r['win_rate']}% over {r['total']} trades")
    
    return tradeable


def analyze_by_confidence(cur):
    """Accuracy by confidence bucket."""
    print_header("ACCURACY BY CONFIDENCE LEVEL")
    
    cur.execute("""
        SELECT 
            CASE 
                WHEN confidence >= 0.80 THEN '80-85%'
                WHEN confidence >= 0.75 THEN '75-79%'
                WHEN confidence >= 0.70 THEN '70-74%'
                WHEN confidence >= 0.65 THEN '65-69%'
                WHEN confidence >= 0.60 THEN '60-64%'
                WHEN confidence >= 0.55 THEN '55-59%'
                ELSE 'Below 55%'
            END as confidence_bucket,
            COUNT(*) as trades,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL AND confidence IS NOT NULL
        GROUP BY confidence_bucket
        ORDER BY confidence_bucket DESC
    """)
    
    rows = cur.fetchall()
    table_rows = [(r['confidence_bucket'], r['trades'], r['wins'], f"{r['win_rate']}%") for r in rows]
    print_table(['Confidence', 'Trades', 'Wins', 'Win Rate'], table_rows,
                lambda r: float(r[3].replace('%', '')) >= 60)
    
    # Find best confidence threshold
    best = max(rows, key=lambda r: r['win_rate']) if rows else None
    if best:
        print(f"\n  🎯 BEST CONFIDENCE BUCKET: {best['confidence_bucket']} → {best['win_rate']}% win rate")


def analyze_by_direction(cur):
    """Accuracy by prediction direction."""
    print_header("ACCURACY BY DIRECTION")
    
    cur.execute("""
        SELECT 
            direction,
            COUNT(*) as trades,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL AND direction IS NOT NULL
        GROUP BY direction
        ORDER BY win_rate DESC
    """)
    
    rows = cur.fetchall()
    table_rows = [(r['direction'], r['trades'], r['wins'], f"{r['win_rate']}%") for r in rows]
    print_table(['Direction', 'Trades', 'Wins', 'Win Rate'], table_rows,
                lambda r: float(r[3].replace('%', '')) >= 60)


def analyze_by_asset_type(cur):
    """Accuracy by crypto vs stocks."""
    print_header("ACCURACY BY ASSET TYPE")
    
    # Common crypto symbols
    crypto_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK', 'MATIC', 
                      'UNI', 'ATOM', 'LTC', 'BCH', 'NEAR', 'APT', 'ARB', 'OP', 'SUI',
                      'INJ', 'TIA', 'SEI', 'RNDR', 'FET', 'TURBO', 'PEPE', 'WIF', 'BONK',
                      'DOGE', 'SHIB', 'FIL', 'ICP', 'HBAR', 'VET', 'ALGO', 'SAND', 'MANA',
                      'AXS', 'GALA', 'ENJ', 'IMX', 'BLUR', 'APE', 'LDO', 'RPL', 'SSV',
                      'AAVE', 'MKR', 'CRV', 'SNX', 'COMP', 'SUSHI', 'YFI', '1INCH',
                      'GRT', 'ENS', 'BAT', 'ZRX', 'CHZ', 'ZEC', 'DASH', 'XMR', 'ETC']
    
    crypto_list = "', '".join(crypto_symbols)
    
    cur.execute(f"""
        SELECT 
            CASE WHEN symbol IN ('{crypto_list}') THEN 'CRYPTO' ELSE 'STOCK' END as asset_type,
            COUNT(*) as trades,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL
        GROUP BY asset_type
        ORDER BY win_rate DESC
    """)
    
    rows = cur.fetchall()
    table_rows = [(r['asset_type'], r['trades'], r['wins'], f"{r['win_rate']}%") for r in rows]
    print_table(['Asset Type', 'Trades', 'Wins', 'Win Rate'], table_rows,
                lambda r: float(r[3].replace('%', '')) >= 60)


def analyze_by_hold_time(cur):
    """Accuracy by hold duration."""
    print_header("ACCURACY BY HOLD TIME")
    
    cur.execute("""
        SELECT 
            CASE 
                WHEN EXTRACT(EPOCH FROM (target_time - created_at))/3600 <= 24 THEN '0-24hr'
                WHEN EXTRACT(EPOCH FROM (target_time - created_at))/3600 <= 48 THEN '24-48hr'
                WHEN EXTRACT(EPOCH FROM (target_time - created_at))/3600 <= 72 THEN '48-72hr'
                WHEN EXTRACT(EPOCH FROM (target_time - created_at))/3600 <= 120 THEN '72-120hr'
                ELSE '120hr+'
            END as hold_time,
            COUNT(*) as trades,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL 
          AND target_time IS NOT NULL 
          AND created_at IS NOT NULL
        GROUP BY hold_time
        ORDER BY hold_time
    """)
    
    rows = cur.fetchall()
    if rows:
        table_rows = [(r['hold_time'], r['trades'], r['wins'], f"{r['win_rate']}%") for r in rows]
        print_table(['Hold Time', 'Trades', 'Wins', 'Win Rate'], table_rows,
                    lambda r: float(r[3].replace('%', '')) >= 60)
    else:
        print("  No hold time data available")


def analyze_by_day_of_week(cur):
    """Accuracy by day trade was opened."""
    print_header("ACCURACY BY DAY OF WEEK")
    
    cur.execute("""
        SELECT 
            TO_CHAR(created_at, 'Day') as day_name,
            EXTRACT(DOW FROM created_at) as day_num,
            COUNT(*) as trades,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL AND created_at IS NOT NULL
        GROUP BY day_name, day_num
        ORDER BY day_num
    """)
    
    rows = cur.fetchall()
    if rows:
        table_rows = [(r['day_name'].strip(), r['trades'], r['wins'], f"{r['win_rate']}%") for r in rows]
        print_table(['Day', 'Trades', 'Wins', 'Win Rate'], table_rows,
                    lambda r: float(r[3].replace('%', '')) >= 60)


def analyze_recent_performance(cur, days=7):
    """Accuracy over last N days."""
    print_header(f"RECENT PERFORMANCE (Last {days} days)")
    
    cur.execute("""
        SELECT 
            DATE(created_at) as trade_date,
            COUNT(*) as trades,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN final_outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / 
                  NULLIF(SUM(CASE WHEN final_outcome IS NOT NULL THEN 1 ELSE 0 END), 0), 1) as win_rate
        FROM paper_trades
        WHERE created_at >= NOW() - INTERVAL '%s days'
        GROUP BY trade_date
        ORDER BY trade_date DESC
    """, (days,))
    
    rows = cur.fetchall()
    if rows:
        table_rows = [(str(r['trade_date']), r['trades'], r['wins'] or 0, r['losses'] or 0, 
                       f"{r['win_rate']}%" if r['win_rate'] else 'Pending') for r in rows]
        print_table(['Date', 'Trades', 'Wins', 'Losses', 'Win Rate'], table_rows,
                    lambda r: r[4] != 'Pending' and float(r[4].replace('%', '')) >= 60)


def analyze_symbol_direction_combo(cur, min_trades=15):
    """Find best symbol + direction combinations."""
    print_header(f"BEST SYMBOL + DIRECTION COMBOS (min {min_trades} trades)")
    
    cur.execute("""
        SELECT 
            symbol,
            direction,
            COUNT(*) as trades,
            SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL AND direction IS NOT NULL
        GROUP BY symbol, direction
        HAVING COUNT(*) >= %s
        ORDER BY win_rate DESC
        LIMIT 20
    """, (min_trades,))
    
    rows = cur.fetchall()
    if rows:
        table_rows = [(r['symbol'], r['direction'], r['trades'], f"{r['win_rate']}%") for r in rows]
        print_table(['Symbol', 'Direction', 'Trades', 'Win Rate'], table_rows,
                    lambda r: float(r[3].replace('%', '')) >= 65)
        
        # Recommendations
        strong = [r for r in rows if r['win_rate'] >= 65]
        if strong:
            print(f"\n  🎯 STRONG SIGNALS (65%+):")
            for r in strong:
                print(f"     {r['symbol']} {r['direction']}: {r['win_rate']}%")


def generate_recommendations(cur):
    """Generate final recommendations."""
    print_header("🎯 GHOST SWEET SPOT RECOMMENDATIONS")
    
    # Get tradeable symbols
    cur.execute("""
        SELECT symbol, 
               COUNT(*) as trades,
               ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL
        GROUP BY symbol
        HAVING COUNT(*) >= 20 AND 
               ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) >= 60
        ORDER BY win_rate DESC
    """)
    tradeable = cur.fetchall()
    
    # Get best confidence threshold
    cur.execute("""
        SELECT 
            CASE WHEN confidence >= 0.70 THEN 'HIGH' ELSE 'LOW' END as conf_level,
            COUNT(*) as trades,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL AND confidence IS NOT NULL
        GROUP BY conf_level
    """)
    conf_rows = {r['conf_level']: r for r in cur.fetchall()}
    
    # Get asset type performance
    crypto_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'LINK', 'RNDR', 'SUI']
    crypto_list = "', '".join(crypto_symbols)
    cur.execute(f"""
        SELECT 
            CASE WHEN symbol IN ('{crypto_list}') THEN 'CRYPTO' ELSE 'STOCK' END as asset_type,
            ROUND(100.0 * SUM(CASE WHEN final_outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM paper_trades
        WHERE final_outcome IS NOT NULL
        GROUP BY asset_type
    """)
    asset_perf = {r['asset_type']: r['win_rate'] for r in cur.fetchall()}
    
    print("  RECOMMENDED WHITELIST:")
    if tradeable:
        for r in tradeable[:10]:
            print(f"    ✅ {r['symbol']}: {r['win_rate']}% ({r['trades']} trades)")
    else:
        print("    ❌ No symbols with 60%+ win rate found")
    
    print("\n  CONFIDENCE THRESHOLD:")
    if 'HIGH' in conf_rows and 'LOW' in conf_rows:
        high = conf_rows['HIGH']
        low = conf_rows['LOW']
        print(f"    70%+ confidence: {high['win_rate']}% win rate ({high['trades']} trades)")
        print(f"    Below 70%:       {low['win_rate']}% win rate ({low['trades']} trades)")
        if high['win_rate'] > low['win_rate']:
            print(f"    → RECOMMENDATION: Only trade when confidence >= 70%")
    
    print("\n  ASSET TYPE:")
    for asset, rate in sorted(asset_perf.items(), key=lambda x: x[1], reverse=True):
        status = "✅" if rate >= 60 else "⚠️" if rate >= 55 else "❌"
        print(f"    {status} {asset}: {rate}%")
    
    print("\n" + "="*60)
    print("  GHOST V3 STRATEGY")
    print("="*60)
    
    if tradeable:
        symbols = [r['symbol'] for r in tradeable[:5]]
        print(f"\n  Whitelist: {', '.join(symbols)}")
        print(f"  Min confidence: 70%")
        print(f"  Expected win rate: 60%+")
        print(f"\n  Update in ghost_notifications.py V2_WHITELIST:")
        print(f"  V2_WHITELIST = {symbols}")
    else:
        print("\n  ⚠️ No symbols meet 60% threshold.")
        print("  Options:")
        print("    1. Lower threshold to 55% temporarily")
        print("    2. Collect more data")
        print("    3. Improve prediction model")


def main():
    print("\n" + "🔍 "*20)
    print("       GHOST SWEET SPOT ANALYSIS")
    print("🔍 "*20)
    
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        overall_rate = analyze_overall(cur)
        tradeable = analyze_by_symbol(cur)
        analyze_by_confidence(cur)
        analyze_by_direction(cur)
        analyze_by_asset_type(cur)
        analyze_by_hold_time(cur)
        analyze_by_day_of_week(cur)
        analyze_recent_performance(cur)
        analyze_symbol_direction_combo(cur)
        generate_recommendations(cur)
        
    finally:
        cur.close()
        conn.close()
    
    print("\n" + "="*60)
    print("  Analysis complete. Use findings to focus Ghost on winners.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
