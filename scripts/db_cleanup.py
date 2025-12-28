#!/usr/bin/env python3
"""
Ghost Protocol Database Cleanup Script
=======================================
Audits and cleans corrupt data from PostgreSQL.

Usage:
    python3 db_cleanup.py --audit           # Show corrupt data (safe)
    python3 db_cleanup.py --clean --dry-run # Preview deletions
    python3 db_cleanup.py --clean           # Actually delete

Requires:
    export DATABASE_URL="postgresql://..."
    pip install psycopg2-binary
"""

import os
import sys
import argparse
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ Missing psycopg2. Install with: pip install psycopg2-binary")
    sys.exit(1)


# =============================================================================
# PRICE VALIDATION THRESHOLDS
# =============================================================================

MIN_VALID_PRICES = {
    'BTC': 10000,    # BTC should never be below $10k
    'ETH': 500,      # ETH should never be below $500
    'SOL': 5,        # SOL should never be below $5
    'BNB': 100,      # BNB should never be below $100
    'XRP': 0.10,     # XRP should never be below $0.10
    'ADA': 0.05,     # ADA should never be below $0.05
    'DOGE': 0.001,   # DOGE should never be below $0.001
    'AVAX': 5,       # AVAX should never be below $5
    'DOT': 2,        # DOT should never be below $2
    'LINK': 3,       # LINK should never be below $3
    'MATIC': 0.10,   # MATIC should never be below $0.10
    'UNI': 2,        # UNI should never be below $2
    'ATOM': 3,       # ATOM should never be below $3
    'LTC': 30,       # LTC should never be below $30
}

MAX_VALID_PRICES = {
    'BTC': 500000,   # BTC ceiling
    'ETH': 50000,    # ETH ceiling
    'DOGE': 10,      # DOGE ceiling
    'SHIB': 0.01,    # SHIB ceiling
}

# Symbols that were identified as problematic from real Telegram data
BAD_SYMBOLS = {'SAND', 'FLOW', 'HBAR', 'ILV', 'BAND', 'DIA'}


def get_connection():
    """Get PostgreSQL connection from DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        print("   Set it with: export DATABASE_URL='postgresql://...'")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(database_url)
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)


def audit_predictions(conn):
    """Audit predictions table for corrupt data."""
    print("\n" + "=" * 70)
    print("📊 AUDITING PREDICTIONS TABLE")
    print("=" * 70)
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get table overview
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            MIN(created_at) as earliest,
            MAX(created_at) as latest,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM predictions
    """)
    overview = cursor.fetchone()
    
    print(f"\nTotal predictions: {overview['total']:,}")
    print(f"Date range: {overview['earliest']} to {overview['latest']}")
    print(f"Unique symbols: {overview['unique_symbols']}")
    
    # Check for corrupt prices by symbol
    print("\n🔍 Checking for corrupt prices...")
    
    corrupt_by_symbol = {}
    
    for symbol, min_price in MIN_VALID_PRICES.items():
        cursor.execute("""
            SELECT COUNT(*) as cnt, 
                   MIN(price_at_prediction) as min_price,
                   MAX(price_at_prediction) as max_price
            FROM predictions
            WHERE symbol = %s AND price_at_prediction < %s
        """, (symbol, min_price))
        result = cursor.fetchone()
        
        if result['cnt'] > 0:
            corrupt_by_symbol[symbol] = {
                'count': result['cnt'],
                'min_price': result['min_price'],
                'max_price': result['max_price'],
                'threshold': min_price,
                'reason': 'below_minimum'
            }
    
    # Check for above maximum
    for symbol, max_price in MAX_VALID_PRICES.items():
        cursor.execute("""
            SELECT COUNT(*) as cnt,
                   MIN(price_at_prediction) as min_price,
                   MAX(price_at_prediction) as max_price
            FROM predictions
            WHERE symbol = %s AND price_at_prediction > %s
        """, (symbol, max_price))
        result = cursor.fetchone()
        
        if result['cnt'] > 0:
            if symbol in corrupt_by_symbol:
                corrupt_by_symbol[symbol]['count'] += result['cnt']
            else:
                corrupt_by_symbol[symbol] = {
                    'count': result['cnt'],
                    'min_price': result['min_price'],
                    'max_price': result['max_price'],
                    'threshold': max_price,
                    'reason': 'above_maximum'
                }
    
    # Check for zero/negative prices
    cursor.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM predictions
        WHERE price_at_prediction <= 0
        GROUP BY symbol
    """)
    zero_prices = cursor.fetchall()
    
    for row in zero_prices:
        symbol = row['symbol']
        if symbol in corrupt_by_symbol:
            corrupt_by_symbol[symbol]['count'] += row['cnt']
        else:
            corrupt_by_symbol[symbol] = {
                'count': row['cnt'],
                'min_price': 0,
                'max_price': 0,
                'threshold': 0,
                'reason': 'zero_or_negative'
            }
    
    # Print corrupt data summary
    if corrupt_by_symbol:
        print("\n❌ CORRUPT DATA FOUND:")
        total_corrupt = 0
        for symbol, data in sorted(corrupt_by_symbol.items(), key=lambda x: -x[1]['count']):
            total_corrupt += data['count']
            print(f"\n  {symbol}: {data['count']} corrupt records")
            print(f"    Price range: ${data['min_price']:.2f} - ${data['max_price']:.2f}")
            print(f"    Reason: {data['reason']} (threshold: ${data['threshold']})")
        
        print(f"\n📊 TOTAL CORRUPT PREDICTIONS: {total_corrupt}")
    else:
        print("\n✅ No corrupt prices found based on thresholds!")
    
    # Check for bad symbols (from Telegram analysis)
    print("\n🔍 Checking for problematic symbols (from Telegram analysis)...")
    
    cursor.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM predictions
        WHERE symbol IN %s
        GROUP BY symbol
        ORDER BY cnt DESC
    """, (tuple(BAD_SYMBOLS),))
    bad_symbol_counts = cursor.fetchall()
    
    if bad_symbol_counts:
        print("\n⚠️ PROBLEMATIC SYMBOLS FOUND:")
        for row in bad_symbol_counts:
            print(f"  {row['symbol']}: {row['cnt']} predictions")
        print("\n  These symbols had stop-losses triggered or wrong directions")
    else:
        print("\n✅ No problematic symbols found")
    
    cursor.close()
    return corrupt_by_symbol


def audit_outcomes(conn):
    """Audit outcomes table for corrupt data."""
    print("\n" + "=" * 70)
    print("📊 AUDITING OUTCOMES TABLE")
    print("=" * 70)
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get table overview
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN status = 'no_data' THEN 1 ELSE 0 END) as no_data,
            MIN(closed_at) as earliest,
            MAX(closed_at) as latest
        FROM ghost_prediction_outcomes
    """)
    overview = cursor.fetchone()
    
    if overview['total']:
        accuracy = (overview['wins'] / overview['total']) * 100 if overview['total'] > 0 else 0
        print(f"\nTotal outcomes: {overview['total']:,}")
        print(f"  Wins: {overview['wins']:,}")
        print(f"  Losses: {overview['losses']:,}")
        print(f"  No data: {overview['no_data']:,}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"Date range: {overview['earliest']} to {overview['latest']}")
    else:
        print("\nNo outcomes found in table")
    
    # Check for orphan outcomes (prediction_id doesn't exist)
    cursor.execute("""
        SELECT COUNT(*) as orphans
        FROM ghost_prediction_outcomes o
        LEFT JOIN predictions p ON o.prediction_id = p.id
        WHERE p.id IS NULL
    """)
    orphans = cursor.fetchone()
    
    if orphans and orphans['orphans'] > 0:
        print(f"\n⚠️ ORPHAN OUTCOMES: {orphans['orphans']}")
        print("   (These point to deleted predictions)")
    
    # Check for corrupt prices in outcomes
    print("\n🔍 Checking for corrupt resolution prices...")
    
    cursor.execute("""
        SELECT symbol, COUNT(*) as cnt,
               MIN(price_at_resolution) as min_price,
               MAX(price_at_resolution) as max_price
        FROM ghost_prediction_outcomes
        WHERE price_at_resolution <= 0 OR price_at_resolution IS NULL
        GROUP BY symbol
    """)
    corrupt_outcomes = cursor.fetchall()
    
    if corrupt_outcomes:
        print("\n❌ CORRUPT OUTCOME PRICES:")
        for row in corrupt_outcomes:
            print(f"  {row['symbol']}: {row['cnt']} records with invalid resolution price")
    else:
        print("\n✅ No corrupt resolution prices found")
    
    cursor.close()


def clean_corrupt_data(conn, dry_run=True):
    """Clean corrupt data from database."""
    print("\n" + "=" * 70)
    if dry_run:
        print("🔍 DRY RUN - PREVIEWING CLEANUP (no changes)")
    else:
        print("🧹 CLEANING CORRUPT DATA")
    print("=" * 70)
    
    cursor = conn.cursor()
    
    # Build list of corrupt prediction IDs
    corrupt_ids = []
    
    # 1. Find predictions with prices below minimum
    for symbol, min_price in MIN_VALID_PRICES.items():
        cursor.execute("""
            SELECT id FROM predictions
            WHERE symbol = %s AND price_at_prediction < %s
        """, (symbol, min_price))
        ids = [row[0] for row in cursor.fetchall()]
        corrupt_ids.extend(ids)
        if ids:
            print(f"  {symbol}: {len(ids)} predictions below ${min_price}")
    
    # 2. Find predictions with prices above maximum
    for symbol, max_price in MAX_VALID_PRICES.items():
        cursor.execute("""
            SELECT id FROM predictions
            WHERE symbol = %s AND price_at_prediction > %s
        """, (symbol, max_price))
        ids = [row[0] for row in cursor.fetchall()]
        corrupt_ids.extend(ids)
        if ids:
            print(f"  {symbol}: {len(ids)} predictions above ${max_price}")
    
    # 3. Find predictions with zero/negative prices
    cursor.execute("""
        SELECT id FROM predictions WHERE price_at_prediction <= 0
    """)
    zero_ids = [row[0] for row in cursor.fetchall()]
    corrupt_ids.extend(zero_ids)
    if zero_ids:
        print(f"  Zero/negative prices: {len(zero_ids)} predictions")
    
    # Remove duplicates
    corrupt_ids = list(set(corrupt_ids))
    
    print(f"\n📊 Total corrupt predictions to delete: {len(corrupt_ids)}")
    
    if not corrupt_ids:
        print("✅ No corrupt data to clean!")
        return
    
    if dry_run:
        print("\n⚠️ DRY RUN - No changes made")
        print("   Run with --clean (without --dry-run) to delete")
        return
    
    # Actually delete
    print("\n🗑️ Deleting corrupt data...")
    
    # First delete related outcomes
    cursor.execute("""
        DELETE FROM ghost_prediction_outcomes
        WHERE prediction_id = ANY(%s)
    """, (corrupt_ids,))
    outcomes_deleted = cursor.rowcount
    print(f"  Deleted {outcomes_deleted} related outcomes")
    
    # Then delete predictions
    cursor.execute("""
        DELETE FROM predictions
        WHERE id = ANY(%s)
    """, (corrupt_ids,))
    predictions_deleted = cursor.rowcount
    print(f"  Deleted {predictions_deleted} corrupt predictions")
    
    conn.commit()
    print("\n✅ Cleanup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Protocol Database Cleanup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 db_cleanup.py --audit           # Show corrupt data (safe)
  python3 db_cleanup.py --clean --dry-run # Preview what would be deleted
  python3 db_cleanup.py --clean           # Actually delete corrupt data
        """
    )
    parser.add_argument('--audit', action='store_true', help='Audit database for corrupt data')
    parser.add_argument('--clean', action='store_true', help='Clean corrupt data')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without making them')
    
    args = parser.parse_args()
    
    if not args.audit and not args.clean:
        print("❌ Specify --audit or --clean")
        parser.print_help()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🔧 GHOST PROTOCOL DATABASE CLEANUP")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    conn = get_connection()
    
    try:
        if args.audit:
            audit_predictions(conn)
            audit_outcomes(conn)
        
        if args.clean:
            clean_corrupt_data(conn, dry_run=args.dry_run)
    
    finally:
        conn.close()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
