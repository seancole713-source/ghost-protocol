#!/usr/bin/env python3
"""
Verify Postgres watchlist migration and prediction storage
"""
import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

# Use Railway DATABASE_URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:JGSaBXzDIOzHAgoXUtHPWDXCoqPkaoNV@postgres.railway.internal:5432/railway"
)

def check_table_exists(cursor, table_name):
    """Check if a table exists in the database"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
    """, (table_name,))
    return cursor.fetchone()['exists']

def get_table_row_count(cursor, table_name):
    """Get row count for a table"""
    try:
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name};")
        return cursor.fetchone()['count']
    except Exception as e:
        return f"ERROR: {e}"

def get_predictions_info(cursor):
    """Get prediction statistics"""
    cursor.execute("""
        SELECT 
            MIN(id) as min_id,
            MAX(id) as max_id,
            COUNT(*) as total_count,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM ghost_predictions;
    """)
    return cursor.fetchone()

def get_recent_predictions(cursor, limit=15):
    """Get recent predictions"""
    cursor.execute("""
        SELECT id, symbol, direction, confidence, created_at
        FROM ghost_predictions
        ORDER BY id DESC
        LIMIT %s;
    """, (limit,))
    return cursor.fetchall()

def main():
    print("=" * 80)
    print("POSTGRES WATCHLIST MIGRATION VERIFICATION")
    print("=" * 80)
    print()
    
    try:
        # Connect to Postgres
        print(f"📡 Connecting to Postgres...")
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected successfully\n")
        
        # Check watchlist tables
        print("=" * 80)
        print("WATCHLIST TABLES STATUS")
        print("=" * 80)
        watchlist_tables = [
            "ghost_watchlist_items",
            "watchlist_prediction_tracking",
            "watchlist_price_snapshots",
            "watchlist_alerts_log"
        ]
        
        watchlist_status = {}
        for table in watchlist_tables:
            exists = check_table_exists(cursor, table)
            watchlist_status[table] = exists
            if exists:
                row_count = get_table_row_count(cursor, table)
                print(f"✅ {table}: EXISTS ({row_count} rows)")
            else:
                print(f"❌ {table}: MISSING")
        print()
        
        # Check predictions table
        print("=" * 80)
        print("PREDICTIONS TABLE STATUS")
        print("=" * 80)
        
        if check_table_exists(cursor, "ghost_predictions"):
            print("✅ ghost_predictions: EXISTS\n")
            
            stats = get_predictions_info(cursor)
            print(f"📊 Prediction Statistics:")
            print(f"   - ID Range: {stats['min_id']} to {stats['max_id']}")
            print(f"   - Total Predictions: {stats['total_count']}")
            print(f"   - Unique Symbols: {stats['unique_symbols']}")
            print()
            
            print("📋 Recent Predictions (last 15):")
            recent = get_recent_predictions(cursor, 15)
            for pred in recent:
                print(f"   ID {pred['id']:3d} | {pred['symbol']:6s} | {pred['direction']:4s} | "
                      f"Conf: {pred['confidence']:.1%} | {pred['created_at']}")
            print()
            
            # Check for predictions 9-12 specifically
            print("🔍 Checking Predictions 9-12 (User Test Cases):")
            cursor.execute("""
                SELECT id, symbol, direction, confidence, created_at
                FROM ghost_predictions
                WHERE id IN (9, 10, 11, 12)
                ORDER BY id;
            """)
            test_preds = cursor.fetchall()
            if test_preds:
                for pred in test_preds:
                    print(f"   ✅ ID {pred['id']:2d} | {pred['symbol']:6s} | {pred['direction']:4s} | "
                          f"Conf: {pred['confidence']:.1%} | {pred['created_at']}")
            else:
                print("   ⚠️  No predictions found with IDs 9-12")
                print("   (Note: This may indicate predictions were created earlier and IDs are higher)")
            print()
        else:
            print("❌ ghost_predictions: MISSING\n")
        
        # Summary
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        all_watchlist_exist = all(watchlist_status.values())
        if all_watchlist_exist:
            print("✅ All 4 watchlist tables exist in Postgres")
        else:
            missing = [t for t, exists in watchlist_status.items() if not exists]
            print(f"❌ Missing watchlist tables: {', '.join(missing)}")
            print(f"   Run: railway run psql $DATABASE_URL -f migrations/001_personal_watchlist.sql")
        
        if check_table_exists(cursor, "ghost_predictions") and stats['total_count'] > 0:
            print(f"✅ Postgres is operational with {stats['total_count']} predictions stored")
            print(f"✅ Prediction IDs range from {stats['min_id']} to {stats['max_id']}")
        
        cursor.close()
        conn.close()
        
        return 0 if all_watchlist_exist else 1
        
    except psycopg2.OperationalError as e:
        print(f"❌ Cannot connect to Postgres: {e}")
        print()
        print("💡 This script must be run from Railway environment with DATABASE_URL set")
        print("   Run: railway run python3 verify_postgres_migration.py")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
