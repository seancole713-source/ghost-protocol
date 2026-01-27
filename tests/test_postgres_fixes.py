#!/usr/bin/env python3
"""
Test PostgreSQL Fixes - Verify all broken synapses are now connected
====================================================================

Tests:
1. ml_trainer._fetch_training_data() → PostgreSQL
2. learning_loop._get_postgres_direction_accuracy() → PostgreSQL
3. Direct PostgreSQL query to ghost_prediction_outcomes
4. Verify data quality (25,691+ outcomes expected)

Usage:
    python3 test_postgres_fixes.py
"""

import os
import sys

def test_database_url():
    """Test 0: Verify DATABASE_URL is set"""
    print("=" * 70)
    print("TEST 0: DATABASE_URL Configuration")
    print("=" * 70)
    
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        print("❌ FAIL: DATABASE_URL not set")
        print("   Set DATABASE_URL environment variable to PostgreSQL connection string")
        return False
    
    if database_url.startswith(("postgres://", "postgresql://")):
        print(f"✅ PASS: DATABASE_URL configured (postgresql://...)")
        return True
    else:
        print(f"❌ FAIL: DATABASE_URL not PostgreSQL: {database_url[:50]}...")
        return False


def test_ml_trainer():
    """Test 1: ml_trainer._fetch_training_data()"""
    print("\n" + "=" * 70)
    print("TEST 1: ml_trainer._fetch_training_data() → PostgreSQL")
    print("=" * 70)
    
    try:
        from core.ml_trainer import _fetch_training_data
        
        # Fetch last 30 days
        data = _fetch_training_data(symbol=None, lookback_days=30)
        
        if len(data) == 0:
            print("❌ FAIL: No training data fetched")
            print("   This means PostgreSQL query returned 0 rows")
            return False
        
        print(f"✅ PASS: Fetched {len(data)} training samples from PostgreSQL")
        print(f"   Sample: symbol={data[0].get('symbol')}, "
              f"direction_correct={data[0].get('direction_correct')}, "
              f"confidence={data[0].get('confidence')}")
        
        # Verify features exist
        if data[0].get('features'):
            print(f"   Features: {len(data[0]['features'])} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: ml_trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_learning_loop():
    """Test 2: learning_loop._get_postgres_direction_accuracy()"""
    print("\n" + "=" * 70)
    print("TEST 2: learning_loop._get_postgres_direction_accuracy()")
    print("=" * 70)
    
    try:
        from core.learning_loop import get_learning_loop
        
        ll = get_learning_loop()
        accuracy = ll._get_postgres_direction_accuracy(days=7)
        
        if "error" in accuracy:
            print(f"❌ FAIL: {accuracy['error']}")
            return False
        
        if accuracy.get("count", 0) == 0:
            print("❌ FAIL: No outcomes found in last 7 days")
            return False
        
        print(f"✅ PASS: Accuracy calculated from PostgreSQL")
        print(f"   Total: {accuracy['count']} outcomes")
        print(f"   Correct: {accuracy['correct']} ({accuracy['accuracy_pct']:.2f}%)")
        print(f"   Incorrect: {accuracy['incorrect']}")
        print(f"   Data source: {accuracy.get('data_source', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: learning_loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_postgres():
    """Test 3: Direct PostgreSQL query"""
    print("\n" + "=" * 70)
    print("TEST 3: Direct PostgreSQL Query (ghost_prediction_outcomes)")
    print("=" * 70)
    
    try:
        import psycopg2
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            print("❌ FAIL: DATABASE_URL not set")
            return False
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Count total outcomes
        cur.execute("SELECT COUNT(*) FROM ghost_prediction_outcomes")
        total = cur.fetchone()[0]
        
        if total == 0:
            print("❌ FAIL: No outcomes in ghost_prediction_outcomes table")
            cur.close()
            conn.close()
            return False
        
        print(f"✅ PASS: {total} total outcomes in PostgreSQL")
        
        # Get closed outcomes
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE hit_direction = 1) as hits,
                COUNT(*) FILTER (WHERE hit_direction = 0) as misses,
                COUNT(*) FILTER (WHERE hit_direction IS NULL) as pending
            FROM ghost_prediction_outcomes
            WHERE status = 'closed'
        """)
        row = cur.fetchone()
        
        closed = row[0]
        hits = row[1] or 0
        misses = row[2] or 0
        pending = row[3] or 0
        
        print(f"   Closed: {closed} ({hits} hits, {misses} misses, {pending} null)")
        
        if closed > 0:
            accuracy = hits / closed * 100
            print(f"   Accuracy: {accuracy:.2f}%")
        
        # Check recent activity (last 7 days)
        cur.execute("""
            SELECT COUNT(*)
            FROM ghost_prediction_outcomes
            WHERE closed_at > NOW() - INTERVAL '7 days'
        """)
        recent = cur.fetchone()[0]
        print(f"   Recent (7d): {recent} outcomes")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Direct PostgreSQL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality():
    """Test 4: Data Quality Check"""
    print("\n" + "=" * 70)
    print("TEST 4: Data Quality (Features, Symbols, Completeness)")
    print("=" * 70)
    
    try:
        import psycopg2
        database_url = os.getenv("DATABASE_URL")
        
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        
        # Check for null/corrupt data
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE symbol IS NULL) as null_symbol,
                COUNT(*) FILTER (WHERE open_price IS NULL OR open_price <= 0) as null_open,
                COUNT(*) FILTER (WHERE close_price IS NULL OR close_price <= 0) as null_close,
                COUNT(*) as total
            FROM ghost_prediction_outcomes
            WHERE status = 'closed'
        """)
        row = cur.fetchone()
        
        null_symbol = row[0] or 0
        null_open = row[1] or 0
        null_close = row[2] or 0
        total = row[3] or 0
        
        issues = null_symbol + null_open + null_close
        
        if issues > 0:
            print(f"⚠️  WARNING: {issues} data quality issues found:")
            print(f"   - Null symbols: {null_symbol}")
            print(f"   - Null/zero open_price: {null_open}")
            print(f"   - Null/zero close_price: {null_close}")
        else:
            print(f"✅ PASS: No data quality issues (checked {total} closed outcomes)")
        
        # Check symbol distribution
        cur.execute("""
            SELECT symbol, COUNT(*) as count
            FROM ghost_prediction_outcomes
            WHERE status = 'closed'
            GROUP BY symbol
            ORDER BY count DESC
            LIMIT 10
        """)
        
        print("\n   Top 10 symbols by outcome count:")
        for row in cur.fetchall():
            print(f"     {row[0]}: {row[1]} outcomes")
        
        cur.close()
        conn.close()
        
        return issues == 0
        
    except Exception as e:
        print(f"❌ FAIL: Data quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("GHOST PROTOCOL - POSTGRESQL FIXES TEST SUITE")
    print("Testing: ml_trainer, learning_loop, direct PostgreSQL access")
    print("=" * 70)
    
    results = []
    
    # Test 0: DATABASE_URL
    results.append(("DATABASE_URL", test_database_url()))
    
    # Only continue if DATABASE_URL is set
    if not results[0][1]:
        print("\n❌ CRITICAL: DATABASE_URL not configured. Cannot run remaining tests.")
        return 1
    
    # Test 1: ml_trainer
    results.append(("ml_trainer", test_ml_trainer()))
    
    # Test 2: learning_loop
    results.append(("learning_loop", test_learning_loop()))
    
    # Test 3: Direct PostgreSQL
    results.append(("direct_postgres", test_direct_postgres()))
    
    # Test 4: Data Quality
    results.append(("data_quality", test_data_quality()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - PostgreSQL synapses are GREEN!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed - Review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
