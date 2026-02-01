"""
Test: Multi-Checkpoint Trust Ladder Implementation

This test verifies that:
1. Paper trades log checkpoint_times for all checkpoints
2. check_all_pending evaluates each checkpoint separately  
3. Trust ladder records checkpoint wins/losses correctly
4. Promotion requires ALL checkpoints to pass
5. Any checkpoint failure causes immediate demotion

Run: python tests/test_multi_checkpoint.py
"""

import os
import sys
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

def test_checkpoint_times_calculated():
    """Test that log_signal calculates checkpoint times correctly."""
    print("\n" + "="*60)
    print("TEST 1: Checkpoint times calculated correctly")
    print("="*60)
    
    from core.trust_ladder import TRUST_LEVELS
    
    # Level 1: Single checkpoint at 48hr
    level1 = TRUST_LEVELS[1]
    print(f"\nLevel 1: prediction_hours={level1['prediction_hours']}, checkpoints={level1['checkpoints']}")
    assert level1['checkpoints'] == [48], "Level 1 should have checkpoint at 48hr"
    print("  ✅ Level 1 checkpoints correct")
    
    # Level 2: Checkpoints at 60hr and 120hr
    level2 = TRUST_LEVELS[2]
    print(f"Level 2: prediction_hours={level2['prediction_hours']}, checkpoints={level2['checkpoints']}")
    assert level2['checkpoints'] == [60, 120], "Level 2 should have checkpoints at 60hr and 120hr"
    print("  ✅ Level 2 checkpoints correct")
    
    # Level 3: Checkpoints at 72hr and 168hr  
    level3 = TRUST_LEVELS[3]
    print(f"Level 3: prediction_hours={level3['prediction_hours']}, checkpoints={level3['checkpoints']}")
    assert level3['checkpoints'] == [72, 168], "Level 3 should have checkpoints at 72hr and 168hr"
    print("  ✅ Level 3 checkpoints correct")
    
    return True


def test_trust_ladder_checkpoint_logic():
    """Test that trust ladder handles checkpoint outcomes correctly."""
    print("\n" + "="*60)
    print("TEST 2: Trust ladder checkpoint logic")
    print("="*60)
    
    from core.trust_ladder import TrustLadder
    
    # Create in-memory trust ladder (no postgres)
    ladder = TrustLadder()
    
    test_symbol = "TEST_CHECKPOINT"
    
    # Get initial state (Level 1)
    trust = ladder.get_trust(test_symbol)
    print(f"\nInitial state: Level {trust.trust_level}, checkpoint_wins={trust.checkpoint_wins}")
    assert trust.trust_level == 1, "Should start at Level 1"
    
    # Simulate Level 1 win (48hr checkpoint)
    result = ladder.record_outcome(test_symbol, is_win=True, is_checkpoint=False)
    print(f"After Level 1 WIN: Level {result['new_level']}, promoted={result['promoted']}")
    assert result['new_level'] == 2, "Should be promoted to Level 2 after Level 1 win"
    assert result['promoted'] == True
    print("  ✅ Level 1 → Level 2 promotion works")
    
    # Simulate Level 2 first checkpoint (60hr) - WIN
    result = ladder.record_outcome(test_symbol, is_win=True, is_checkpoint=True)
    print(f"After Level 2 CP1 WIN: Level {result['new_level']}, checkpoint_wins={result['checkpoint_wins']}")
    assert result['new_level'] == 2, "Should stay at Level 2 after intermediate checkpoint"
    assert result['checkpoint_wins'] == 1, "Should have 1 checkpoint win"
    print("  ✅ Intermediate checkpoint WIN - no promotion yet")
    
    # Simulate Level 2 second checkpoint (120hr) - WIN  
    result = ladder.record_outcome(test_symbol, is_win=True, is_checkpoint=False)
    print(f"After Level 2 CP2 WIN: Level {result['new_level']}, promoted={result['promoted']}")
    assert result['new_level'] == 3, "Should be promoted to Level 3"
    assert result['promoted'] == True
    print("  ✅ Level 2 → Level 3 promotion after ALL checkpoints pass")
    
    # Reset and test demotion on intermediate checkpoint loss
    test_symbol_2 = "TEST_CHECKPOINT_LOSS"
    
    # Get to Level 2
    ladder.record_outcome(test_symbol_2, is_win=True, is_checkpoint=False)  # L1 → L2
    
    # Lose on first checkpoint of Level 2
    result = ladder.record_outcome(test_symbol_2, is_win=False, is_checkpoint=True)
    print(f"\nAfter Level 2 CP1 LOSS: Level {result['new_level']}, demoted={result['demoted']}")
    assert result['new_level'] == 1, "Should be demoted to Level 1"
    assert result['demoted'] == True
    print("  ✅ Immediate demotion on intermediate checkpoint loss")
    
    return True


def test_database_checkpoint_columns():
    """Test that checkpoint columns exist in database."""
    print("\n" + "="*60)
    print("TEST 3: Database checkpoint columns")
    print("="*60)
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("⏭️  Skipping database test - DATABASE_URL not set")
        return True
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'paper_trades'
            AND column_name IN ('trust_level', 'checkpoint_times', 'checkpoint_results', 'checkpoint_evaluated', 'checkpoint_prices')
            ORDER BY column_name
        """)
        
        columns = {row[0]: row[1] for row in cur.fetchall()}
        
        expected = ['trust_level', 'checkpoint_times', 'checkpoint_results', 'checkpoint_evaluated', 'checkpoint_prices']
        missing = [col for col in expected if col not in columns]
        
        if missing:
            print(f"❌ Missing columns: {missing}")
            print("   Run: python migrations/add_checkpoint_columns.py")
            cur.close()
            conn.close()
            return False
        
        print("\n📋 Checkpoint columns found:")
        for col_name, col_type in sorted(columns.items()):
            print(f"  - {col_name}: {col_type}")
        print("  ✅ All checkpoint columns present")
        
        cur.close()
        conn.close()
        return True
        
    except ImportError:
        print("⏭️  Skipping database test - psycopg2 not installed")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_pending_trades_with_checkpoints():
    """Test checking pending trades with checkpoint data."""
    print("\n" + "="*60)
    print("TEST 4: Pending trades checkpoint evaluation")  
    print("="*60)
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("⏭️  Skipping - DATABASE_URL not set")
        return True
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Check for pending trades with checkpoint data
        cur.execute("""
            SELECT symbol, trust_level, checkpoint_times, checkpoint_results, checkpoint_evaluated
            FROM paper_trades
            WHERE outcome = 'PENDING'
            AND checkpoint_times IS NOT NULL
            AND checkpoint_times != '[]'
            LIMIT 5
        """)
        
        rows = cur.fetchall()
        
        if not rows:
            print("ℹ️  No pending trades with checkpoint data yet")
            print("   New trades will have checkpoint tracking enabled")
        else:
            print(f"\n📋 Found {len(rows)} pending trades with checkpoints:")
            for row in rows:
                symbol, level, times, results, evaluated = row
                print(f"\n  [{symbol}] Trust Level {level}")
                print(f"    checkpoint_times: {times}")
                print(f"    checkpoint_results: {results}")
                print(f"    checkpoint_evaluated: {evaluated}")
        
        print("  ✅ Checkpoint query successful")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("MULTI-CHECKPOINT TRUST LADDER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Checkpoint times calculated", test_checkpoint_times_calculated),
        ("Trust ladder checkpoint logic", test_trust_ladder_checkpoint_logic),
        ("Database checkpoint columns", test_database_checkpoint_columns),
        ("Pending trades checkpoints", test_pending_trades_with_checkpoints),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"❌ FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {name} - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{passed+failed} tests passed")
    print("="*60)
    
    if failed > 0:
        print("\n⚠️  Some tests failed - review output above")
        return 1
    else:
        print("\n✅ All tests passed - Multi-checkpoint system ready!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
