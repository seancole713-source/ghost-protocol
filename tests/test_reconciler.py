#!/usr/bin/env python3
"""
Test reconciler directly by calling the reconcile_outcomes_v2 function.
This bypasses API auth and tests the core reconciliation logic.
"""
import os
import sys

# Set environment to use production Postgres (disable dual-write to avoid SQLite)
os.environ['PREDICTION_STORE_ENGINE'] = 'postgres'
os.environ['PREDICTION_DUAL_WRITE'] = '0'

print("=" * 70)
print("GHOST ACCURACY RECONCILIATION TEST")
print("=" * 70)

try:
    from services.outcome_reconciler_v2 import reconcile_outcomes_v2
    from core.prediction_store import get_prediction_store
    from sqlalchemy import text
    import time
    
    print("\n📊 Checking predictions in database...")
    
    store = get_prediction_store()
    now = time.time()
    cutoff_48h = now - (48 * 3600)
    
    if hasattr(store, 'engine') and store.engine:
        with store.engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM ghost_predictions")).scalar()
            ready = conn.execute(text(
                "SELECT COUNT(*) FROM ghost_predictions WHERE run_at < :cutoff"
            ), {"cutoff": cutoff_48h}).scalar()
            outcomes_before = conn.execute(text("SELECT COUNT(*) FROM ghost_prediction_outcomes")).scalar()
            
            print(f"Total predictions: {total}")
            print(f"Ready for reconciliation (>48h old): {ready}")
            print(f"Existing outcomes: {outcomes_before}")
            
            if ready == 0:
                print("\n⚠️  No predictions ready for reconciliation yet")
                print("   All predictions are < 48h old")
                print("   Wait until some predictions pass their 48h horizon")
                sys.exit(0)
            
            print(f"\n🔄 Running reconciliation on {ready} predictions...")
            
    # Run reconciliation
    results = reconcile_outcomes_v2()
    
    print(f"\n✅ Reconciliation complete!")
    print(f"Results: {results}")
    
    # Check outcomes after
    if hasattr(store, 'engine') and store.engine:
        with store.engine.connect() as conn:
            outcomes_after = conn.execute(text("SELECT COUNT(*) FROM ghost_prediction_outcomes")).scalar()
            
            print(f"\nOutcomes before: {outcomes_before}")
            print(f"Outcomes after: {outcomes_after}")
            print(f"New outcomes: {outcomes_after - outcomes_before}")
            
            if outcomes_after > 0:
                # Get accuracy summary
                accuracy_results = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                        AVG(CASE WHEN hit_direction = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_pct
                    FROM ghost_prediction_outcomes
                    WHERE hit_direction IS NOT NULL
                """)).fetchone()
                
                if accuracy_results:
                    total_outcomes, correct, accuracy = accuracy_results
                    print(f"\n" + "=" * 70)
                    print("ACCURACY RESULTS")
                    print("=" * 70)
                    print(f"Total reconciled: {total_outcomes}")
                    print(f"Correct: {correct}")
                    print(f"Accuracy: {accuracy:.1f}%")
                    
                    if accuracy >= 70:
                        print(f"\n🎯 SUCCESS: Ghost meets 70% accuracy target!")
                    elif accuracy >= 60:
                        print(f"\n⚠️  Close: {70 - accuracy:.1f}% below 70% target")
                    else:
                        print(f"\n❌ Below target: {70 - accuracy:.1f}% gap to 70%")
                    
                    print("=" * 70)
                    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("This script needs to run in the Ghost environment with all dependencies")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
