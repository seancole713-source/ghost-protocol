#!/usr/bin/env python3
"""Quick check of prediction outcomes data."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.prediction_store import get_prediction_store
    print("✅ Prediction store imported successfully")
    
    store = get_prediction_store()
    print(f"✅ Store backend: {type(store.backend).__name__}")
    
    # Try to count predictions
    conn = store.backend.get_connection()
    cur = conn.cursor()
    
    # Check ghost_predictions table
    cur.execute("SELECT COUNT(*) FROM ghost_predictions")
    total_preds = cur.fetchone()[0]
    print(f"📊 Total predictions: {total_preds}")
    
    # Check if outcomes table exists
    try:
        cur.execute("SELECT COUNT(*) FROM ghost_prediction_outcomes")
        total_outcomes = cur.fetchone()[0]
        print(f"📊 Total outcomes: {total_outcomes}")
        
        # Check recent outcomes
        cur.execute("""
            SELECT COUNT(*) 
            FROM ghost_prediction_outcomes 
            WHERE closed_at >= NOW() - INTERVAL '30 days'
            AND hit_direction IS NOT NULL
        """)
        recent = cur.fetchone()[0]
        print(f"📊 Outcomes (last 30d): {recent}")
        
        if recent > 0:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                    ROUND(SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100, 2) as accuracy
                FROM ghost_prediction_outcomes
                WHERE closed_at >= NOW() - INTERVAL '30 days'
                AND hit_direction IS NOT NULL
            """)
            row = cur.fetchone()
            print(f"\n✨ QUICK ACCURACY (Last 30 days):")
            print(f"   Total: {row[0]}")
            print(f"   Correct: {row[1]}")
            print(f"   Accuracy: {row[2]}%")
        else:
            print("\n⚠️  No outcomes in last 30 days")
            
    except Exception as e:
        print(f"❌ Outcomes table check failed: {e}")
        print("   (Table may not exist yet)")
    
    cur.close()
    store.backend.release_connection(conn)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
