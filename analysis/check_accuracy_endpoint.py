#!/usr/bin/env python3
"""
Manual Accuracy Check Endpoint
===============================

Add this to ghost_endpoints.py or run standalone to get accuracy stats.

Quick Check URL: /api/v3/accuracy/check
"""

from fastapi import APIRouter
from typing import Dict
import os
import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter()


@router.get("/api/v3/accuracy/check")
async def check_accuracy_data() -> Dict:
    """
    Quick check of prediction outcomes table and accuracy stats.
    
    Returns:
        - Table existence
        - Row counts
        - Last 30d accuracy if data exists
    """
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        return {"error": "DATABASE_URL not set"}
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'ghost_prediction_outcomes'
            )
        """)
        table_exists = cur.fetchone()['exists']
        
        if not table_exists:
            return {
                "table_exists": False,
                "message": "ghost_prediction_outcomes table not created yet",
                "action": "Run: railway run python3 apply_outcome_migration.py (from Railway shell)"
            }
        
        # Count predictions
        cur.execute("SELECT COUNT(*) as count FROM ghost_predictions")
        total_preds = cur.fetchone()['count']
        
        # Count outcomes
        cur.execute("SELECT COUNT(*) as count FROM ghost_prediction_outcomes")
        total_outcomes = cur.fetchone()['count']
        
        # Last 30 days with data
        cur.execute("""
            SELECT COUNT(*) as count
            FROM ghost_prediction_outcomes
            WHERE closed_at >= NOW() - INTERVAL '30 days'
            AND hit_direction IS NOT NULL
        """)
        recent_outcomes = cur.fetchone()['count']
        
        # Calculate accuracy if data exists
        accuracy_30d = None
        if recent_outcomes > 0:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                    ROUND(SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100, 2) as accuracy_pct
                FROM ghost_prediction_outcomes
                WHERE closed_at >= NOW() - INTERVAL '30 days'
                AND hit_direction IS NOT NULL
            """)
            row = cur.fetchone()
            accuracy_30d = {
                "total": row['total'],
                "correct": row['correct'],
                "accuracy": float(row['accuracy_pct']),
                "meets_70_target": float(row['accuracy_pct']) >= 70
            }
        
        # Check pending reconciliations
        cur.execute("""
            SELECT COUNT(*) as count
            FROM ghost_predictions gp
            LEFT JOIN ghost_prediction_outcomes gpo ON gp.id = gpo.prediction_id
            WHERE gpo.id IS NULL
            AND (gp.run_at + (gp.horizon_h * INTERVAL '1 hour')) <= NOW()
        """)
        pending = cur.fetchone()['count']
        
        cur.close()
        conn.close()
        
        return {
            "table_exists": True,
            "total_predictions": total_preds,
            "total_outcomes": total_outcomes,
            "recent_outcomes_30d": recent_outcomes,
            "pending_reconciliations": pending,
            "accuracy_30d": accuracy_30d,
            "status": "✅ Data available" if recent_outcomes > 0 else "⚠️ No outcomes yet - reconciler needs to run"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "traceback": str(e.__class__.__name__)
        }


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(check_accuracy_data())
    
    print("\n" + "=" * 60)
    print("GHOST ACCURACY DATA CHECK")
    print("=" * 60)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    elif not result.get("table_exists"):
        print(f"⚠️  {result['message']}")
        print(f"\nAction: {result['action']}")
    else:
        print(f"✅ Table exists: ghost_prediction_outcomes")
        print(f"\n📊 Counts:")
        print(f"   Total predictions: {result['total_predictions']}")
        print(f"   Total outcomes: {result['total_outcomes']}")
        print(f"   Recent outcomes (30d): {result['recent_outcomes_30d']}")
        print(f"   Pending reconciliations: {result['pending_reconciliations']}")
        
        if result['accuracy_30d']:
            acc = result['accuracy_30d']
            print(f"\n✨ Accuracy (Last 30 days):")
            print(f"   Total: {acc['total']}")
            print(f"   Correct: {acc['correct']}")
            print(f"   Accuracy: {acc['accuracy']}%")
            if acc['meets_70_target']:
                print(f"   ✅ MEETS 70% TARGET!")
            else:
                print(f"   ⚠️  Below 70% target")
        else:
            print(f"\n{result['status']}")
    
    print("=" * 60)
