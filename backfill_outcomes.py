#!/usr/bin/env python3
"""
Backfill Outcomes for Recent Predictions
=========================================
Finds predictions from the last 7 days that are >48h old and manually
triggers outcome reconciliation with Polygon historical prices.

This accelerates the bootstrap process by immediately seeding the calibration
system with real outcomes instead of waiting for new predictions to age.
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Setup environment
sys.path.insert(0, "/workspaces/ghost-protocol")
os.environ.setdefault("GHOST_PREDICT_DB", "./data/ghost_predictions.db")

from core.prediction_store import get_prediction_store
from services.outcome_reconciler_v2 import _get_price_at_time, _reconcile_single_v2

def backfill_recent_outcomes(max_backfill=100, min_age_hours=48, max_age_days=7):
    """
    Find recent predictions ready for reconciliation and process them.
    
    Args:
        max_backfill: Max predictions to process in one run
        min_age_hours: Minimum age (48h for Ghost's prediction window)
        max_age_days: Maximum age (7 days for Polygon free tier)
    
    Returns:
        dict with reconciliation stats
    """
    print("=" * 70)
    print("  BACKFILL OUTCOMES — BOOTSTRAP ACCELERATION")
    print("=" * 70)
    print()
    
    store = get_prediction_store()
    
    # Calculate time windows
    now = time.time()
    min_timestamp = now - (max_age_days * 86400)  # 7 days ago
    max_timestamp = now - (min_age_hours * 3600)   # 48 hours ago
    
    min_dt = datetime.fromtimestamp(min_timestamp)
    max_dt = datetime.fromtimestamp(max_timestamp)
    
    print(f"Looking for predictions:")
    print(f"  • Created between: {min_dt} and {max_dt}")
    print(f"  • Age range: {min_age_hours}h to {max_age_days}d")
    print(f"  • Max to process: {max_backfill}")
    print()
    
    # Query for predictions in the window
    # SQLite query to find predictions that:
    # 1. Are older than 48h (resolution window closed)
    # 2. Are newer than 7d (within Polygon free tier)
    # 3. Don't have outcomes yet (NOT EXISTS in ghost_prediction_outcomes)
    
    try:
        backend = store.backend
        
        # Get candidates (this is a simplified query; production uses get_pending_outcomes)
        query = """
            SELECT 
                id, symbol, run_at, direction, confidence, 
                price_at_prediction, expected_move_pct
            FROM predictions
            WHERE run_at BETWEEN ? AND ?
            AND id NOT IN (
                SELECT prediction_id FROM ghost_prediction_outcomes
            )
            ORDER BY run_at DESC
            LIMIT ?
        """
        
        # For SQLite backend
        if hasattr(backend, 'conn'):
            cursor = backend.conn.cursor()
            cursor.execute(query, (min_timestamp, max_timestamp, max_backfill))
            candidates = cursor.fetchall()
        else:
            print("⚠️  Warning: Using fallback query method")
            candidates = []
        
        print(f"Found {len(candidates)} predictions ready for backfill")
        print()
        
        if not candidates:
            print("✅ No predictions in the 48h-7d window")
            print("   System is up-to-date!")
            return {"backfilled": 0, "no_candidates": True}
        
        # Process each candidate
        success = 0
        no_data = 0
        errors = 0
        
        for idx, row in enumerate(candidates, 1):
            pred_id, symbol, run_at, direction, confidence, price_at_pred, expected_move = row
            
            pred_dt = datetime.fromtimestamp(run_at)
            age_hours = (now - run_at) / 3600
            
            print(f"[{idx}/{len(candidates)}] {symbol} (ID={pred_id})")
            print(f"  Created: {pred_dt} ({age_hours:.1f}h ago)")
            print(f"  Price at prediction: ${price_at_pred:.4f}")
            
            # Build prediction dict for reconciliation
            pred = {
                "id": pred_id,
                "symbol": symbol,
                "run_at": run_at,
                "direction": direction,
                "confidence": confidence,
                "price_at_prediction": price_at_pred,
                "expected_move_pct": expected_move,
                "horizon_hours": 48
            }
            
            # Attempt reconciliation
            try:
                result = _reconcile_single_v2(pred)
                
                if result == "success":
                    print(f"  ✅ Reconciled successfully")
                    success += 1
                elif result == "no_data":
                    print(f"  ⚠️  No historical price available")
                    no_data += 1
                else:
                    print(f"  ⚠️  Result: {result}")
                    errors += 1
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                errors += 1
            
            print()
            
            # Rate limiting for Polygon API
            if idx < len(candidates):
                time.sleep(0.2)  # 5 req/sec max on free tier
        
        # Summary
        print("=" * 70)
        print("BACKFILL COMPLETE")
        print("=" * 70)
        print(f"  ✅ Success: {success}/{len(candidates)}")
        print(f"  ⚠️  No Data: {no_data}/{len(candidates)}")
        print(f"  ❌ Errors: {errors}/{len(candidates)}")
        print()
        
        if success >= 30:
            print("🎉 MILESTONE: 30+ outcomes reconciled!")
            print("   → Calibration should activate on next prediction cycle")
            print("   → Check /api/v3/predictions/latest for stage5_ok=true")
        elif success > 0:
            print(f"⏳ Progress: {success}/30 outcomes needed for calibration")
            print(f"   → {30 - success} more needed to activate stage5/stage6 gates")
        else:
            print("⚠️  No outcomes reconciled")
            print("   → May need to wait for predictions within Polygon's 7-day window")
        
        return {
            "backfilled": success,
            "no_data": no_data,
            "errors": errors,
            "total_candidates": len(candidates)
        }
        
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"backfilled": 0, "error": str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill outcomes for recent predictions")
    parser.add_argument("--max", type=int, default=100, help="Max predictions to process")
    parser.add_argument("--min-age", type=int, default=48, help="Minimum age in hours (default: 48)")
    parser.add_argument("--max-age", type=int, default=7, help="Maximum age in days (default: 7)")
    
    args = parser.parse_args()
    
    result = backfill_recent_outcomes(
        max_backfill=args.max,
        min_age_hours=args.min_age,
        max_age_days=args.max_age
    )
    
    sys.exit(0 if result.get("backfilled", 0) > 0 else 1)
