#!/usr/bin/env python3
"""
Check production Postgres database for predictions ready for reconciliation.
Connects directly to DATABASE_URL from Railway.
"""
import os
import sys
from sqlalchemy import create_engine, text
from datetime import datetime
import time

# Get DATABASE_URL from environment (Railway)
database_url = os.getenv('DATABASE_URL')
if not database_url:
    print("❌ DATABASE_URL not found in environment")
    print("Run: railway run python3 check_production_predictions.py")
    sys.exit(1)

print("=" * 70)
print("GHOST PRODUCTION PREDICTION DIAGNOSTICS")
print("=" * 70)

try:
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        now = time.time()
        cutoff_48h = now - (48 * 3600)
        cutoff_7d = now - (7 * 86400)
        
        # Count total predictions
        total = conn.execute(text("SELECT COUNT(*) FROM ghost_predictions")).scalar()
        print(f"\n📊 Total Predictions: {total}")
        
        # Count predictions ready for reconciliation (>48h old)
        ready_48h = conn.execute(text(
            "SELECT COUNT(*) FROM ghost_predictions WHERE run_at < :cutoff"
        ), {"cutoff": cutoff_48h}).scalar()
        print(f"⏰ Ready for Reconciliation (>48h): {ready_48h}")
        
        # Count recent predictions
        recent_7d = conn.execute(text(
            "SELECT COUNT(*) FROM ghost_predictions WHERE run_at > :cutoff"
        ), {"cutoff": cutoff_7d}).scalar()
        print(f"🆕 Recent (Last 7 days): {recent_7d}")
        
        # Get date range
        oldest = conn.execute(text("SELECT MIN(run_at) FROM ghost_predictions")).scalar()
        newest = conn.execute(text("SELECT MAX(run_at) FROM ghost_predictions")).scalar()
        
        if oldest:
            oldest_dt = datetime.fromtimestamp(oldest)
            newest_dt = datetime.fromtimestamp(newest)
            age_days = (now - oldest) / 86400
            
            print(f"\n📅 Date Range:")
            print(f"  Oldest: {oldest_dt.isoformat()} ({age_days:.1f} days ago)")
            print(f"  Newest: {newest_dt.isoformat()}")
        
        # Count outcomes
        outcomes_total = conn.execute(text("SELECT COUNT(*) FROM ghost_prediction_outcomes")).scalar()
        print(f"\n✅ Reconciled Outcomes: {outcomes_total}")
        
        # Reconciliation status
        print(f"\n🔍 Reconciliation Status:")
        print(f"  Expected: {ready_48h} predictions ready")
        print(f"  Actual: {outcomes_total} outcomes")
        print(f"  Missing: {ready_48h - outcomes_total}")
        
        if outcomes_total == 0 and ready_48h > 0:
            print(f"\n⚠️  WARNING: {ready_48h} predictions ready but 0 outcomes!")
            print(f"   Reconciler may not be running or failing silently.")
        elif outcomes_total > 0:
            rate = (outcomes_total / ready_48h * 100) if ready_48h > 0 else 0
            print(f"\n✅ Reconciliation Rate: {rate:.1f}%")
        
        # Sample predictions
        print(f"\n📝 Sample Predictions Ready for Reconciliation:")
        samples = conn.execute(text("""
            SELECT id, symbol, run_at, horizon_h, direction, confidence
            FROM ghost_predictions
            WHERE run_at < :cutoff
            ORDER BY run_at DESC
            LIMIT 10
        """), {"cutoff": cutoff_48h}).fetchall()
        
        if samples:
            for row in samples:
                pred_id, symbol, run_at, horizon_h, direction, confidence = row
                pred_dt = datetime.fromtimestamp(run_at)
                age_h = (now - run_at) / 3600
                print(f"  ID {pred_id:6} {symbol:6} {direction:4} {confidence:5.2f} "
                      f"{pred_dt.strftime('%m-%d %H:%M')} ({age_h:.0f}h ago)")
        else:
            print("  (No predictions ready)")
        
        print("\n" + "=" * 70)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
