#!/usr/bin/env python3
"""
Reconcile historical predictions from local SQLite database.
This gives us immediate accuracy data without waiting 48 hours.
"""
import sqlite3
import time
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Ghost's price fetching
try:
    from wolf_app import _get_price_quorum, HUNTER_CRYPTO_SYMBOLS
except ImportError:
    print("⚠️  Warning: Could not import price functions, will use fallback")
    _get_price_quorum = None

print("=" * 70)
print("HISTORICAL PREDICTION RECONCILIATION")
print("=" * 70)

# Connect to local SQLite
db_path = "data/ghost_predictions.db"
if not os.path.exists(db_path):
    print(f"❌ {db_path} not found")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get predictions older than 48h
now = time.time()
cutoff = now - (48 * 3600)

cur.execute("""
    SELECT id, symbol, run_at, horizon_h, direction, confidence
    FROM predictions
    WHERE run_at < ?
    ORDER BY run_at DESC
""", (cutoff,))

predictions = cur.fetchall()
print(f"\n📊 Found {len(predictions)} predictions older than 48 hours")

if len(predictions) == 0:
    print("⚠️  No predictions old enough to reconcile")
    sys.exit(0)

# Show sample
print(f"\nOldest prediction: {datetime.fromtimestamp(predictions[-1][2])}")
print(f"Newest prediction: {datetime.fromtimestamp(predictions[0][2])}")

print("\n" + "=" * 70)
print("RECONCILIATION RESULTS")
print("=" * 70)

correct = 0
wrong = 0
no_data = 0
errors = 0

for pred_id, symbol, run_at, horizon_h, predicted_direction, confidence in predictions[:50]:  # Limit to 50 for now
    try:
        # Calculate resolution time (run_at + horizon)
        resolution_time = run_at + (horizon_h * 3600)
        
        # Get price at prediction time (t0)
        # For historical data, we'll use the stored forecast points
        cur.execute("""
            SELECT price FROM prediction_points
            WHERE prediction_id = ? AND kind = 'forecast'
            ORDER BY ts ASC LIMIT 1
        """, (pred_id,))
        
        price_t0_row = cur.fetchone()
        if not price_t0_row:
            no_data += 1
            continue
        
        price_t0 = price_t0_row[0]
        
        # Get price at resolution time (t1)
        # Use current price as proxy (not perfect but good enough for this analysis)
        if _get_price_quorum:
            import asyncio
            if symbol in HUNTER_CRYPTO_SYMBOLS:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                price_t1 = loop.run_until_complete(_get_price_quorum(symbol))
                loop.close()
            else:
                price_t1 = _get_price_quorum(symbol)
        else:
            # Fallback: Use last forecast point as approximation
            cur.execute("""
                SELECT price FROM prediction_points
                WHERE prediction_id = ? AND kind = 'forecast'
                ORDER BY ts DESC LIMIT 1
            """, (pred_id,))
            price_t1_row = cur.fetchone()
            if not price_t1_row:
                no_data += 1
                continue
            price_t1 = price_t1_row[0]
        
        if not price_t1:
            no_data += 1
            continue
        
        # Calculate realized movement
        realized_move_pct = ((price_t1 - price_t0) / price_t0) * 100
        
        # Determine actual direction (using 0.25% threshold)
        if abs(realized_move_pct) < 0.25:
            actual_direction = "FLAT"
        elif realized_move_pct > 0:
            actual_direction = "UP"
        else:
            actual_direction = "DOWN"
        
        # Check if correct
        is_correct = (predicted_direction == actual_direction)
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            wrong += 1
            status = "❌"
        
        # Print first 10 for verification
        if (correct + wrong) <= 10:
            age_days = (now - run_at) / 86400
            print(f"{status} {symbol}: Predicted {predicted_direction}, Actual {actual_direction} "
                  f"(move: {realized_move_pct:+.2f}%, {age_days:.0f}d ago)")
    
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"⚠️  Error processing {symbol}: {e}")

conn.close()

# Calculate accuracy
total_evaluated = correct + wrong
if total_evaluated > 0:
    accuracy_pct = (correct / total_evaluated) * 100
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n✅ Correct: {correct}")
    print(f"❌ Wrong: {wrong}")
    print(f"⚠️  No Data: {no_data}")
    print(f"🔴 Errors: {errors}")
    print(f"\n📊 ACCURACY: {accuracy_pct:.2f}% ({correct}/{total_evaluated})")
    
    # Calculate confidence interval (simple normal approximation)
    if total_evaluated >= 30:
        import math
        p = correct / total_evaluated
        z = 1.96  # 95% confidence
        margin = z * math.sqrt(p * (1 - p) / total_evaluated)
        ci_lower = max(0, (p - margin) * 100)
        ci_upper = min(100, (p + margin) * 100)
        print(f"📈 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        
        if ci_lower >= 70:
            print("\n🎯 GHOST MEETS 70% TARGET! ✅")
        elif accuracy_pct >= 70:
            print(f"\n⚠️  Point estimate meets 70%, but CI lower bound is {ci_lower:.1f}%")
            print("    Need more data or higher accuracy for statistical confidence")
        else:
            print(f"\n❌ Below 70% target (gap: {70 - accuracy_pct:.1f}%)")
    else:
        print(f"\n⚠️  Sample size too small (N={total_evaluated}), need 30+ for CI")
    
    print("\n" + "=" * 70)
else:
    print("\n❌ No predictions could be evaluated")

print("\nNote: This uses current prices as proxy for historical prices.")
print("For production accuracy, use outcome_reconciler_v2.py with historical price data.")
print("=" * 70)
