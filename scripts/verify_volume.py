#!/usr/bin/env python3
"""
Railway Volume Verification Script

Checks if persistent volume is properly mounted and writable.
Run this after deploying to Railway with volume configuration.
"""

import os
import sqlite3
import sys
from pathlib import Path


def main():
    """Verify Railway volume mount and database setup"""
    
    print("=" * 60)
    print("RAILWAY VOLUME VERIFICATION")
    print("=" * 60)
    
    data_dir = Path("/app/data")
    db_path = Path(os.getenv("GHOST_PREDICT_DB", "/app/data/ghost_predictions.db"))
    
    checks = {
        "data_dir_exists": False,
        "data_dir_writable": False,
        "volume_mounted": False,
        "db_path_correct": False,
        "db_writable": False,
        "predictions_exist": False,
    }
    
    # Check 1: Data directory exists
    print("\n[1/6] Checking data directory exists...")
    if data_dir.exists():
        checks["data_dir_exists"] = True
        print(f"  ✅ {data_dir} exists")
    else:
        print(f"  ❌ {data_dir} does NOT exist")
        print(f"     Run: mkdir -p {data_dir}")
    
    # Check 2: Data directory writable
    print("\n[2/6] Checking data directory is writable...")
    try:
        test_file = data_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        checks["data_dir_writable"] = True
        print(f"  ✅ {data_dir} is writable")
    except Exception as e:
        print(f"  ❌ {data_dir} is NOT writable: {e}")
        print(f"     Run: chmod 777 {data_dir}")
    
    # Check 3: Volume mounted (check if it's persistent storage)
    print("\n[3/6] Checking if Railway volume is mounted...")
    # Railway volumes set RAILWAY_VOLUME_MOUNT_PATH environment variable
    volume_mount = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    if volume_mount:
        checks["volume_mounted"] = True
        print(f"  ✅ Railway volume mounted at: {volume_mount}")
    else:
        print(f"  ⚠️  RAILWAY_VOLUME_MOUNT_PATH not set")
        print(f"     Volume may not be configured yet")
        print(f"     See: PHASE3_RAILWAY_VOLUME_SETUP.md")
    
    # Check 4: Database path matches environment
    print("\n[4/6] Checking database path configuration...")
    expected = "/app/data/ghost_predictions.db"
    actual = str(db_path)
    if actual == expected:
        checks["db_path_correct"] = True
        print(f"  ✅ GHOST_PREDICT_DB = {actual}")
    else:
        print(f"  ❌ Path mismatch!")
        print(f"     Expected: {expected}")
        print(f"     Actual:   {actual}")
        print(f"     Set: GHOST_PREDICT_DB={expected}")
    
    # Check 5: Database writable
    print("\n[5/6] Checking database is writable...")
    try:
        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to open/create database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS _test (id INTEGER)")
        conn.execute("INSERT INTO _test VALUES (1)")
        conn.execute("DROP TABLE _test")
        conn.commit()
        conn.close()
        
        checks["db_writable"] = True
        print(f"  ✅ Database writable at: {db_path}")
        
        # Check file size
        if db_path.exists():
            size = db_path.stat().st_size
            print(f"     Database size: {size:,} bytes")
    except Exception as e:
        print(f"  ❌ Database NOT writable: {e}")
    
    # Check 6: Predictions exist
    print("\n[6/6] Checking for existing predictions...")
    try:
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM predictions")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                checks["predictions_exist"] = True
                print(f"  ✅ Found {count} predictions in database")
                
                # Show latest prediction
                conn = sqlite3.connect(str(db_path))
                cursor = conn.execute(
                    "SELECT symbol, run_at, confidence, direction FROM predictions "
                    "ORDER BY run_at DESC LIMIT 1"
                )
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    from datetime import datetime
                    symbol, run_at, confidence, direction = row
                    dt = datetime.fromtimestamp(run_at)
                    print(f"     Latest: {symbol} @ {dt} ({direction}, {confidence:.1%} confidence)")
            else:
                print(f"  ⚠️  No predictions in database yet")
                print(f"     Run: POST /api/predict/force to generate predictions")
        else:
            print(f"  ⚠️  Database file does not exist yet")
            print(f"     Will be created on first prediction")
    except Exception as e:
        print(f"  ⚠️  Could not query predictions: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {check}")
    
    print(f"\n  Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 All checks passed! Volume is properly configured.")
        print("     Predictions will persist across redeploys.")
        sys.exit(0)
    elif checks["volume_mounted"]:
        print("\n  ⚠️  Volume mounted but some checks failed.")
        print("     Check file permissions and environment variables.")
        sys.exit(1)
    else:
        print("\n  ❌ Railway volume NOT configured yet!")
        print("\n  ACTION REQUIRED:")
        print("     1. Open Railway dashboard")
        print("     2. Go to your service → Variables tab")
        print("     3. Click '+ New Volume'")
        print("     4. Mount path: /app/data")
        print("     5. Size: 1GB")
        print("     6. Click 'Add' → Railway will auto-redeploy")
        print("\n  See: PHASE3_RAILWAY_VOLUME_SETUP.md for detailed instructions")
        sys.exit(1)


if __name__ == "__main__":
    main()
