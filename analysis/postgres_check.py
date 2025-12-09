#!/usr/bin/env python3
"""Direct Postgres query for outcomes data."""
import os
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL not set")
    exit(1)

# Replace internal hostname with localhost when running via Railway tunnel
if "railway.internal" in DATABASE_URL:
    print("⚠️  Railway internal hostname detected - using proxy")
    # Railway CLI tunnels to localhost
    DATABASE_URL = DATABASE_URL.replace("postgres.railway.internal", "localhost")

try:
    print(f"🔌 Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Check predictions
    cur.execute("SELECT COUNT(*) FROM ghost_predictions")
    total_preds = cur.fetchone()[0]
    print(f"📊 Total predictions: {total_preds}")
    
    # Check outcomes
    try:
        cur.execute("SELECT COUNT(*) FROM ghost_prediction_outcomes")
        total_outcomes = cur.fetchone()[0]
        print(f"📊 Total outcomes: {total_outcomes}")
        
        # Last 30 days
        cur.execute("""
            SELECT COUNT(*)
            FROM ghost_prediction_outcomes
            WHERE closed_at >= NOW() - INTERVAL '30 days'
            AND hit_direction IS NOT NULL
        """)
        recent = cur.fetchone()[0]
        print(f"📊 Outcomes (last 30d with data): {recent}")
        
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
            
            if row[2] >= 70:
                print("\n✅ MEETS 70% TARGET!")
            else:
                print(f"\n⚠️  Below 70% target (gap: {70 - row[2]:.2f}%)")
        else:
            print("\n⚠️  No outcomes in last 30 days - audit cannot proceed")
            
    except psycopg2.Error as e:
        print(f"❌ Outcomes table error: {e}")
        print("   Table may not exist - check migrations")
    
    cur.close()
    conn.close()
    
except psycopg2.OperationalError as e:
    print(f"❌ Connection error: {e}")
    print("\nTry running:")
    print("  railway link")
    print("  railway run python3 analysis/postgres_check.py")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
