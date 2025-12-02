#!/usr/bin/env python3
"""
Apply Postgres Outcome Migration
================================
Applies migrations/002_prediction_outcomes.sql to production Postgres.

Usage:
    railway run python3 apply_outcome_migration.py
"""
import os
import sys
import psycopg2
from pathlib import Path

def apply_outcome_migration():
    """Apply the outcome tracking migration to Postgres."""
    
    # Get DATABASE_URL from environment
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Read migration file
    migration_file = Path(__file__).parent / "migrations" / "002_prediction_outcomes.sql"
    if not migration_file.exists():
        print(f"❌ ERROR: Migration file not found: {migration_file}")
        sys.exit(1)
    
    print(f"📁 Reading migration: {migration_file}")
    migration_sql = migration_file.read_text()
    
    # Connect to Postgres
    print(f"🔌 Connecting to Postgres...")
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Apply migration
        print("⚙️  Applying migration 002_prediction_outcomes.sql...")
        cursor.execute(migration_sql)
        
        # Verify table created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
              AND table_name = 'ghost_prediction_outcomes'
        """)
        
        if cursor.fetchone():
            print("✅ Table ghost_prediction_outcomes created successfully")
        else:
            print("❌ ERROR: Table creation failed")
            sys.exit(1)
        
        # Verify views created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.views 
            WHERE table_schema = 'public' 
              AND table_name LIKE 'v_accuracy%'
        """)
        
        views = cursor.fetchall()
        print(f"✅ Created {len(views)} accuracy views")
        for view in views:
            print(f"   - {view[0]}")
        
        # Check for predictions needing reconciliation
        cursor.execute("""
            SELECT COUNT(*) 
            FROM ghost_predictions gp
            LEFT JOIN ghost_prediction_outcomes gpo ON gp.id = gpo.prediction_id
            WHERE gpo.id IS NULL
              AND (gp.run_at + (gp.horizon_h * INTERVAL '1 hour')) <= NOW()
        """)
        
        pending_count = cursor.fetchone()[0]
        print(f"\n📊 Found {pending_count} predictions ready for outcome reconciliation")
        
        # Commit
        conn.commit()
        print("\n✅ Migration 002_prediction_outcomes.sql applied successfully!")
        
        print("\n📝 Next steps:")
        print("   1. Start outcome reconciler background task")
        print("   2. Fix accuracy API endpoint syntax errors")
        print("   3. Test /api/v3/accuracy/summary returns real data")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        conn.rollback()
        sys.exit(1)
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    apply_outcome_migration()
