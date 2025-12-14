#!/usr/bin/env python3
"""
Apply Symbol Column Migration
==============================
Adds symbol column to ghost_prediction_outcomes table for by-symbol analytics.

Usage:
    railway run python3 apply_symbol_migration.py
    # OR locally:
    python3 apply_symbol_migration.py
"""
import os
import sys
import psycopg2
from pathlib import Path

def apply_symbol_migration():
    """Apply the symbol column migration to Postgres."""
    
    # Get DATABASE_URL from environment
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Read migration file
    migration_file = Path(__file__).parent / "migrations" / "003_add_symbol_to_outcomes.sql"
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
        print("⚙️  Applying migration 003_add_symbol_to_outcomes.sql...")
        cursor.execute(migration_sql)
        conn.commit()
        
        # Verify column added
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
              AND table_name = 'ghost_prediction_outcomes'
              AND column_name = 'symbol'
        """)
        
        if cursor.fetchone():
            print("✅ Column 'symbol' added to ghost_prediction_outcomes successfully")
        else:
            print("❌ ERROR: Column addition failed")
            sys.exit(1)
        
        # Verify index created
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
              AND tablename = 'ghost_prediction_outcomes'
              AND indexname = 'idx_outcomes_symbol'
        """)
        
        if cursor.fetchone():
            print("✅ Index 'idx_outcomes_symbol' created successfully")
        else:
            print("⚠️  WARNING: Index creation may have failed")
        
        print("\n🎉 Migration 003 applied successfully!")
        print("   - Symbol column added for by-symbol accuracy tracking")
        print("   - Index created for fast symbol-based queries")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ ERROR: Database operation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    apply_symbol_migration()
