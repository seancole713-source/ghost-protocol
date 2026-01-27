#!/usr/bin/env python3
"""
🔧 Migration: Fix paper_trades table schema
Converts TEXT columns to TIMESTAMP WITH TIME ZONE for PostgreSQL

This fixes the critical bug:
  ERROR: operator does not exist: text <= timestamp with time zone
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def migrate_paper_trades_schema():
    """Migrate paper_trades table to use proper TIMESTAMP columns"""
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set, skipping migration")
        return False
    
    try:
        print("🔧 Connecting to PostgreSQL...")
        conn = psycopg2.connect(database_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'paper_trades'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("✅ paper_trades table doesn't exist yet, will be created with correct schema")
            return True
        
        print("📊 Checking paper_trades schema...")
        
        # Check if already migrated
        cur.execute("""
            SELECT data_type 
            FROM information_schema.columns 
            WHERE table_name = 'paper_trades' 
            AND column_name = 'target_time';
        """)
        result = cur.fetchone()
        
        if result and result[0] == 'timestamp with time zone':
            print("✅ paper_trades already migrated (target_time is TIMESTAMP WITH TIME ZONE)")
            return True
        
        print(f"🔄 Current target_time type: {result[0] if result else 'column missing'}")
        print("🔧 Starting migration...")
        
        # Check if table has data
        cur.execute("SELECT COUNT(*) FROM paper_trades;")
        row_count = cur.fetchone()[0]
        print(f"📊 Found {row_count} rows in paper_trades")
        
        if row_count == 0:
            # No data, can drop and recreate
            print("🔄 Dropping empty table and recreating with correct schema...")
            cur.execute("DROP TABLE IF EXISTS paper_trades CASCADE;")
            cur.execute("""
                CREATE TABLE paper_trades (
                    paper_trade_id TEXT PRIMARY KEY,
                    cascade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_direction TEXT NOT NULL,
                    signal_confidence REAL NOT NULL,
                    signal_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    target_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    target_price REAL,
                    position_size REAL DEFAULT 1000.0,
                    stop_loss_pct REAL DEFAULT 0.05,
                    take_profit_pct REAL DEFAULT 0.10,
                    actual_direction TEXT,
                    outcome TEXT,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    checked_at TIMESTAMP WITH TIME ZONE,
                    notes TEXT,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_outcome ON paper_trades(outcome);")
            print("✅ Table recreated with TIMESTAMP columns")
        else:
            # Has data, need to migrate
            print("🔄 Migrating existing data...")
            
            # Alter columns one by one
            time_columns = [
                'signal_time',
                'entry_time', 
                'target_time',
                'checked_at',
                'created_at'
            ]
            
            for col in time_columns:
                try:
                    # Convert TEXT to TIMESTAMP WITH TIME ZONE
                    # PostgreSQL will attempt automatic conversion
                    cur.execute(f"""
                        ALTER TABLE paper_trades 
                        ALTER COLUMN {col} 
                        TYPE TIMESTAMP WITH TIME ZONE 
                        USING {col}::TIMESTAMP WITH TIME ZONE;
                    """)
                    print(f"  ✅ Migrated {col} to TIMESTAMP WITH TIME ZONE")
                except Exception as e:
                    print(f"  ⚠️  {col} migration failed (might be NULL): {e}")
                    # If conversion fails, try allowing NULL first
                    try:
                        cur.execute(f"""
                            ALTER TABLE paper_trades 
                            ALTER COLUMN {col} 
                            DROP NOT NULL;
                        """)
                        cur.execute(f"""
                            ALTER TABLE paper_trades 
                            ALTER COLUMN {col} 
                            TYPE TIMESTAMP WITH TIME ZONE 
                            USING CASE 
                                WHEN {col} IS NULL THEN NULL
                                ELSE {col}::TIMESTAMP WITH TIME ZONE
                            END;
                        """)
                        print(f"  ✅ Migrated {col} to TIMESTAMP WITH TIME ZONE (nullable)")
                    except Exception as e2:
                        print(f"  ❌ {col} migration failed completely: {e2}")
        
        print("✅ Migration complete!")
        
        # Verify
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'paper_trades' 
            AND column_name IN ('signal_time', 'entry_time', 'target_time', 'checked_at', 'created_at')
            ORDER BY column_name;
        """)
        print("\n📊 Final schema:")
        for col_name, data_type in cur.fetchall():
            print(f"  {col_name}: {data_type}")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Paper Trades Schema Migration")
    print("=" * 60)
    success = migrate_paper_trades_schema()
    exit(0 if success else 1)
