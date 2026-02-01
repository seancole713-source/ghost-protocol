"""
Database Migration: Add checkpoint columns to paper_trades table

This migration adds multi-checkpoint tracking columns for Trust Ladder:
- checkpoint_times: JSONB array of checkpoint timestamps
- checkpoint_results: JSONB array of WIN/LOSS results per checkpoint  
- checkpoint_evaluated: JSONB array of booleans for evaluated status
- checkpoint_prices: JSONB array of prices at each checkpoint

Run: python migrations/add_checkpoint_columns.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.getenv("DATABASE_URL")

def migrate():
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set - cannot run migration")
        return False
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        print("🔄 Adding checkpoint columns to paper_trades...")
        
        # Add trust_level column
        try:
            cur.execute("""
                ALTER TABLE paper_trades 
                ADD COLUMN IF NOT EXISTS trust_level INTEGER DEFAULT 1
            """)
            print("  ✅ Added trust_level column")
        except Exception as e:
            if "already exists" in str(e):
                print("  ⏭️  trust_level column already exists")
            else:
                print(f"  ⚠️  trust_level: {e}")
        
        # Add checkpoint_times column
        try:
            cur.execute("""
                ALTER TABLE paper_trades 
                ADD COLUMN IF NOT EXISTS checkpoint_times JSONB DEFAULT '[]'
            """)
            print("  ✅ Added checkpoint_times column")
        except Exception as e:
            if "already exists" in str(e):
                print("  ⏭️  checkpoint_times column already exists")
            else:
                print(f"  ⚠️  checkpoint_times: {e}")
        
        # Add checkpoint_results column
        try:
            cur.execute("""
                ALTER TABLE paper_trades 
                ADD COLUMN IF NOT EXISTS checkpoint_results JSONB DEFAULT '[]'
            """)
            print("  ✅ Added checkpoint_results column")
        except Exception as e:
            if "already exists" in str(e):
                print("  ⏭️  checkpoint_results column already exists")
            else:
                print(f"  ⚠️  checkpoint_results: {e}")
        
        # Add checkpoint_evaluated column
        try:
            cur.execute("""
                ALTER TABLE paper_trades 
                ADD COLUMN IF NOT EXISTS checkpoint_evaluated JSONB DEFAULT '[]'
            """)
            print("  ✅ Added checkpoint_evaluated column")
        except Exception as e:
            if "already exists" in str(e):
                print("  ⏭️  checkpoint_evaluated column already exists")
            else:
                print(f"  ⚠️  checkpoint_evaluated: {e}")
        
        # Add checkpoint_prices column
        try:
            cur.execute("""
                ALTER TABLE paper_trades 
                ADD COLUMN IF NOT EXISTS checkpoint_prices JSONB DEFAULT '[]'
            """)
            print("  ✅ Added checkpoint_prices column")
        except Exception as e:
            if "already exists" in str(e):
                print("  ⏭️  checkpoint_prices column already exists")
            else:
                print(f"  ⚠️  checkpoint_prices: {e}")
        
        conn.commit()
        
        # Verify columns exist
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'paper_trades'
            AND column_name IN ('trust_level', 'checkpoint_times', 'checkpoint_results', 'checkpoint_evaluated', 'checkpoint_prices')
            ORDER BY column_name
        """)
        
        columns = cur.fetchall()
        print("\n📋 Checkpoint columns in database:")
        for col_name, col_type in columns:
            print(f"  - {col_name}: {col_type}")
        
        if len(columns) == 5:
            print("\n✅ Migration SUCCESSFUL - All 5 checkpoint columns present")
        else:
            print(f"\n⚠️  Expected 5 columns, found {len(columns)}")
        
        cur.close()
        conn.close()
        return True
        
    except ImportError:
        print("❌ psycopg2 not installed")
        return False
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    migrate()
