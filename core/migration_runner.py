#!/usr/bin/env python3
"""
Ghost Protocol - Automatic Database Migration Runner
====================================================

Runs SQL migrations from migrations/ directory on startup.
Ensures personal watchlist tables exist in Postgres.

This runs automatically when wolf_app starts.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

LOGGER = logging.getLogger(__name__)


def run_migrations() -> Tuple[bool, List[str]]:
    """
    Run all pending migrations from migrations/ directory.
    
    Returns:
        (success: bool, messages: List[str])
    """
    messages = []
    
    try:
        from core.db_engine import get_db_connection, IS_POSTGRES
        
        if not IS_POSTGRES:
            msg = "[MIGRATION] Skipping migrations (SQLite mode - no migrations needed)"
            LOGGER.info(msg)
            messages.append(msg)
            return True, messages
        
        # Get migrations directory
        migrations_dir = Path(__file__).parent.parent / "migrations"
        if not migrations_dir.exists():
            msg = f"[MIGRATION] ⚠️  Migrations directory not found: {migrations_dir}"
            LOGGER.warning(msg)
            messages.append(msg)
            return True, messages  # Not critical - may not need migrations
        
        # Find all .sql migration files
        migration_files = sorted(migrations_dir.glob("*.sql"))
        if not migration_files:
            msg = "[MIGRATION] No SQL migration files found"
            LOGGER.info(msg)
            messages.append(msg)
            return True, messages
        
        # Execute each migration
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            for migration_file in migration_files:
                migration_name = migration_file.name
                
                # Check if this migration was already applied
                # (Simple approach: check if table exists)
                if "personal_watchlist" in migration_name.lower():
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public'
                            AND table_name = 'ghost_watchlist_items'
                        ) as exists
                    """)
                    result = cursor.fetchone()
                    # Handle both dict-like (RealDictCursor) and tuple access
                    if isinstance(result, dict):
                        table_exists = result.get('exists', False)
                    else:
                        table_exists = result[0] if result else False
                    
                    if table_exists:
                        msg = f"[MIGRATION] ✅ {migration_name} - already applied (table exists)"
                        LOGGER.info(msg)
                        messages.append(msg)
                        continue
                
                # Execute migration
                try:
                    sql = migration_file.read_text()
                    
                    # PostgreSQL psycopg2 can handle multiple statements in one execute()
                    # but we need to ensure the connection is in the right state
                    cursor.execute(sql)
                    conn.commit()
                    
                    msg = f"[MIGRATION] ✅ {migration_name} - applied successfully"
                    LOGGER.info(msg)
                    messages.append(msg)
                except Exception as e:
                    error_str = str(e)
                    # If error contains "already exists", it's OK (idempotent)
                    if "already exists" in error_str.lower():
                        msg = f"[MIGRATION] ✅ {migration_name} - already applied (idempotent)"
                        LOGGER.info(msg)
                        messages.append(msg)
                        conn.rollback()
                    else:
                        msg = f"[MIGRATION] ❌ {migration_name} - failed: {error_str}"
                        LOGGER.error(msg, exc_info=True)
                        messages.append(msg)
                        conn.rollback()
                        # Don't stop on first failure - continue with other migrations
                        continue
        
        return True, messages
        
    except Exception as e:
        # Handle ANY exception type (including KeyError, TypeError, etc.)
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else error_type
        msg = f"[MIGRATION] ❌ Migration runner failed: {error_msg}"
        LOGGER.error(msg, exc_info=True)
        messages.append(msg)
        return False, messages


def ensure_personal_watchlist_table() -> bool:
    """
    Ensure ghost_watchlist_items table exists.
    
    Returns:
        True if table exists or was created successfully
    """
    try:
        from core.db_engine import get_db_connection, IS_POSTGRES
        
        if not IS_POSTGRES:
            return True  # SQLite doesn't need this table
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name = 'ghost_watchlist_items'
                ) as exists
            """)
            result = cursor.fetchone()
            # Handle both dict-like (RealDictCursor) and tuple access
            if isinstance(result, dict):
                table_exists = result.get('exists', False)
            else:
                table_exists = result[0] if result else False
            
            if table_exists:
                LOGGER.info("[MIGRATION] ✅ ghost_watchlist_items table exists")
                return True
            else:
                LOGGER.warning("[MIGRATION] ⚠️  ghost_watchlist_items table missing - creating now...")
                # Create the table directly
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ghost_watchlist_items (
                        id BIGSERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        asset_type TEXT NOT NULL CHECK (asset_type IN ('crypto', 'stock')),
                        owns_position BOOLEAN DEFAULT FALSE,
                        notes TEXT DEFAULT '',
                        added_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        active BOOLEAN DEFAULT TRUE,
                        price_at_add REAL,
                        alert_threshold_pct REAL DEFAULT 5.0,
                        priority INTEGER DEFAULT 1,
                        CHECK (LENGTH(symbol) > 0 AND LENGTH(symbol) <= 20)
                    )
                """)
                conn.commit()
                LOGGER.info("[MIGRATION] ✅ ghost_watchlist_items table created")
                return True
                
    except Exception as e:
        LOGGER.error(f"[MIGRATION] ❌ Table check/create failed: {e}", exc_info=True)
        return False


def ensure_checkpoint_columns() -> bool:
    """
    Ensure paper_trades table has checkpoint columns for multi-checkpoint Trust Ladder.
    
    Columns added:
    - trust_level: INTEGER - The trust level when trade was logged
    - checkpoint_times: JSONB - Array of checkpoint timestamps
    - checkpoint_results: JSONB - Array of WIN/LOSS results per checkpoint
    - checkpoint_evaluated: JSONB - Array of booleans for evaluated status
    - checkpoint_prices: JSONB - Array of prices at each checkpoint
    
    Returns:
        True if columns exist or were created successfully
    """
    try:
        from core.db_engine import get_db_connection, IS_POSTGRES
        
        if not IS_POSTGRES:
            return True  # SQLite handles this in paper_tracker.py
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if checkpoint_times column exists (indicates migration already done)
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'paper_trades'
                    AND column_name = 'checkpoint_times'
                ) as exists
            """)
            result = cursor.fetchone()
            if isinstance(result, dict):
                column_exists = result.get('exists', False)
            else:
                column_exists = result[0] if result else False
            
            if column_exists:
                LOGGER.info("[MIGRATION] ✅ paper_trades checkpoint columns exist")
                return True
            
            # Add all checkpoint columns
            LOGGER.info("[MIGRATION] Adding checkpoint columns to paper_trades...")
            
            columns_to_add = [
                ("trust_level", "INTEGER DEFAULT 1"),
                ("checkpoint_times", "JSONB DEFAULT '[]'"),
                ("checkpoint_results", "JSONB DEFAULT '[]'"),
                ("checkpoint_evaluated", "JSONB DEFAULT '[]'"),
                ("checkpoint_prices", "JSONB DEFAULT '[]'"),
            ]
            
            for col_name, col_def in columns_to_add:
                try:
                    cursor.execute(f"""
                        ALTER TABLE paper_trades 
                        ADD COLUMN IF NOT EXISTS {col_name} {col_def}
                    """)
                    LOGGER.info(f"[MIGRATION] ✅ Added {col_name} column")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        LOGGER.info(f"[MIGRATION] ⏭️  {col_name} already exists")
                    else:
                        LOGGER.warning(f"[MIGRATION] ⚠️  {col_name}: {e}")
            
            conn.commit()
            LOGGER.info("[MIGRATION] ✅ paper_trades checkpoint columns ready")
            return True
                
    except Exception as e:
        LOGGER.error(f"[MIGRATION] ❌ Checkpoint columns migration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Standalone execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    success, messages = run_migrations()
    
    print("\n=== MIGRATION REPORT ===")
    for msg in messages:
        print(msg)
    
    if success:
        print("\n✅ All migrations completed successfully")
        exit(0)
    else:
        print("\n❌ Migration failed")
        exit(1)
