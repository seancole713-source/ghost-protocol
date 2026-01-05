#!/usr/bin/env python3
"""
Migrate Ghost Predictions from SQLite to PostgreSQL
====================================================
One-time migration script to move prediction data from local SQLite
to Railway PostgreSQL for production use.

Usage:
    python scripts/migrate_sqlite_to_postgres.py [--dry-run]
"""

import os
import sys
import sqlite3
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# SQLite source databases
SQLITE_PREDICTIONS_DB = os.getenv("GHOST_PREDICT_DB", "./data/ghost_predictions.db")
SQLITE_WOLF_DB = "./data/wolf.db"

# PostgreSQL target
DATABASE_URL = os.getenv("DATABASE_URL", "")


def get_postgres_connection():
    """Get PostgreSQL connection."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def migrate_predictions(dry_run: bool = False):
    """Migrate predictions table from SQLite to PostgreSQL."""
    LOGGER.info(f"Migrating predictions from {SQLITE_PREDICTIONS_DB}...")
    
    # Connect to SQLite
    if not os.path.exists(SQLITE_PREDICTIONS_DB):
        LOGGER.warning(f"SQLite database not found: {SQLITE_PREDICTIONS_DB}")
        return 0
    
    sqlite_conn = sqlite3.connect(SQLITE_PREDICTIONS_DB)
    sqlite_conn.row_factory = sqlite3.Row
    
    # Get predictions
    cursor = sqlite_conn.execute("""
        SELECT id, symbol, run_at, horizon_h, method, confidence, direction, 
               features_json, params_json, tag
        FROM predictions
        ORDER BY id
    """)
    predictions = cursor.fetchall()
    LOGGER.info(f"Found {len(predictions)} predictions in SQLite")
    
    if dry_run:
        LOGGER.info("[DRY-RUN] Would migrate predictions")
        sqlite_conn.close()
        return len(predictions)
    
    # Connect to PostgreSQL
    pg_conn = get_postgres_connection()
    pg_cursor = pg_conn.cursor()
    
    # Migrate each prediction
    migrated = 0
    for pred in predictions:
        try:
            # Check if already exists (by original ID and run_at)
            pg_cursor.execute(
                "SELECT id FROM predictions WHERE symbol=%s AND run_at=%s",
                (pred['symbol'], pred['run_at'])
            )
            if pg_cursor.fetchone():
                continue  # Already migrated
            
            # Insert prediction
            pg_cursor.execute("""
                INSERT INTO predictions (symbol, run_at, horizon_h, method, confidence, direction, features_json, params_json, tag)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                pred['symbol'], pred['run_at'], pred['horizon_h'], pred['method'],
                pred['confidence'], pred['direction'], pred['features_json'],
                pred['params_json'], pred['tag']
            ))
            
            new_id = pg_cursor.fetchone()['id']
            
            # Migrate associated prediction_points
            points_cursor = sqlite_conn.execute(
                "SELECT ts, kind, price FROM prediction_points WHERE prediction_id=?",
                (pred['id'],)
            )
            points = points_cursor.fetchall()
            
            for point in points:
                pg_cursor.execute("""
                    INSERT INTO prediction_points (prediction_id, ts, kind, price)
                    VALUES (%s, %s, %s, %s)
                """, (new_id, point['ts'], point['kind'], point['price']))
            
            migrated += 1
            
            if migrated % 50 == 0:
                LOGGER.info(f"Migrated {migrated} predictions...")
                pg_conn.commit()
                
        except Exception as e:
            LOGGER.error(f"Error migrating prediction {pred['id']}: {e}")
            pg_conn.rollback()
    
    pg_conn.commit()
    LOGGER.info(f"✅ Migrated {migrated} predictions to PostgreSQL")
    
    sqlite_conn.close()
    pg_conn.close()
    
    return migrated


def migrate_outcomes(dry_run: bool = False):
    """Migrate outcomes table from SQLite to PostgreSQL."""
    LOGGER.info(f"Migrating outcomes from {SQLITE_PREDICTIONS_DB}...")
    
    if not os.path.exists(SQLITE_PREDICTIONS_DB):
        return 0
    
    sqlite_conn = sqlite3.connect(SQLITE_PREDICTIONS_DB)
    sqlite_conn.row_factory = sqlite3.Row
    
    # Check which schema we have
    cursor = sqlite_conn.execute("PRAGMA table_info(outcomes)")
    columns = [row[1] for row in cursor.fetchall()]
    LOGGER.info(f"SQLite outcomes columns: {columns}")
    
    # Old schema has: prediction_id, symbol, predicted_direction, actual_direction, etc.
    # New schema has: prediction_id, closed_at, mae, map, rmse, hit_direction, etc.
    
    if 'closed_at' in columns:
        # New schema
        cursor = sqlite_conn.execute("""
            SELECT prediction_id, closed_at, mae, map, rmse, hit_direction, hit_ratio_window, notes
            FROM outcomes
        """)
    else:
        # Old schema - need to transform
        cursor = sqlite_conn.execute("""
            SELECT prediction_id, symbol, predicted_direction, actual_direction, 
                   predicted_confidence, actual_price_change_pct, was_correct, evaluated_at
            FROM outcomes
        """)
    
    outcomes = cursor.fetchall()
    LOGGER.info(f"Found {len(outcomes)} outcomes in SQLite")
    
    if dry_run:
        LOGGER.info("[DRY-RUN] Would migrate outcomes")
        sqlite_conn.close()
        return len(outcomes)
    
    pg_conn = get_postgres_connection()
    pg_cursor = pg_conn.cursor()
    
    migrated = 0
    for outcome in outcomes:
        try:
            # Find corresponding prediction in PostgreSQL
            sqlite_pred = sqlite_conn.execute(
                "SELECT symbol, run_at FROM predictions WHERE id=?",
                (outcome['prediction_id'],)
            ).fetchone()
            
            if not sqlite_pred:
                continue
            
            pg_cursor.execute(
                "SELECT id FROM predictions WHERE symbol=%s AND run_at=%s",
                (sqlite_pred['symbol'], sqlite_pred['run_at'])
            )
            pg_pred = pg_cursor.fetchone()
            
            if not pg_pred:
                continue
            
            pg_pred_id = pg_pred['id']
            
            # Check if outcome already exists
            pg_cursor.execute(
                "SELECT prediction_id FROM outcomes WHERE prediction_id=%s",
                (pg_pred_id,)
            )
            if pg_cursor.fetchone():
                continue
            
            # Handle different schemas
            if 'closed_at' in columns:
                # New schema - direct insert
                pg_cursor.execute("""
                    INSERT INTO outcomes (prediction_id, closed_at, mae, map, rmse, hit_direction, hit_ratio_window, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    pg_pred_id, outcome['closed_at'], outcome['mae'], outcome['map'],
                    outcome['rmse'], outcome['hit_direction'], outcome['hit_ratio_window'], outcome['notes']
                ))
            else:
                # Old schema - transform to new
                hit_direction = 1 if outcome['was_correct'] else 0
                closed_at = float(outcome['evaluated_at'])
                mae = abs(outcome['actual_price_change_pct'])  # Approximate
                
                pg_cursor.execute("""
                    INSERT INTO outcomes (prediction_id, closed_at, mae, map, rmse, hit_direction, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    pg_pred_id, closed_at, mae, mae, mae, hit_direction, 
                    f"Migrated from old schema, was_correct={outcome['was_correct']}"
                ))
            
            migrated += 1
            
        except Exception as e:
            LOGGER.error(f"Error migrating outcome for prediction {outcome['prediction_id']}: {e}")
    
    pg_conn.commit()
    LOGGER.info(f"✅ Migrated {migrated} outcomes to PostgreSQL")
    
    sqlite_conn.close()
    pg_conn.close()
    
    return migrated


def verify_migration():
    """Verify migration completed successfully."""
    LOGGER.info("Verifying migration...")
    
    # Count in SQLite
    if os.path.exists(SQLITE_PREDICTIONS_DB):
        sqlite_conn = sqlite3.connect(SQLITE_PREDICTIONS_DB)
        sqlite_preds = sqlite_conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        sqlite_outcomes = sqlite_conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        sqlite_conn.close()
    else:
        sqlite_preds = 0
        sqlite_outcomes = 0
    
    # Count in PostgreSQL
    pg_conn = get_postgres_connection()
    pg_cursor = pg_conn.cursor()
    pg_cursor.execute("SELECT COUNT(*) as cnt FROM predictions")
    pg_preds = pg_cursor.fetchone()['cnt']
    pg_cursor.execute("SELECT COUNT(*) as cnt FROM outcomes")
    pg_outcomes = pg_cursor.fetchone()['cnt']
    pg_conn.close()
    
    LOGGER.info(f"SQLite: {sqlite_preds} predictions, {sqlite_outcomes} outcomes")
    LOGGER.info(f"PostgreSQL: {pg_preds} predictions, {pg_outcomes} outcomes")
    
    if pg_preds >= sqlite_preds:
        LOGGER.info("✅ PASS: PostgreSQL has all predictions")
    else:
        LOGGER.warning(f"⚠️ PostgreSQL missing {sqlite_preds - pg_preds} predictions")
    
    return pg_preds >= sqlite_preds


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite predictions to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without doing it")
    args = parser.parse_args()
    
    if not DATABASE_URL:
        LOGGER.error("DATABASE_URL environment variable not set")
        LOGGER.info("Set it with: export DATABASE_URL='postgresql://...'")
        sys.exit(1)
    
    LOGGER.info("=" * 60)
    LOGGER.info("Ghost SQLite → PostgreSQL Migration")
    LOGGER.info("=" * 60)
    
    if args.dry_run:
        LOGGER.info("DRY-RUN MODE - No changes will be made")
    
    predictions_migrated = migrate_predictions(args.dry_run)
    outcomes_migrated = migrate_outcomes(args.dry_run)
    
    LOGGER.info("=" * 60)
    LOGGER.info(f"Migration Summary:")
    LOGGER.info(f"  Predictions: {predictions_migrated}")
    LOGGER.info(f"  Outcomes: {outcomes_migrated}")
    
    if not args.dry_run:
        verify_migration()
    
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
